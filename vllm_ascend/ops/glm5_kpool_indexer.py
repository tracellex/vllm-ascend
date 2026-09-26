# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatcher for the GLM-Next KPool lightning indexer.

Routes between the fused AscendC operator (``npu_glm5_kpool_indexer``: paged
pool gather + cube matmul + in-kernel radix top-k + pool-to-token expansion +
causal tail) and the Triton fallback (``glm5_next_lightning_indexer_triton``).
The public signature matches the Triton wrapper exactly so the model-side
backend can switch implementations with a one-line change.

Selection is controlled by ``VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL``:
``auto`` (default) prefers the AscendC operator when the custom op is loaded
and the shapes are supported, falling back to Triton with a one-time warning;
``triton`` / ``ascendc`` force one path (the latter raises if unavailable).
"""

from __future__ import annotations

import torch

from vllm_ascend.envs import VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL

from .triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton

_warned_fallback = False


def _ascendc_available() -> bool:
    try:
        torch.ops._C_ascend.npu_glm5_kpool_indexer  # noqa: B018
        return True
    except (AttributeError, RuntimeError):
        return False


def _compute_qbar(query: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Head-weighted query [T, heads, dim] + [T, heads] -> [T, dim] bf16.

    Rows are zero-padded to a multiple of 32 so full cube tiles stay in
    bounds; padded rows are skipped by the kernel on the vector side.
    """
    qbar = (query.float() * weights.float().unsqueeze(-1)).sum(dim=1).to(query.dtype)
    num_tokens = qbar.shape[0]
    pad_rows = (-num_tokens) % 32
    if pad_rows:
        qbar = torch.nn.functional.pad(qbar, (0, 0, 0, pad_rows))
    return qbar.contiguous()


def _pad_positions(positions: torch.Tensor) -> torch.Tensor:
    pad_rows = (-positions.shape[0]) % 32
    if pad_rows:
        return torch.nn.functional.pad(positions, (0, pad_rows), value=0).contiguous()
    return positions


def _gather8_lens(x: torch.Tensor) -> torch.Tensor:
    """Mirror a [B] lens vector to stride 8: element i lands at byte offset
    32*i, i.e. every element 32B-aligned. The AIV scalar GM read compiles to
    a vector-granularity access and faults at unaligned offsets, so the kernel
    reads these arrays as ``GetValue(idx * 8)`` on the mirrored layout."""
    # Pad to a multiple of 32 elements (128B): the AICore scalar GM read
    # fetches a full 128B block, so a shorter tail block would read past the
    # tensor (observed as probabilistic hangs AND faults on both AIC/AIV).
    n = x.shape[0] * 8
    n = ((n + 31) // 32) * 32
    out = torch.zeros(n, dtype=torch.int32, device=x.device)
    out[0::8][: x.shape[0]] = x.to(torch.int32)
    return out.contiguous()


def _expand_pool_ids(pool_ids: torch.Tensor, positions: torch.Tensor, index_topk: int,
                     index_kpool: int) -> torch.Tensor:
    """Pool ids -> token indices, mirroring the Triton wrapper post-processing.

    history: pid*kpool + {0..kpool-1}, -1 padded to index_topk columns; the
    causal tail (kpool-1 columns) mirrors the Triton operator's fixed tail
    block (append_causal_tail on the model side is idempotent over it).
    """
    num_tokens = pool_ids.shape[0]
    device = pool_ids.device
    pool_topk = pool_ids.shape[1]
    lane = torch.arange(index_kpool, device=device, dtype=pool_ids.dtype)
    hist = pool_ids.unsqueeze(-1) * index_kpool + lane  # [T, poolTopk, kpool]
    hist = torch.where((pool_ids < 0).unsqueeze(-1), torch.full_like(hist, -1), hist)
    hist = hist.reshape(num_tokens, index_topk)
    if pool_topk * index_kpool < index_topk:
        hist = torch.nn.functional.pad(hist, (0, index_topk - pool_topk * index_kpool), value=-1)

    tail_cols = torch.arange(index_kpool - 1, device=device)
    tail_start = (positions + 1) // index_kpool * index_kpool
    tail = torch.where(tail_cols.unsqueeze(0) < (positions + 1 - tail_start).unsqueeze(1),
                       tail_start.unsqueeze(1) + tail_cols.unsqueeze(0),
                       torch.full((num_tokens, index_kpool - 1), -1, dtype=torch.long, device=device))
    return torch.cat([hist, tail.to(hist.dtype)], dim=1).view(num_tokens, 1, -1)


def _visible_pool_lengths(
    positions: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    """Return each token's visible pool count for packed multi-request input."""
    token_ids = torch.arange(positions.shape[0], device=positions.device, dtype=cum_query_lens.dtype)
    request_ids = torch.searchsorted(cum_query_lens, token_ids, right=True)
    request_ids = request_ids.clamp_max(indexer_seq_lens.shape[0] - 1).to(torch.int64)
    request_pool_lens = indexer_seq_lens.index_select(0, request_ids).clamp_min(0).to(positions.dtype)
    visible = torch.minimum((positions + 1) // index_kpool, request_pool_lens)
    return visible.clamp_max(max_pool_seq_len)


def glm5_kpool_indexer(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    """Select sparse token indices; returns [T, 1, index_topk + kpool - 1] int32."""
    global _warned_fallback

    impl = VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL
    # Corner shapes stay on the Triton wrapper: it has the tail-only / empty
    # early-exit paths and there is nothing for the fused operator to win.
    corner_shape = query.shape[0] == 0 or max_pool_seq_len == 0
    if corner_shape:
        use_ascendc = False
    elif impl in ("auto", "ascendc"):
        use_ascendc = True
    else:
        if impl != "triton":
            import logging

            logging.getLogger(__name__).warning(
                "Unknown VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL=%r; expected "
                "auto/triton/ascendc. Using the Triton path.", impl,
            )
        use_ascendc = False

    if use_ascendc:
        if not _ascendc_available():
            if impl == "ascendc":
                raise RuntimeError(
                    "VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL=ascendc but the "
                    "npu_glm5_kpool_indexer custom op is not loaded; build the "
                    "custom kernels or switch the selector to auto/triton."
                )
            if not _warned_fallback:
                _warned_fallback = True
                import logging

                logging.getLogger(__name__).warning(
                    "npu_glm5_kpool_indexer unavailable; falling back to the "
                    "Triton lightning indexer path."
                )
            return glm5_next_lightning_indexer_triton(
                query, indexer_cache, weights, cum_query_lens, indexer_seq_lens,
                indexer_block_table, positions, index_topk=index_topk,
                index_kpool=index_kpool, max_pool_seq_len=max_pool_seq_len,
            )
        qbar = _compute_qbar(query, weights.to(query.dtype))
        positions_pad = _pad_positions(positions.to(torch.int32))
        pool_ids, _ = torch.ops._C_ascend.npu_glm5_kpool_indexer(
            qbar,
            indexer_cache,
            _gather8_lens(cum_query_lens),
            _gather8_lens(indexer_seq_lens),
            indexer_block_table,
            positions_pad,
            index_topk,
            index_kpool,
            query.shape[2],
            max_pool_seq_len,
            0,
        )
        pool_ids = pool_ids[:query.shape[0], 0]  # [T, poolTopk]
        # kernel ships raw ids: mask sentinel / beyond-visibility lanes here.
        visible = _visible_pool_lengths(
            positions,
            cum_query_lens,
            indexer_seq_lens,
            index_kpool,
            max_pool_seq_len,
        )
        pool_ids = torch.where(pool_ids < visible.unsqueeze(1), pool_ids,
                               torch.full_like(pool_ids, -1))
        return _expand_pool_ids(pool_ids, positions, index_topk, index_kpool)

    return glm5_next_lightning_indexer_triton(
        query, indexer_cache, weights, cum_query_lens, indexer_seq_lens,
        indexer_block_table, positions, index_topk=index_topk,
        index_kpool=index_kpool, max_pool_seq_len=max_pool_seq_len,
    )
