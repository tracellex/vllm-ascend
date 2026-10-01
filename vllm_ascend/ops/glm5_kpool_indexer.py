# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatcher for the GLM-Next KPool lightning indexer.

Routes between the fused AscendC operator (``npu_glm5_kpool_indexer``: paged
pool gather + cube matmul + in-kernel radix top-k + pool-to-token expansion +
causal tail) and the Triton fallback (``glm5_next_lightning_indexer_triton``).
The public signature matches the Triton wrapper exactly so the model-side
backend can switch implementations with a one-line change.

Selection is controlled by ``VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL``:
``triton`` (default) keeps the validated fallback active; ``auto`` prefers the
fused AscendC operator when available; ``ascendc`` forces the fused operator;
``ascendc_group_topk`` selects the experimental 32-row A3 grouped path;
``ascendc_group_topk_m64`` selects its 64-row Cube-tile candidate.
Explicit AscendC selections raise if their required custom operator is unavailable.
"""

from __future__ import annotations

import torch

from vllm_ascend.envs import VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL

from .triton.glm5_next_lightning_indexer import glm5_next_lightning_indexer_triton

_GROUP_TOPK_OUTPUT_MODE = 3
_GROUP_TOPK_M64_OUTPUT_MODE = 4
_KERNEL_ROW_ALIGNMENT = 128

_warned_fallback = False


def _ascendc_available() -> bool:
    try:
        torch.ops._C_ascend.npu_glm5_kpool_indexer  # noqa: B018
        return True
    except (AttributeError, RuntimeError):
        return False


def _compute_qbar(query: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Head-weighted query [T, heads, dim] + [T, heads] -> [T, 2*dim] bf16.

    The FP32 qbar is split as [q_hi | q_lo] bf16 halves per row (H9): a
    single bf16 rounding costs 8 mantissa bits and measurably flips top-k
    boundaries against the FP32 Triton reference, while the two-half split
    keeps ~16 bits via the cube's FP32-accumulated dual Mmad.
    Rows are zero-padded to a multiple of 128 so both arch22 (32/64-row)
    and arch35 (128-row) cube tiles stay in
    bounds; padded rows are skipped by the kernel on the vector side.
    """
    qbar = (query.float() * weights.float().unsqueeze(-1)).sum(dim=1)
    q_hi = qbar.to(query.dtype)
    q_lo = (qbar - q_hi.float()).to(query.dtype)
    qbar2 = torch.cat([q_hi, q_lo], dim=1)
    num_tokens = qbar2.shape[0]
    pad_rows = (-num_tokens) % _KERNEL_ROW_ALIGNMENT
    if pad_rows:
        qbar2 = torch.nn.functional.pad(qbar2, (0, 0, 0, pad_rows))
    return qbar2.contiguous()


def _pad_positions(positions: torch.Tensor) -> torch.Tensor:
    pad_rows = (-positions.shape[0]) % _KERNEL_ROW_ALIGNMENT
    if pad_rows:
        return torch.nn.functional.pad(positions, (0, pad_rows), value=0).contiguous()
    return positions


def _supports_ascendc_config(index_topk: int, index_kpool: int) -> bool:
    return index_topk == 2048 and index_kpool == 4


def _gather8_lens(x: torch.Tensor) -> torch.Tensor:
    """Mirror a [N] int32 vector to stride 8: element i lands at byte offset
    32*i, i.e. every element 32B-aligned. The AIV scalar GM read compiles to
    a vector-granularity access and faults at unaligned offsets, so the kernel
    reads these arrays as ``GetValue(idx * 8)`` on the mirrored layout."""
    out = torch.zeros(x.shape[0] * 8, dtype=torch.int32, device=x.device)
    out[0::8] = x.to(torch.int32)
    return out.contiguous()


def _gather8_positions(positions_pad: torch.Tensor) -> torch.Tensor:
    """Mirror padded int32 positions [T_pad] to the same stride-8 layout.

    The split AIV kernel reads per-row causal positions scalar-wise from GM
    (``GetValue(row * 8)``); absolute positions — not row-in-request — drive
    visibility, so prefix-cache hits and decode select the right pool window
    (defect t_e3dfea56). dim0 stays T_pad to keep the op's infer-shape view
    of the token count unchanged."""
    out = torch.zeros(positions_pad.shape[0] * 8, dtype=torch.int32, device=positions_pad.device)
    out[0::8] = positions_pad
    return out.contiguous()


def _expand_pool_ids(pool_ids: torch.Tensor, positions: torch.Tensor, index_topk: int,
                     index_kpool: int, *, pack_tail: bool = False) -> torch.Tensor:
    """Pool ids -> token indices, mirroring the Triton wrapper post-processing.

    history: pid*kpool + {0..kpool-1}, -1 padded to index_topk columns; with
    ``pack_tail`` the causal tail (kpool-1 columns) is scattered directly after
    the last valid history column (mirroring the Triton operator's packed tail
    and the model-side ``append_causal_tail``), otherwise it keeps the fixed
    tail block starting at column index_topk.
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
    tail = tail.to(hist.dtype)
    if pack_tail:
        # PR#17542 packed layout: the tail follows the last valid history
        # column (min(tail_start, topk)) instead of a fixed top-k offset, so
        # short requests expose their unpooled tokens without -1 holes.
        tail_slots = tail_start.clamp_max(index_topk).unsqueeze(1) + tail_cols.unsqueeze(0)
        packed = torch.cat(
            [hist, torch.full((num_tokens, index_kpool - 1), -1, dtype=hist.dtype, device=device)],
            dim=1,
        )
        packed.scatter_(1, tail_slots, tail)
        return packed.view(num_tokens, 1, -1)
    return torch.cat([hist, tail], dim=1).view(num_tokens, 1, -1)


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


def _validated_output_buffer(
    output_buffer: torch.Tensor | None,
    query: torch.Tensor,
    num_tokens: int,
    output_width: int,
) -> torch.Tensor | None:
    """Mirror the Triton wrapper's output-buffer contract (dtype/layout/size)."""
    if output_buffer is None:
        return None
    if (
        output_buffer.ndim != 2
        or output_buffer.dtype != torch.int32
        or output_buffer.device != query.device
        or output_buffer.stride(1) != 1
        or (num_tokens > 1 and output_buffer.stride(0) < output_buffer.shape[1])
    ):
        raise ValueError("GLM KPool output buffer requires int32 contiguous columns on the query device.")
    if output_buffer.shape[0] < num_tokens or output_buffer.shape[1] < output_width:
        raise ValueError("GLM KPool output buffer must cover the token rows and top-k width.")
    return output_buffer


def _return_indices(
    indices: torch.Tensor,
    cum_query_lens: torch.Tensor,
    output_buffer: torch.Tensor | None,
) -> torch.Tensor:
    """Return [T, 1, W] indices, writing through to the caller's buffer if set.

    FULL-graph batches carry padded token rows; the wrapper masks them to -1
    in-kernel (live = token < query_ends[-1]) and PR#17542 removed the
    model-side masked_fill_ that used to do it — mirror it here so both
    implementations stay value-identical. With a capture buffer the final
    rows are copied into it and its view returned; addresses stay stable
    across replays.
    """
    num_tokens = indices.shape[0]
    live = torch.arange(num_tokens, device=indices.device) < cum_query_lens[-1]
    indices.masked_fill_(~live[:, None, None], -1)
    if output_buffer is None:
        return indices
    _, _, output_width = indices.shape
    output_buffer[:num_tokens, :output_width].copy_(indices[:, 0, :])
    return output_buffer[:num_tokens, :output_width].unsqueeze(1)


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
    output_buffer: torch.Tensor | None = None,
    pack_tail: bool = False,
    allow_cache_packing: bool = True,
) -> torch.Tensor:
    """Select sparse token indices; returns [T, 1, index_topk + kpool - 1] int32.

    Keyword arguments match the Triton wrapper (PR#17542 contract):
    ``output_buffer`` captures the [T, W] int32 output for cudagraph FULL
    replay (both impls write through when provided); ``pack_tail`` scatters
    the causal tail right after the valid history (consumed by the Triton
    kernel and the AscendC expansion); ``allow_cache_packing`` only affects
    the Triton path — the AscendC kernels always read pages directly.
    """
    global _warned_fallback

    impl = VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL
    # Corner shapes stay on the Triton wrapper: it has the tail-only / empty
    # early-exit paths and there is nothing for the fused operator to win.
    corner_shape = query.shape[0] == 0 or max_pool_seq_len == 0
    if corner_shape:
        use_ascendc = False
    elif impl in ("auto", "ascendc", "ascendc_group_topk", "ascendc_group_topk_m64",
                  "ascendc_group_topk_split"):
        use_ascendc = True
    else:
        if impl != "triton":
            import logging

            logging.getLogger(__name__).warning(
                "Unknown VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL=%r; expected "
                "auto/triton/ascendc/ascendc_group_topk/ascendc_group_topk_m64. "
                "Using the Triton path.", impl,
            )
        use_ascendc = False

    if use_ascendc and not _supports_ascendc_config(index_topk, index_kpool):
        if impl == "auto":
            use_ascendc = False
        else:
            raise ValueError(
                "The AscendC GLM5 KPool indexer requires index_topk=2048 "
                "and index_kpool=4."
            )

    if use_ascendc:
        if not _ascendc_available():
            if impl in ("ascendc", "ascendc_group_topk", "ascendc_group_topk_m64",
                        "ascendc_group_topk_split"):
                raise RuntimeError(
                    f"VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL={impl} but the "
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
                output_buffer=output_buffer, pack_tail=pack_tail,
                allow_cache_packing=allow_cache_packing,
            )
        # Both AscendC paths materialize their outputs before the expansion, so
        # validate the capture buffer once up front (mirror of the wrapper's
        # checks) — a later copy_ into a wrong-dtype buffer would silently cast.
        _validated_output_buffer(output_buffer, query, query.shape[0],
                                 index_topk + index_kpool - 1)
        qbar = _compute_qbar(query, weights.to(query.dtype))
        positions_pad = _pad_positions(positions.to(torch.int32))
        if impl == "ascendc_group_topk_split":
            # De-mixed launch pair (pure AIC + pure AIV per 4096-pool batch) so
            # the indexer co-schedules with the HCCL stream instead of excluding
            # it like the fused MIX kernel; see kernel_split.h for the contract.
            pool_topk = index_topk // index_kpool
            t_pad = qbar.shape[0]
            # The AIC's fixpipe always writes a full M_BLOCK=32-row tile, so
            # the AIC-owned staging tensors must be tile-padded (the fused
            # kernel got this for free from its tiling-allocated workspace).
            t_rows = (t_pad + 31) // 32 * 32
            n_batches = (max_pool_seq_len + 4095) // 4096
            scores = torch.empty((t_rows, 4096), dtype=torch.float32, device=qbar.device)
            strip = torch.empty((t_rows, pool_topk * 2), dtype=torch.float32, device=qbar.device)
            pool_ids = torch.empty((t_pad, 1, pool_topk), dtype=torch.int32, device=qbar.device)
            gather_cum = _gather8_lens(cum_query_lens)
            gather_seq = _gather8_lens(indexer_seq_lens)
            # Absolute positions on the same stride-8 mirror: the AIV folds
            # causal visibility per row from these (t_e3dfea56).
            gather_pos = _gather8_positions(positions_pad)
            for batch in range(n_batches):
                torch.ops._C_ascend.npu_glm5_kpool_split_aic(
                    qbar, indexer_cache, gather_cum, gather_seq,
                    indexer_block_table, gather_pos,
                    index_topk, index_kpool, query.shape[2], max_pool_seq_len, batch, scores)
                torch.ops._C_ascend.npu_glm5_kpool_split_aiv(
                    gather_cum, gather_seq, gather_pos, scores, strip,
                    index_topk, index_kpool, query.shape[2], max_pool_seq_len, batch, pool_ids)
            del scores, strip
            pool_ids = pool_ids[:query.shape[0], 0]
            visible = _visible_pool_lengths(
                positions, cum_query_lens, indexer_seq_lens, index_kpool, max_pool_seq_len)
            pool_ids = torch.where(pool_ids < visible.unsqueeze(1), pool_ids,
                                   torch.full_like(pool_ids, -1))
            indices = _expand_pool_ids(pool_ids, positions, index_topk, index_kpool,
                                       pack_tail=pack_tail)
            return _return_indices(indices, cum_query_lens, output_buffer)
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
            (_GROUP_TOPK_OUTPUT_MODE if impl == "ascendc_group_topk" else
             _GROUP_TOPK_M64_OUTPUT_MODE if impl == "ascendc_group_topk_m64" else 0),
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
        indices = _expand_pool_ids(pool_ids, positions, index_topk, index_kpool,
                                   pack_tail=pack_tail)
        return _return_indices(indices, cum_query_lens, output_buffer)

    return glm5_next_lightning_indexer_triton(
        query, indexer_cache, weights, cum_query_lens, indexer_seq_lens,
        indexer_block_table, positions, index_topk=index_topk,
        index_kpool=index_kpool, max_pool_seq_len=max_pool_seq_len,
        output_buffer=output_buffer, pack_tail=pack_tail,
        allow_cache_packing=allow_cache_packing,
    )
