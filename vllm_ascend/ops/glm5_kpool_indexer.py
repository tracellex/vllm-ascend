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
    """Head-weighted query [T, heads, dim] + [T, heads] -> [T, dim] bf16."""
    qbar = (query.float() * weights.float().unsqueeze(-1)).sum(dim=1)
    return qbar.to(query.dtype).contiguous()


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
        indices, _ = torch.ops._C_ascend.npu_glm5_kpool_indexer(
            qbar,
            indexer_cache,
            cum_query_lens,
            indexer_seq_lens,
            indexer_block_table,
            positions,
            index_topk,
            index_kpool,
            query.shape[2],
            max_pool_seq_len,
            0,
        )
        return indices

    return glm5_next_lightning_indexer_triton(
        query, indexer_cache, weights, cum_query_lens, indexer_seq_lens,
        indexer_block_table, positions, index_topk=index_topk,
        index_kpool=index_kpool, max_pool_seq_len=max_pool_seq_len,
    )
