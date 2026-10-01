# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the glm5_kpool_indexer dispatcher PR#17542 contract.

The engine call site (sparse_attn_indexer_kpool) passes output_buffer /
pack_tail / allow_cache_packing since PR#17542; the dispatcher used to miss
them and every IMPL crashed at cudagraph capture with TypeError. These
tests pin the signature and the AscendC-path post-processing semantics on
CPU by stubbing the custom ops and the Triton wrapper (no NPU required).
"""

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops.glm5_kpool_indexer as dispatcher


def _install_fake_kernels(monkeypatch, impl: str, pool_ids_fn):
    """Point the dispatcher at a fake env value and fake AscendC kernels."""
    triton_stub = ModuleType("triton_stub")
    triton_stub.glm5_next_lightning_indexer_triton = MagicMock(
        side_effect=AssertionError("triton path must not run in these tests")
    )
    fake_c = SimpleNamespace(
        npu_glm5_kpool_split_aic=lambda *a, **k: None,
        npu_glm5_kpool_split_aiv=pool_ids_fn,
        npu_glm5_kpool_indexer=lambda *a, **k: None,
    )

    class OpsProxy:
        def __getattr__(self, item):
            if item == "_C_ascend":
                return fake_c
            return getattr(torch._C._get_builtin_ops() and torch.ops, item)

    monkeypatch.setattr(dispatcher, "VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL", impl)
    monkeypatch.setattr(
        dispatcher, "glm5_next_lightning_indexer_triton", triton_stub.glm5_next_lightning_indexer_triton
    )
    monkeypatch.setattr(dispatcher, "_ascendc_available", lambda: True)
    monkeypatch.setattr(dispatcher.torch, "ops", OpsProxy())
    return triton_stub.glm5_next_lightning_indexer_triton


def _pool_ids_fill(*args, **kwargs):
    """Fake AIV output: pool id = row % 16 for lane 0, -1 elsewhere."""
    out = args[-1]
    out.fill_(-1)
    out[:, :, 0] = torch.arange(out.shape[0], dtype=torch.int32).remainder(16).unsqueeze(1)
    return out


def _inputs(T: int, P: int):
    query = torch.randn(T, 2, 128, dtype=torch.bfloat16)
    weights = torch.ones(T, 2, dtype=torch.bfloat16)
    cache = torch.randn(8, 16, 1, 128, dtype=torch.bfloat16)
    cum = torch.tensor([T], dtype=torch.int32)
    seqs = torch.tensor([P], dtype=torch.int32)
    table = torch.zeros(1, 8, dtype=torch.int32)
    positions = torch.arange(T, dtype=torch.int64) * 2 + 1
    return query, cache, weights, cum, seqs, table, positions


def test_dispatcher_accepts_pr17542_kwargs_triton(monkeypatch):
    """Signature acceptance on the triton IMPL (wrapper stubbed)."""
    mock = _install_fake_kernels(monkeypatch, "triton", _pool_ids_fill)
    args = _inputs(4, 128)
    buffer = torch.zeros(4, 2051, dtype=torch.int32)
    with pytest.raises(AssertionError):
        dispatcher.glm5_kpool_indexer(
            *args, index_topk=2048, index_kpool=4, max_pool_seq_len=128,
            output_buffer=buffer, pack_tail=True, allow_cache_packing=True,
        )
    # the kwargs reached the wrapper
    _, kwargs = mock.call_args
    assert kwargs["output_buffer"] is buffer and kwargs["pack_tail"] is True


def test_ascendc_packed_tail_and_buffer_write_through(monkeypatch):
    _install_fake_kernels(monkeypatch, "ascendc_group_topk_split", _pool_ids_fill)
    T, P = 4, 256
    args = _inputs(T, P)
    buffer = torch.full((T, 2051), 7, dtype=torch.int32)

    out = dispatcher.glm5_kpool_indexer(
        *args, index_topk=2048, index_kpool=4, max_pool_seq_len=P,
        output_buffer=buffer, pack_tail=True, allow_cache_packing=False,
    )
    assert out.shape == (T, 1, 2051)
    assert out.data_ptr() == buffer.data_ptr(), "must return a view of the capture buffer"
    assert not (out[:, 0, :] == 7).any(), "whole width must be written"
    # positions = [1, 3, 5, 7]; history lanes all masked (visible <= lane id):
    #   pos=1 -> tail_start=0, count=2, tail [0,1] at cols 0,1
    #   pos=3 -> tail_start=4, count=0, no tail (pool 0 complete)
    #   pos=5 -> tail_start=4, count=2, tail [4,5] at cols 4,5
    #   pos=7 -> tail_start=8, count=0, no tail (pool 1 complete)
    assert out[0, 0, :4].tolist() == [0, 1, -1, -1]
    assert out[1, 0, :8].tolist() == [-1, -1, -1, -1, -1, -1, -1, -1]
    assert out[2, 0, :8].tolist() == [-1, -1, -1, -1, 4, 5, -1, -1]
    assert out[3, 0, :8].tolist() == [-1, -1, -1, -1, -1, -1, -1, -1]


def test_ascendc_masks_padded_rows(monkeypatch):
    _install_fake_kernels(monkeypatch, "ascendc_group_topk_split", _pool_ids_fill)
    T, P = 8, 256
    query, cache, weights, cum, seqs, table, positions = _inputs(T, P)
    cum = torch.tensor([3, 6], dtype=torch.int32)  # 6 live tokens, rows 6-7 padded
    out = dispatcher.glm5_kpool_indexer(
        query, cache, weights, cum, seqs, table, positions,
        index_topk=2048, index_kpool=4, max_pool_seq_len=P,
        output_buffer=None, pack_tail=True, allow_cache_packing=True,
    )
    assert (out[6:, 0, :] == -1).all(), "padded rows must be fully -1"
    # live rows keep their packed tail: row 0 pos=1 -> tail token 0 at col 0
    assert out[0, 0, 0].item() == 0


def test_ascendc_rejects_bad_output_buffer(monkeypatch):
    _install_fake_kernels(monkeypatch, "ascendc_group_topk_split", _pool_ids_fill)
    args = _inputs(2, 64)
    bad = torch.zeros(2, 16, dtype=torch.int64)  # wrong dtype + too narrow
    with pytest.raises(ValueError):
        dispatcher.glm5_kpool_indexer(
            *args, index_topk=2048, index_kpool=4, max_pool_seq_len=64,
            output_buffer=bad, pack_tail=True, allow_cache_packing=True,
        )
