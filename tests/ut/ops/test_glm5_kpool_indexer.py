# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.ops.glm5_kpool_indexer import (
    _expand_pool_ids,
    _pad_positions,
    _supports_ascendc_config,
    _visible_pool_lengths,
)


def test_ascendc_config_matches_fixed_glm5_pool_contract() -> None:
    assert _supports_ascendc_config(index_topk=2048, index_kpool=4)
    assert not _supports_ascendc_config(index_topk=4096, index_kpool=8)
    assert not _supports_ascendc_config(index_topk=2048, index_kpool=8)


def test_visible_pool_lengths_map_tokens_to_requests() -> None:
    positions = torch.tensor([0, 3, 7, 0, 7, 15], dtype=torch.int64)
    query_ends = torch.tensor([3, 6], dtype=torch.int32)
    pool_lens = torch.tensor([9, 2], dtype=torch.int32)

    visible = _visible_pool_lengths(positions, query_ends, pool_lens, index_kpool=4, max_pool_seq_len=8)

    torch.testing.assert_close(visible, torch.tensor([0, 1, 2, 0, 2, 2], dtype=torch.int64))


def test_visible_pool_lengths_clamp_graph_padding_to_last_request() -> None:
    positions = torch.tensor([0, 3, 7, 11, 15], dtype=torch.int64)
    query_ends = torch.tensor([2, 4], dtype=torch.int32)
    pool_lens = torch.tensor([8, 3], dtype=torch.int32)

    visible = _visible_pool_lengths(positions, query_ends, pool_lens, index_kpool=4, max_pool_seq_len=2)

    torch.testing.assert_close(visible, torch.tensor([0, 1, 2, 2, 2], dtype=torch.int64))


def test_expand_pool_ids_preserves_padding_and_appends_tail() -> None:
    pool_ids = torch.tensor([[3, 1, -1], [2, -1, -1]], dtype=torch.int32)
    positions = torch.tensor([9, 11], dtype=torch.int64)

    output = _expand_pool_ids(pool_ids, positions, index_topk=12, index_kpool=4)

    expected = torch.tensor(
        [
            [12, 13, 14, 15, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, -1],
            [8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    ).view(2, 1, 15)
    torch.testing.assert_close(output, expected)


def test_expand_empty_pool_ids_returns_only_causal_tail() -> None:
    pool_ids = torch.full((2, 512), -1, dtype=torch.int32)
    positions = torch.tensor([0, 5], dtype=torch.int64)

    output = _expand_pool_ids(pool_ids, positions, index_topk=2048, index_kpool=4)

    assert output.shape == (2, 1, 2051)
    torch.testing.assert_close(output[0, 0, :2048], torch.full((2048,), -1, dtype=torch.int32))
    torch.testing.assert_close(output[0, 0, 2048:], torch.tensor([0, -1, -1], dtype=torch.int32))
    torch.testing.assert_close(output[1, 0, :2048], torch.full((2048,), -1, dtype=torch.int32))
    torch.testing.assert_close(output[1, 0, 2048:], torch.tensor([4, 5, -1], dtype=torch.int32))


def test_pad_positions_aligns_for_largest_kernel_tile() -> None:
    positions = torch.arange(129, dtype=torch.int32)

    padded = _pad_positions(positions)

    assert padded.shape == (256,)
    torch.testing.assert_close(padded[:129], positions)
    torch.testing.assert_close(padded[129:], torch.zeros(127, dtype=torch.int32))
