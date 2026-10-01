# SPDX-License-Identifier: Apache-2.0
"""Verify that the arch22 score path follows the paged block table."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from importlib import import_module

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import_module("vllm_ascend.vllm_ascend_C")
from vllm_ascend.ops import glm5_kpool_indexer as G  # noqa: E402


TOKENS = 4096
HEAD_DIM = 128
POOLS_PER_BLOCK = 32
NUM_BLOCKS = 32
NUM_POOLS = POOLS_PER_BLOCK * NUM_BLOCKS


def build_encoded_inputs(device: str, hi: float, lo: float):
    """qbar packs [q_hi | q_lo] bf16 halves (H9). The (hi, lo) pair doubles
    as an accumulate discriminator: hi=lo=1 must double the encoded scores —
    a second Mmad that overwrites instead of accumulating halves them."""
    qbar = torch.zeros((TOKENS, 2 * HEAD_DIM), dtype=torch.bfloat16)
    qbar[:, 0] = hi          # q_hi lane 0
    qbar[:, HEAD_DIM] = lo   # q_lo lane 0
    cache = torch.zeros(
        (NUM_BLOCKS, POOLS_PER_BLOCK, 1, HEAD_DIM), dtype=torch.bfloat16)
    for pool_id in range(NUM_POOLS):
        cache[pool_id // POOLS_PER_BLOCK, pool_id % POOLS_PER_BLOCK, 0, 0] = pool_id

    cum_query_lens = G._gather8_lens(
        torch.tensor([TOKENS], dtype=torch.int32, device=device))
    indexer_seq_lens = G._gather8_lens(
        torch.tensor([NUM_POOLS], dtype=torch.int32, device=device))
    positions = torch.arange(TOKENS, dtype=torch.int32, device=device)
    return qbar.to(device), cache, cum_query_lens, indexer_seq_lens, positions


def expected_scores(cache: torch.Tensor, block_table: torch.Tensor,
                    scale: float) -> torch.Tensor:
    return torch.cat([
        cache[int(block_id), :, 0, 0].float()
        for block_id in block_table.tolist()
    ]) * scale


def main() -> None:
    device = "npu:0"
    # (name, q_hi lane0, q_lo lane0): control keeps lo at zero; the two
    # nonzero-lo cases discriminate accumulate vs overwrite in the second
    # Mmad (expected score = (hi + lo) * pool_id, exact in fp32).
    halves_cases = {
        "hi-only": (1.0, 0.0),
        "lo-only": (0.0, 1.0),
        "split-half": (0.5, 0.5),
        "double": (1.0, 1.0),
    }
    tables = {
        "identity": torch.arange(NUM_BLOCKS, dtype=torch.int32),
        "reverse": torch.arange(NUM_BLOCKS - 1, -1, -1, dtype=torch.int32),
        "constant5": torch.full((NUM_BLOCKS,), 5, dtype=torch.int32),
    }
    # Cover both AIV halves and the visibility boundary around poolTopk=512.
    rows = (2047, 2051, 4079, 4080, 4095)
    ok = True

    for half_name, (hi, lo) in halves_cases.items():
        qbar, cache_cpu, cum, seq, positions = build_encoded_inputs(device, hi, lo)
        cache = cache_cpu.to(device)
        scale = hi + lo
        for name, block_table_cpu in tables.items():
            _, scores = torch.ops._C_ascend.npu_glm5_kpool_indexer(
                qbar,
                cache,
                cum,
                seq,
                block_table_cpu.view(1, -1).to(device),
                positions,
                2048,
                4,
                HEAD_DIM,
                2048,
                1,
            )
            torch.npu.synchronize()
            expected = expected_scores(cache_cpu, block_table_cpu, scale)
            for row in rows:
                visible = min((row + 1) // 4, NUM_POOLS)
                observed = scores[row, :visible].float().cpu()
                exact = torch.equal(observed, expected[:visible])
                ok &= exact
                max_error = float((observed - expected[:visible]).abs().max())
                print(
                    f"[{half_name}/{name}] row={row} visible={visible} exact={exact} "
                    f"max_error={max_error}",
                    flush=True,
                )

    print("BLOCK_TABLE_PROBE:", "PASS" if ok else "FAIL", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
