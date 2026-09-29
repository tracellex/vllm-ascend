# SPDX-License-Identifier: Apache-2.0
"""H9 measurement: AscendC raw scores must track the FP32-qbar reference.

Runs the contend4k contention shape through the operator in output_mode=1
and compares the raw fp32 pool scores against two CPU controls computed
from the same inputs:

  * ref32 — the head-weighted query kept in FP32 (what the Triton reference
    feeds its score kernel), and
  * ref16 — the same qbar rounded to a single bf16 (the pre-H9 AscendC
    behavior that flipped 436/4096 contend4k top-k rows).

With the [q_hi | q_lo] dual-Mmad fix the per-row max error must collapse
towards ref32 (bf16 products are exact in fp32, so the only residual is the
~2^-17 lo-half truncation and summation order), and per-row top-512 sets
must agree with ref32 everywhere while disagreeing with ref16 on the
quantization-boundary rows.

    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    ASCEND_LAUNCH_BLOCKING=1 python3 \
      csrc/attention/glm5_kpool_indexer/probe_qbar_precision.py
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from importlib import import_module

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import_module("vllm_ascend.vllm_ascend_C")
from vllm_ascend.ops import glm5_kpool_indexer as G  # noqa: E402

sys.path.insert(0, "csrc/attention/glm5_kpool_indexer")
import smoke_compare as S  # noqa: E402

POOL_TOPK = 512


def logical_keys(inputs):
    """[numPools, dim] bf16 keys in logical pool order for request 0."""
    cache = inputs["indexer_cache"].float().cpu()
    bt = inputs["indexer_block_table"][0].cpu()
    ppb = cache.shape[1]
    blocks = cache[bt.long()]  # [maxBlocks, ppb, 1, dim]
    return blocks.reshape(-1, cache.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="ascendc_group_topk",
                        choices=("ascendc_group_topk", "ascendc_group_topk_split"))
    args = parser.parse_args()
    inputs = S.build_inputs([4096], 32, 2048, seed=17)
    dev = "npu:0"
    query = inputs["query"].float().cpu()
    weights = inputs["weights"].float().cpu()

    qbar32 = (query * weights.unsqueeze(-1)).sum(dim=1)  # [T, dim] fp32
    qbar16 = qbar32.to(torch.bfloat16)
    keys = logical_keys(inputs)  # [pools, dim] (bf16-valued, fp32 storage)
    pool_lens = inputs["indexer_seq_lens"].cpu()
    positions = inputs["positions"].cpu()

    ref32 = qbar32 @ keys.t()  # fp32 qbar control
    ref16 = qbar16.float() @ keys.t()  # single-bf16 qbar control

    qbar2 = G._compute_qbar(inputs["query"], inputs["weights"])
    gather_cum = G._gather8_lens(inputs["cum_query_lens"])
    gather_seq = G._gather8_lens(inputs["indexer_seq_lens"])
    positions_pad = G._pad_positions(inputs["positions"].to(torch.int32))
    if args.impl == "ascendc_group_topk_split":
        scores = torch.empty((qbar2.shape[0], 4096), dtype=torch.float32, device=dev)
        torch.ops._C_ascend.npu_glm5_kpool_split_aic(
            qbar2, inputs["indexer_cache"], gather_cum, gather_seq,
            inputs["indexer_block_table"], positions_pad, 2048, 4,
            inputs["query"].shape[2], inputs["max_pool"], 0, scores)
    else:
        _, scores = torch.ops._C_ascend.npu_glm5_kpool_indexer(
            qbar2, inputs["indexer_cache"], gather_cum, gather_seq,
            inputs["indexer_block_table"], positions_pad, 2048, 4,
            inputs["query"].shape[2], inputs["max_pool"], 1)
    torch.npu.synchronize()
    asc = scores[: positions.shape[0]].float().cpu()

    max_pool = min(inputs["max_pool"], keys.shape[0])  # actual logical pools
    err32 = (asc[:, :max_pool] - ref32[:, :max_pool]).abs().max()
    err16 = (asc[:, :max_pool] - ref16[:, :max_pool]).abs().max()
    scale = ref32[:, :max_pool].abs().max()

    set32 = set16 = ties = rows = 0
    for r in range(asc.shape[0]):
        visible = int(min((int(positions[r]) + 1) // 4, int(pool_lens[0]), max_pool))
        if visible <= POOL_TOPK:
            continue  # no competition: every visible pool survives
        rows += 1
        a = set(torch.topk(asc[r, :visible], POOL_TOPK).indices.tolist())
        e32 = set(torch.topk(ref32[r, :visible], POOL_TOPK).indices.tolist())
        e16 = set(torch.topk(ref16[r, :visible], POOL_TOPK).indices.tolist())
        if a == e32:
            set32 += 1
        else:
            # adjudicate: every swapped pair must be a below-fp32-noise tie
            gaps = [abs(float(ref32[r, p]) - float(ref32[r, q]))
                    for p in a - e32 for q in e32 - a]
            if gaps and max(gaps) <= 1e-4:
                ties += 1
                set32 += 1
        set16 += a == e16

    print(f"max|asc-ref32|={float(err32):.6f} max|asc-ref16|={float(err16):.6f} "
          f"(score scale ~{float(scale):.1f})", flush=True)
    print(f"competitive rows (visible>{POOL_TOPK}): {rows}; "
          f"top512 == ref32: {set32}/{rows} (incl. {ties} fp32-noise ties), "
          f"== ref16: {set16}/{rows}", flush=True)

    ok = set32 == rows and set16 < rows
    print("QBAR_PRECISION_PROBE:", "PASS" if ok else "FAIL", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
