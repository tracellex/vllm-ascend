# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device smoke + parity check for the Glm5KpoolIndexer AscendC operator.

Feeds identical packed multi-request inputs through the Triton reference
wrapper and the AscendC operator and compares the expanded token-index rows
as sets: top-k ordering may legally differ between implementations, so each
output row is reduced to its sorted set of valid (>= 0) token indices plus
the -1 count. Use after ``build_aclnn.sh`` + ``setup.py build_ext --inplace``
inside the build container:

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
    ASCEND_LAUNCH_BLOCKING=1 python3 csrc/attention/glm5_kpool_indexer/smoke_compare.py \
        --cases tiny packed3 widescreen big
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch_npu  # noqa: F401  (device registration before any npu tensor)
import vllm_ascend  # noqa: F401
# Mirror Platform.visible_device_id_to_physical_device_id: the _C_ascend
# namespace is registered by the extension module, and loading it must be
# preceded by bootstrap_custom_op_env (import_kernels notes that a bare
# import breaks ASCEND_RT_VISIBLE_DEVICES handling).
from importlib import import_module

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import_module("vllm_ascend.vllm_ascend_C")
from vllm_ascend.ops import glm5_kpool_indexer as G  # noqa: E402

HEADS = 8
DIM = 128
KPOOL = 4
INDEX_TOPK = 2048  # pool_topk = 512, matches the serving attr


def build_inputs(t_lens, ppb, max_pool, seed):
    gen = torch.Generator().manual_seed(seed)
    device = "npu:0"
    n_req = len(t_lens)
    T = sum(t_lens)

    query = torch.randn(T, HEADS, DIM, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    logits = torch.randn(T, HEADS, generator=gen, dtype=torch.float32)
    weights = torch.softmax(logits, dim=1).to(torch.bfloat16)

    # Pool units per request: ceil(len / kpool); clamp to max_pool for the
    # visible-pool ceiling path (max_pool < needed also exercises masking).
    pool_lens = torch.tensor(
        [(l + KPOOL - 1) // KPOOL for l in t_lens], dtype=torch.int32)
    total_pools = max(int(pool_lens.max()), 1)
    n_blocks = (total_pools + ppb - 1) // ppb

    cache = (torch.randn(n_blocks, ppb, 1, DIM, generator=gen,
                         dtype=torch.float32) * 0.5).to(torch.bfloat16)
    max_blocks = (total_pools + ppb - 1) // ppb
    bt = torch.randperm(n_blocks, generator=gen)[:max_blocks]
    while bt.numel() < max_blocks:  # n_blocks < max_blocks cannot happen, kept for safety
        bt = torch.cat([bt, bt[: max_blocks - bt.numel()]])
    block_table = bt.unsqueeze(0).expand(n_req, -1).contiguous().to(torch.int32)

    cum = torch.tensor([sum(t_lens[:i + 1]) for i in range(n_req)], dtype=torch.int32)
    positions = torch.cat([torch.arange(l) for l in t_lens]).to(torch.int32)

    return dict(
        query=query.to(device),
        indexer_cache=cache.to(device),
        weights=weights.to(device),
        cum_query_lens=cum.to(device),
        indexer_seq_lens=pool_lens.to(device),
        indexer_block_table=block_table.to(device),
        positions=positions.to(device),
        max_pool=max_pool,
    )


def run_impl(inputs):
    saved = G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL
    try:
        G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL = "triton"
        ref = G.glm5_kpool_indexer(
            inputs["query"], inputs["indexer_cache"], inputs["weights"],
            inputs["cum_query_lens"], inputs["indexer_seq_lens"],
            inputs["indexer_block_table"], inputs["positions"],
            index_topk=INDEX_TOPK, index_kpool=KPOOL,
            max_pool_seq_len=inputs["max_pool"],
        )
        G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL = "ascendc"
        test = G.glm5_kpool_indexer(
            inputs["query"], inputs["indexer_cache"], inputs["weights"],
            inputs["cum_query_lens"], inputs["indexer_seq_lens"],
            inputs["indexer_block_table"], inputs["positions"],
            index_topk=INDEX_TOPK, index_kpool=KPOOL,
            max_pool_seq_len=inputs["max_pool"],
        )
    finally:
        G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL = saved
    torch.npu.synchronize()
    return ref.cpu(), test.cpu()


def compare(ref, test, name, dump_rows=4):
    assert ref.shape == test.shape, f"{name}: shape {ref.shape} != {test.shape}"
    T = ref.shape[0]
    bad = 0
    first = []
    for r in range(T):
        a = ref[r, 0]
        b = test[r, 0]
        sa = tuple(sorted(a[a >= 0].tolist()))
        sb = tuple(sorted(b[b >= 0].tolist()))
        na = int((a < 0).sum())
        nb = int((b < 0).sum())
        if sa != sb or na != nb:
            bad += 1
            if len(first) < dump_rows:
                only_a = sorted(set(sa) - set(sb))
                only_b = sorted(set(sb) - set(sa))
                first.append((r, na, nb, only_a[:8], only_b[:8]))
    status = "PASS" if bad == 0 else "FAIL"
    print(f"[{name}] T={T} rows_mismatch={bad} -> {status}", flush=True)
    for r, na, nb, oa, ob in first:
        print(f"  row {r}: neg(ref/test)={na}/{nb} only_ref={oa} only_test={ob}")
    return bad == 0


CASES = {
    # non-multiple-of-32 T exercises the pad path; few pools < pool_topk
    "tiny": (lambda: build_inputs([33], 32, 128, seed=1)),
    # two requests: isolates reqIdx>0 paths from 3-way core spread
    "packed2": (lambda: build_inputs([64, 100], 32, 64, seed=5)),
    # two requests, one unit each: the minimal multi-request shape
    "packed2x1": (lambda: build_inputs([32, 32], 32, 64, seed=6)),
    # unit-count ladder for the multi-request deadlock trigger
    "packed3u": (lambda: build_inputs([64, 32], 32, 64, seed=7)),
    "packed4u": (lambda: build_inputs([64, 64], 32, 64, seed=8)),
    "packed3eq": (lambda: build_inputs([32, 32, 32], 32, 64, seed=9)),
    # single request spanning 3 core pairs: cores-vs-requests discriminator
    "single96": (lambda: build_inputs([96], 32, 64, seed=14)),
    "packed4eq": (lambda: build_inputs([32, 32, 32, 32], 32, 64, seed=15)),
    # 4 units incl. mStart=64: 16KB-offset hypothesis probe
    "single128": (lambda: build_inputs([128], 32, 64, seed=16)),
    # 2 units exactly, no mStart=64: control
    "single64": (lambda: build_inputs([64], 32, 64, seed=17)),


    # packed multi-request with unequal lengths and per-request tails
    "packed3": (lambda: build_inputs([64, 100, 37], 32, 64, seed=2)),
    # single long request, pools span many blocks through the shuffled table
    "widescreen": (lambda: build_inputs([256], 32, 2048, seed=3)),
    # 8K pools: the shape class the operator exists for
    "big": (lambda: build_inputs([1024], 32, 8192, seed=4)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["tiny"], choices=sorted(CASES))
    p.add_argument("--ppb", type=int, default=None,
                   help="override pools-per-block (cache.shape[1]) for all cases")
    args = p.parse_args()

    if not G._ascendc_available():
        print("FATAL: torch.ops._C_ascend.npu_glm5_kpool_indexer not loaded",
              file=sys.stderr)
        sys.exit(2)

    ok = True
    for name in args.cases:
        inputs = CASES[name]()
        if args.ppb is not None:
            inputs = build_inputs(
                [int(inputs["positions"].shape[0])] if name != "packed3"
                else [64, 100, 37],
                args.ppb, inputs["max_pool"], seed=hash(name) % 1000)
        ref, test = run_impl(inputs)
        ok &= compare(ref, test, name)
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
