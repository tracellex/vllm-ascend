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


def build_inputs(t_lens, ppb, max_pool, seed, pool_lens=None):
    gen = torch.Generator().manual_seed(seed)
    device = "npu:0"
    n_req = len(t_lens)
    T = sum(t_lens)

    query = torch.randn(T, HEADS, DIM, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    logits = torch.randn(T, HEADS, generator=gen, dtype=torch.float32)
    weights = torch.softmax(logits, dim=1).to(torch.bfloat16)

    # Pool units per request: ceil(len / kpool) by default (a 2k-token row
    # sees only ~500 pools — the ceiling arg alone never grows the real
    # pool count, which once made a "pools=49k" bench silently measure 2k);
    # pass pool_lens explicitly to emulate long-context cache states.
    if pool_lens is None:
        pool_lens = [(l + KPOOL - 1) // KPOOL for l in t_lens]
    pool_lens = torch.tensor(pool_lens, dtype=torch.int32)
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


# A row may legally swap pools whose fp32 reference scores sit closer than
# this: below it the ordering is under the accumulated noise of the H9
# [q_hi | q_lo] k=256 form (own sum order + double-bf16 lo truncation;
# measured device-vs-model error ~1e-3 at score scale ~300) and of the fp32
# reference's own summation order (~1e-5 at the same scale). Anything above
# is a real mismatch: the smallest defect-caused flip observed was 0.0247,
# 12x above this bound.
TIE_TOL = 2e-3


def _tie_adjudicate(inputs, row, only_ref, only_test):
    """True when every ref/test pool pair at this row is a score tie.
    Everything runs on CPU in fp32: the adjudicator itself must be quieter
    than the tie it judges (NPU-computed qbar pushed sub-1e-4 gaps over the
    tolerance)."""
    cache = inputs["indexer_cache"].float().cpu()
    bt = inputs["indexer_block_table"][0].cpu()
    ppb = cache.shape[1]
    keys = cache[bt.long()].reshape(-1, cache.shape[-1])
    q = inputs["query"][row].cpu().float()
    w = inputs["weights"][row].cpu().float()
    qbar = (q * w.unsqueeze(-1)).sum(dim=0)
    pools = sorted({t // KPOOL for t in only_ref} | {t // KPOOL for t in only_test})
    s = {p: float(qbar @ keys[p]) for p in pools}
    ok = True
    for a in sorted({t // KPOOL for t in only_ref}):
        for b in sorted({t // KPOOL for t in only_test}):
            if abs(s[a] - s[b]) > TIE_TOL:
                ok = False
    return ok, {p: s[p] for p in pools}


def compare(ref, test, name, inputs=None, dump_rows=4):
    assert ref.shape == test.shape, f"{name}: shape {ref.shape} != {test.shape}"
    T = ref.shape[0]
    bad = 0
    ties = 0
    first = []
    first_tie = []
    for r in range(T):
        a = ref[r, 0]
        b = test[r, 0]
        sa = tuple(sorted(a[a >= 0].tolist()))
        sb = tuple(sorted(b[b >= 0].tolist()))
        na = int((a < 0).sum())
        nb = int((b < 0).sum())
        if sa != sb or na != nb:
            only_a = sorted(set(sa) - set(sb))
            only_b = sorted(set(sb) - set(sa))
            is_tie = False
            if inputs is not None and na == nb and len(only_a) == len(only_b):
                is_tie, _ = _tie_adjudicate(inputs, r, only_a, only_b)
            if is_tie:
                ties += 1
                if len(first_tie) < dump_rows:
                    first_tie.append((r, only_a[:8], only_b[:8]))
            else:
                bad += 1
                if len(first) < dump_rows:
                    first.append((r, na, nb, only_a[:8], only_b[:8]))
    status = "PASS" if bad == 0 else "FAIL"
    tie_note = f" ties(below-fp32-noise)={ties}" if ties else ""
    print(f"[{name}] T={T} rows_mismatch={bad}{tie_note} -> {status}", flush=True)
    for r, na, nb, oa, ob in first:
        print(f"  row {r}: neg(ref/test)={na}/{nb} only_ref={oa} only_test={ob}")
    for r, oa, ob in first_tie:
        print(f"  row {r}: TIE only_ref={oa} only_test={ob}")
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

    # packed multi-request with unequal lengths and per-request tails
    "packed3": (lambda: build_inputs([64, 100, 37], 32, 64, seed=2)),
    # single long request, pools span many blocks through the shuffled table
    "widescreen": (lambda: build_inputs([256], 32, 2048, seed=3)),
    # 8K pools: the shape class the operator exists for
    "big": (lambda: build_inputs([1024], 32, 8192, seed=4)),
    # topk competition region: visible_pools reaches/exceeds poolTopk=512,
    # where the running top-k must evict losers (M5c real-activation bug class)
    "contend2k": (lambda: build_inputs([2048], 32, 2048, seed=16)),
    "contend4k": (lambda: build_inputs([4096], 32, 2048, seed=17)),
    "contend-pack": (lambda: build_inputs([2048, 1024, 512], 32, 2048, seed=18)),
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
        ok &= compare(ref, test, name, inputs=inputs)
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
