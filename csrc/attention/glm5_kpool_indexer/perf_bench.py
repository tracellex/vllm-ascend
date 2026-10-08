# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3 perf bench for Glm5KpoolIndexer: triton vs ascendc Task Duration.

Run under msprof (the only hang-immune environment); read kernel times from
the op_summary CSV the profiler emits. Each impl gets WARMUP launches then
TIMED launches interleaved per shape:

    msprof --application=/tmp/run_perf.sh --output=/tmp/prof_perf ...
"""

from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from importlib import import_module

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import_module("vllm_ascend.vllm_ascend_C")
from vllm_ascend.ops import glm5_kpool_indexer as G  # noqa: E402
from smoke_compare import build_inputs, INDEX_TOPK, KPOOL  # noqa: E402

# (name, t_lens, max_pool): the shapes that matter for the P-side prefill
# hotspot. 128K context -> 32K pools over thousands of tokens.
SHAPES = [
    ("mid", [1024], 8192),
    ("long", [2048], 32768),
    ("wide", [4096], 32768),
]
WARMUP = 3
TIMED = 10


def bench(impl, inputs, tag):
    G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL = impl
    for _ in range(WARMUP):
        G.glm5_kpool_indexer(
            inputs["query"], inputs["indexer_cache"], inputs["weights"],
            inputs["cum_query_lens"], inputs["indexer_seq_lens"],
            inputs["indexer_block_table"], inputs["positions"],
            index_topk=INDEX_TOPK, index_kpool=KPOOL,
            max_pool_seq_len=inputs["max_pool"])
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(TIMED):
        G.glm5_kpool_indexer(
            inputs["query"], inputs["indexer_cache"], inputs["weights"],
            inputs["cum_query_lens"], inputs["indexer_seq_lens"],
            inputs["indexer_block_table"], inputs["positions"],
            index_topk=INDEX_TOPK, index_kpool=KPOOL,
            max_pool_seq_len=inputs["max_pool"])
    torch.npu.synchronize()
    wall = (time.time() - t0) / TIMED * 1e3
    print(f"BENCH {tag} {impl} wall_avg_ms={wall:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["mid", "long", "wide"])
    p.add_argument("--only", choices=["triton", "ascendc"], default=None)
    args = p.parse_args()
    global WARMUP, TIMED
    if args.only:  # hang-rate-aware single-shot sampling mode
        WARMUP, TIMED = 1, 1
    table = {n: (t, m) for n, t, m in SHAPES}
    for name in args.shapes:
        t_lens, max_pool = table[name]
        inputs = build_inputs(t_lens, 32, max_pool, seed=42)
        impls = [args.only] if args.only else ["triton", "ascendc"]
        for impl in impls:
            bench(impl, inputs, name)
        del inputs
        torch.npu.empty_cache() if hasattr(torch.npu, "empty_cache") else None
    print("BENCHDONE", flush=True)


if __name__ == "__main__":
    main()
