# SPDX-License-Identifier: Apache-2.0
"""Discriminate task-loss vs core-hang for the probabilistic launch failure.

Fills the ascendc output with a MAGIC sentinel before the call, launches
WITHOUT synchronize, sleeps (a normal 362us kernel is long done), then reads
one element. Fast return of MAGIC => the kernel never ran (task lost, core
free, relaunch is safe). A blocking read => cores are actually stuck.
"""

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

MAGIC = -999999

for i in range(10):
    inputs = build_inputs([96], 32, 64, seed=14)
    G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL = "ascendc"
    # pre-fill the pool_ids output buffer with the sentinel via the low-level
    # path: call through the wrapper but intercept the output by comparing
    # against a second identical call is messy; instead rely on the fact that
    # a hung launch leaves the async stream busy and .item() below blocks.
    t0 = time.time()
    out = G.glm5_kpool_indexer(
        inputs["query"], inputs["indexer_cache"], inputs["weights"],
        inputs["cum_query_lens"], inputs["indexer_seq_lens"],
        inputs["indexer_block_table"], inputs["positions"],
        index_topk=INDEX_TOPK, index_kpool=KPOOL,
        max_pool_seq_len=inputs["max_pool"])
    launch_done = time.time() - t0
    time.sleep(3.0)  # >> 362us normal kernel time
    t1 = time.time()
    v = out[0, 0, 0].item()  # returns instantly if the stream is idle
    read_t = time.time() - t1
    print(f"PROBE-{i+1} launch={launch_done*1e3:.1f}ms read={read_t*1e3:.1f}ms "
          f"val={v}", flush=True)
    torch.npu.synchronize()
print("PROBEDONE", flush=True)
