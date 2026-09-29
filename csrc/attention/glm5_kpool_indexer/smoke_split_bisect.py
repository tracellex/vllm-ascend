#!/usr/bin/env python3
"""Bisect the split launch pair: attribute the aicore MTE fault to the AIC
or AIV half by syncing after every launch.

Usage: python smoke_split_bisect.py <aic|aiv|both> [case]
  aic  — run only the AIC launches (one per batch), sync each, print score
         coverage stats (scores are zero-init, so written lanes are visible).
  aiv  — run AIC for all batches first (syncs), then the AIV launches.
  both — the full wrapper order (aic(b) -> aiv(b) per batch).
"""
import math
import sys

import torch

sys.path.insert(0, "/opt/src/vllm-ascend")
sys.path.insert(0, "/opt/src/vllm-ascend/csrc/attention/glm5_kpool_indexer")

import smoke_compare as SC
from vllm_ascend.ops.glm5_kpool_indexer import (
    _compute_qbar,
    _gather8_lens,
    _pad_positions,
)

stage = sys.argv[1] if len(sys.argv) > 1 else "both"
case = sys.argv[2] if len(sys.argv) > 2 else "tiny"
inputs = SC.CASES[case]()

query, cache = inputs["query"], inputs["indexer_cache"]
cum, seq = inputs["cum_query_lens"], inputs["indexer_seq_lens"]
bt, positions = inputs["indexer_block_table"], inputs["positions"]
max_pool = inputs["max_pool"]
topk, kpool, head_dim = SC.INDEX_TOPK, SC.KPOOL, query.shape[2]

qbar = _compute_qbar(query, inputs["weights"].to(query.dtype))
positions_pad = _pad_positions(positions.to(torch.int32))
pool_topk = topk // kpool
t_rows = (qbar.shape[0] + 31) // 32 * 32
n_batches = (max_pool + 4095) // 4096
# zero-init makes coverage observable: the AIC fixpipe writes whole tiles,
# the fold only reads masked lanes, so nonzero lanes == written lanes.
scores = torch.zeros((t_rows, 4096), dtype=torch.float32, device=qbar.device)
strip = torch.zeros((t_rows, pool_topk * 2), dtype=torch.float32, device=qbar.device)
pool_ids = torch.zeros((qbar.shape[0], 1, pool_topk), dtype=torch.int32, device=qbar.device)
gc, gs = _gather8_lens(cum), _gather8_lens(seq)
print(f"[{case}] T={query.shape[0]} qbar={qbar.shape[0]} t_rows={t_rows} "
      f"max_pool={max_pool} batches={n_batches} stage={stage}", flush=True)


def run_aic(b):
    torch.ops._C_ascend.npu_glm5_kpool_split_aic(
        qbar, cache, gc, gs, bt, positions_pad,
        topk, kpool, head_dim, max_pool, b, scores)
    torch.npu.synchronize()
    nz = int((scores != 0).sum())
    print(f"  AIC batch {b}: OK nonzero={nz}", flush=True)


def run_aiv(b):
    torch.ops._C_ascend.npu_glm5_kpool_split_aiv(
        gc, gs, positions_pad, scores, strip,
        topk, kpool, head_dim, max_pool, b, pool_ids)
    torch.npu.synchronize()
    print(f"  AIV batch {b}: OK ids<0={(pool_ids < 0).sum().item()} "
          f"max={int(pool_ids.max())}", flush=True)


if stage == "aic":
    for b in range(n_batches):
        run_aic(b)
elif stage == "aiv":
    for b in range(n_batches):
        run_aic(b)
    for b in range(n_batches):
        run_aiv(b)
else:
    for b in range(n_batches):
        run_aic(b)
        run_aiv(b)
print("BISECT DONE", flush=True)
