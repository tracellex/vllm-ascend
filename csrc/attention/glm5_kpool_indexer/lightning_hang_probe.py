# SPDX-License-Identifier: Apache-2.0
"""Hang probe for the vendored lightning_indexer (same build tree, same
arch22 MIX_AIC_1_2 handshake pattern as Glm5KpoolIndexer).

If this exonerated-in-production operator also hangs at the same rate in
this environment, the defect is platform-wide; if it never hangs, it is a
live reference to diff against."""

import time

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from importlib import import_module

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import_module("vllm_ascend.vllm_ascend_C")

T, HEADS, D = 96, 1, 128          # 3 M-tiles -> >=3 lockstep rounds
BLOCK_SIZE, NUM_BLOCKS = 64, 64   # PA cache: [blocks, 64, 1, 128]
KV_LEN = 2048

q = torch.randn(T, HEADS, D, dtype=torch.float16, device="npu:0")
w = torch.rand(T, HEADS, dtype=torch.float16, device="npu:0")
key = torch.randn(NUM_BLOCKS, BLOCK_SIZE, 1, D, dtype=torch.float16, device="npu:0")
seq_q = torch.tensor([T], dtype=torch.int32, device="npu:0")
seq_k = torch.tensor([KV_LEN], dtype=torch.int32, device="npu:0")
bt = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="npu:0").unsqueeze(0)

for i in range(8):
    t0 = time.time()
    topk, _ = torch.ops._C_ascend.npu_lightning_indexer(
        query=q, key=key, weights=w,
        actual_seq_lengths_query=seq_q, actual_seq_lengths_key=seq_k,
        block_table=bt,
        layout_query="TND", layout_key="PA_BSND",
        sparse_count=2048, sparse_mode=3)
    torch.npu.synchronize()
    print(f"LI-{i+1} ok t={(time.time()-t0)*1e3:.1f}ms shape={tuple(topk.shape)}", flush=True)
print("LIDONE", flush=True)
