import sys, time
sys.path.insert(0, "/opt/src/vllm-ascend/csrc/attention/glm5_kpool_indexer")
import torch, torch_npu, vllm_ascend
from importlib import import_module
from vllm_ascend.utils import bootstrap_custom_op_env
bootstrap_custom_op_env()
import_module("vllm_ascend.vllm_ascend_C")
from vllm_ascend.ops import glm5_kpool_indexer as G
from smoke_compare import build_inputs, INDEX_TOPK, KPOOL

for i in range(8):
    inputs = build_inputs([96], 32, 64, seed=14)
    G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL = "triton"
    s = time.time()
    out = G.glm5_kpool_indexer(
        inputs["query"], inputs["indexer_cache"], inputs["weights"],
        inputs["cum_query_lens"], inputs["indexer_seq_lens"],
        inputs["indexer_block_table"], inputs["positions"],
        index_topk=INDEX_TOPK, index_kpool=KPOOL, max_pool_seq_len=inputs["max_pool"])
    torch.npu.synchronize()
    print(f"TRITON-{i+1} ok t={time.time()-s:.1f}s", flush=True)
print("TRITONDONE")
