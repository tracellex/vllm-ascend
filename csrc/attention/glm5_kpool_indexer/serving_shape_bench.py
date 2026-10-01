# SPDX-License-Identifier: Apache-2.0
"""TRUE serving-shape bench: pool cache length decoupled from T (the engine
feeds ceil(ctx/4) pools: 16k @64k ctx, 48k @192k)."""
import sys
sys.path.insert(0, "csrc/attention/glm5_kpool_indexer")
import time
import torch
import smoke_compare as S

G = S.G


def bench(impl, T, pools, tag):
    # pool_lens decoupled: a T-token chunk seeing a `pools`-deep cache
    inputs = S.build_inputs([T], 32, pools, seed=99, pool_lens=[pools])
    G.VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL = impl
    for _ in range(3):
        G.glm5_kpool_indexer(inputs["query"], inputs["indexer_cache"], inputs["weights"],
                             inputs["cum_query_lens"], inputs["indexer_seq_lens"],
                             inputs["indexer_block_table"], inputs["positions"],
                             index_topk=2048, index_kpool=4,
                             max_pool_seq_len=inputs["max_pool"])
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(10):
        G.glm5_kpool_indexer(inputs["query"], inputs["indexer_cache"], inputs["weights"],
                             inputs["cum_query_lens"], inputs["indexer_seq_lens"],
                             inputs["indexer_block_table"], inputs["positions"],
                             index_topk=2048, index_kpool=4,
                             max_pool_seq_len=inputs["max_pool"])
    torch.npu.synchronize()
    print(f"BENCH {tag} T={T} pools={pools} {impl} wall_avg_ms={(time.time()-t0)/10*1e3:.3f}", flush=True)


if __name__ == "__main__":
    for pools in (2048, 8192, 16000, 32768, 48000):
        for impl in ("triton", "ascendc_group_topk", "ascendc_group_topk_m64", "ascendc_group_topk_split"):
            bench(impl, 7680, pools, "srv")
