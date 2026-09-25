# Glm5KpoolIndexer

Fused AscendC operator for the GLM-Next KPool lightning indexer: scores every
visible compressed-K pool against each token's head-weighted query and selects
the top-k pools. The Python wrapper expands pool ids and appends the causal tail.
It replaces the Ascend-Triton `_glm5_next_lightning_indexer_score_kernel`
path, which re-reads the whole pool cache once per token (zero cross-token data
reuse) and dominates P-side prefill profiling (~50% of op time at 128K+).

## Semantics

```
inputs : qbar[T, 128] bf16            (head-weighted query, precomputed)
        indexer_cache[B_blocks, poolsPerBlock, 1, 128] bf16  (paged pool K)
        cum_query_lens[B] i32         (inclusive ends, no leading zero)
        indexer_seq_lens[B] i32       (per-request length in POOL units)
        indexer_block_table[B, maxBlocks] i32
        positions[T] i32              (absolute token positions)
attrs  : topk_tokens=2048, kpool=4, head_dim=128, max_pool_seq_len, output_mode=0|1
output : pool_ids[T, 1, topk_tokens / kpool] i32
         scores_debug[T, max_pool_seq_len] fp32   (output_mode=1 only)
```

Per row: `visible = min((pos+1)>>log2(kpool), seq_lens_pool[req])`; pool p in
[0, visible) scores `dot(qbar, cache[bt[req, p/poolsPerBlock], p%ppb, 0, :])`;
the operator returns the `topk_tokens/kpool` best pool ids. The Python wrapper
expands them to `pool*kpool + {0..kpool-1}`, pads with -1, and appends the
causal tail, matching `append_causal_tail` semantics.

## Status

- **M2 active**: arch22 MIX_AIC_1_2 implementation is under device validation;
  AIC score generation is traced through completion and AIV top-k remains WIP.
- arch35 is retained as the 950 draft path and follows the same int32 positions
  and raw pool-id contract.
- Selection: `VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL` (auto/triton/ascendc) in
  `vllm_ascend/ops/glm5_kpool_indexer.py`; the Triton path stays as fallback.
