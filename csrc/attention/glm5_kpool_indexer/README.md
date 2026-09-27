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
attrs  : topk_tokens=2048, kpool=4, head_dim=128, max_pool_seq_len, output_mode=0|1|3
output : pool_ids[T, 1, topk_tokens / kpool] i32
         scores_debug[T, max_pool_seq_len] fp32   (output_mode=1/2)
```

Per row: `visible = min((pos+1)>>log2(kpool), seq_lens_pool[req])`; pool p in
[0, visible) scores `dot(qbar, cache[bt[req, p/poolsPerBlock], p%ppb, 0, :])`;
the operator returns the `topk_tokens/kpool` best pool ids. The Python wrapper
expands them to `pool*kpool + {0..kpool-1}`, pads with -1, and appends the
causal tail, matching `append_causal_tail` semantics.

## Status

- The existing fused mode is the validated direct-device-top-k baseline.
- arch35 is retained as the 950 draft path and follows the same int32 positions
  and raw pool-id contract.
- Selection: `VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL`
  (`triton`/`auto`/`ascendc`/`ascendc_group_topk`/`ascendc_group_topk_m64`) in
  `vllm_ascend/ops/glm5_kpool_indexer.py`; Triton stays the default.
- `ascendc_group_topk` is an A3-only experiment: Cube stages eight S2 tiles in
  per-core scratch, and AIV reduces each 1024-pool group to top-512 before
  merging it into the running top-k. It avoids both a full `[T, P]` score
  matrix and per-tile GM candidate-state spills; total score scratch traffic
  remains unchanged. It is not the default pending NPU acceptance.
- `ascendc_group_topk_m64` selects an opt-in 64-row arch22 group-topk candidate
  (output mode 4). It preserves the 32-row group implementation as a separate
  control; CANN build resource allocation, NPU correctness, and latency are
  pending. Triton remains the default.
