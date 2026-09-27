/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#ifndef GLM5_KPOOL_INDEXER_TORCH_ADPT_H
#define GLM5_KPOOL_INDEXER_TORCH_ADPT_H

namespace vllm_ascend::glm5_kpool {

inline std::tuple<at::Tensor, at::Tensor> construct_glm5_kpool_indexer_output_tensor(
    const at::Tensor &qbar, const at::Tensor &positions, int64_t topkTokens, int64_t kpool,
    int64_t maxPoolSeqLen, int64_t outputMode)
{
    constexpr int64_t DIM_0 = 0;
    constexpr int64_t DIM_1 = 1;
    TORCH_CHECK(qbar.size(DIM_0) == positions.size(DIM_0),
                "qbar rows (", qbar.size(DIM_0), ") must match positions (", positions.size(DIM_0), ").");
    TORCH_CHECK(topkTokens > 0 && kpool > 0 && topkTokens % kpool == 0,
                "topk_tokens (", topkTokens, ") must be positive and divisible by kpool (", kpool, ").");

    int64_t poolTopk = topkTokens / kpool; // raw pool ids; expansion in python
    at::Tensor indicesOut = at::empty({qbar.size(DIM_0), 1, poolTopk}, qbar.options().dtype(at::kInt));
    at::Tensor scoresDebugOut;
    if (outputMode == 1 || outputMode == 2) { // diagnostics only
        scoresDebugOut = at::empty({qbar.size(DIM_0), maxPoolSeqLen}, qbar.options().dtype(at::kFloat));
    } else {
        scoresDebugOut = at::empty({0}, qbar.options().dtype(at::kFloat));
    }
    return std::tuple<at::Tensor, at::Tensor>(indicesOut, scoresDebugOut);
}

inline std::tuple<at::Tensor, at::Tensor> npu_glm5_kpool_indexer(
    const at::Tensor &qbar, const at::Tensor &indexerCache, const at::Tensor &cumQueryLens,
    const at::Tensor &indexerSeqLens, const at::Tensor &indexerBlockTable, const at::Tensor &positions,
    int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t outputMode)
{
    TORCH_CHECK(qbar.numel() > 0, "qbar is empty.");
    TORCH_CHECK(indexerCache.numel() > 0, "indexer_cache is empty.");
    TORCH_CHECK(indexerCache.is_contiguous(), "indexer_cache must be contiguous [blocks, poolsPerBlock, 1, headDim].");
    TORCH_CHECK(qbar.scalar_type() == at::kBFloat16 || qbar.scalar_type() == at::kHalf, "qbar dtype must be bf16/fp16.");
    TORCH_CHECK(qbar.scalar_type() == indexerCache.scalar_type(), "qbar and indexer_cache dtypes must match.");
    TORCH_CHECK(positions.scalar_type() == at::kInt, "positions must be int32.");

    auto outputs = construct_glm5_kpool_indexer_output_tensor(qbar, positions, topkTokens, kpool, maxPoolSeqLen,
                                                              outputMode);
    at::Tensor indicesOut = std::get<0>(outputs);
    at::Tensor scoresDebugOut = std::get<1>(outputs);

    if (qbar.device().is_meta()) {
        return outputs;
    }

    EXEC_NPU_CMD(aclnnGlm5KpoolIndexer, qbar, indexerCache, cumQueryLens, indexerSeqLens, indexerBlockTable,
                 positions, topkTokens, kpool, headDim, maxPoolSeqLen, outputMode, indicesOut, scoresDebugOut);

    return outputs;
}

} // namespace vllm_ascend::glm5_kpool

#endif // GLM5_KPOOL_INDEXER_TORCH_ADPT_H
