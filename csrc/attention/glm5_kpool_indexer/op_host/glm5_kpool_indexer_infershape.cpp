/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QBAR_INDEX = 0;
constexpr uint32_t INDEXER_CACHE_INDEX = 1;
constexpr uint32_t CUM_QUERY_LENS_INDEX = 2;
constexpr uint32_t INDEXER_SEQ_LENS_INDEX = 3;
constexpr uint32_t INDEXER_BLOCK_TABLE_INDEX = 4;
constexpr uint32_t POSITIONS_INDEX = 5;
constexpr uint32_t ATTR_TOPK_TOKENS_INDEX = 0;
constexpr uint32_t ATTR_KPOOL_INDEX = 1;
constexpr uint32_t ATTR_HEAD_DIM_INDEX = 2;
constexpr uint32_t ATTR_MAX_POOL_SEQ_LEN_INDEX = 3;
constexpr uint32_t ATTR_OUTPUT_MODE_INDEX = 4;

static ge::graphStatus InferShapeGlm5KpoolIndexer(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Glm5KpoolIndexer", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *qbarShape = context->GetInputShape(QBAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qbarShape);
    const gert::Shape *cacheShape = context->GetInputShape(INDEXER_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheShape);

    gert::Shape *indicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    gert::Shape *scoresDebugShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresDebugShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *topkTokens = attrs->GetInt(ATTR_TOPK_TOKENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkTokens);
    const int64_t *kpool = attrs->GetInt(ATTR_KPOOL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kpool);
    const int64_t *headDim = attrs->GetInt(ATTR_HEAD_DIM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, headDim);
    const int64_t *maxPoolSeqLen = attrs->GetInt(ATTR_MAX_POOL_SEQ_LEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxPoolSeqLen);
    const int64_t *outputMode = attrs->GetInt(ATTR_OUTPUT_MODE_INDEX);

    OP_CHECK_IF(qbarShape->GetDimNum() != 2,
                OP_LOGE(context, "qbar dims (%zu) must be 2 [T, 2*headDim]!", qbarShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    // qbar rows pack the FP32 head-weighted query as [q_hi | q_lo] halves
    // (H9): the cube accumulates both Mmads in FP32 to match the Triton
    // reference's FP32 qbar fidelity.
    OP_CHECK_IF(qbarShape->GetDim(1) != 2 * *headDim,
                OP_LOGE(context, "qbar width (%ld) != 2 * head_dim (%ld)!", qbarShape->GetDim(1), *headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cacheShape->GetDimNum() != 4 || cacheShape->GetDim(2) != 1 ||
                    cacheShape->GetDim(3) != *headDim,
                OP_LOGE(context, "indexer_cache must be [blocks, poolsPerBlock, 1, headDim], got dimNum %zu!",
                        cacheShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*topkTokens <= 0 || *kpool <= 0 || *topkTokens % *kpool != 0,
                OP_LOGE(context, "topk_tokens (%ld) must be positive and divisible by kpool (%ld)!", *topkTokens,
                        *kpool),
                return ge::GRAPH_FAILED);

    // Raw pool ids: Python expands these to token ids and appends the tail.
    indicesShape->SetDimNum(3);
    indicesShape->SetDim(0, qbarShape->GetDim(0));
    indicesShape->SetDim(1, 1);
    indicesShape->SetDim(2, *topkTokens / *kpool);

    // scores_debug: [T, maxPoolSeqLen] when output_mode >= 1 (1 = combined
    // scores, 2 = q_lo staging diagnostic), else empty.
    if (outputMode != nullptr && *outputMode >= 1) {
        scoresDebugShape->SetDimNum(2);
        scoresDebugShape->SetDim(0, qbarShape->GetDim(0));
        scoresDebugShape->SetDim(1, *maxPoolSeqLen);
    } else {
        scoresDebugShape->SetDimNum(1);
        scoresDebugShape->SetDim(0, 0);
    }
    OP_LOGI(context->GetNodeName(), "Glm5KpoolIndexer InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGlm5KpoolIndexer(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Glm5KpoolIndexer", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGI(context->GetNodeName(), "Glm5KpoolIndexer InferDataType end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Glm5KpoolIndexer)
    .InferShape(InferShapeGlm5KpoolIndexer)
    .InferDataType(InferDataTypeGlm5KpoolIndexer);
} // namespace ops
