/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_tiling.cpp
 * \brief
 */

#include "graph/utils/type_utils.h"
#include "graph/defs.h"
#include "glm5_kpool_indexer_tiling.h"
#include "../op_kernel/glm5_kpool_indexer_template_tiling_key.h"

using namespace ge;

namespace optiling {

static ge::graphStatus ParseAndCheckGlm5Kpool(gert::TilingContext *context, Glm5KpoolTilingInfo &info)
{
    const char *opName = context->GetNodeName();
    OP_CHECK_IF(opName == nullptr, OP_LOGE("Glm5KpoolIndexer", "opName got from TilingContext is nullptr"),
                return ge::GRAPH_FAILED);
    info.opName = opName;

    for (uint32_t idx = 0; idx <= POSITIONS_INDEX; idx++) {
        OP_CHECK_IF(context->GetInputShape(idx) == nullptr,
                    OP_LOGE(opName, "Shape of input[%u] is nullptr", idx), return ge::GRAPH_FAILED);
    }

    info.platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(info.platformInfo == nullptr, OP_LOGE(opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(info.platformInfo);
    info.aivNum = ascendcPlatform.GetCoreNumAiv();
    info.aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(info.aicNum == 0 || info.aivNum == 0, OP_LOGE(opName, "num of core obtained is 0."),
                return ge::GRAPH_FAILED);
    auto socVersion = ascendcPlatform.GetSocVersion();
    OP_CHECK_IF(socVersion != platform_ascendc::SocVersion::ASCEND910B &&
                    socVersion != platform_ascendc::SocVersion::ASCEND910_93 &&
                    socVersion != platform_ascendc::SocVersion::ASCEND950,
                OP_LOGE(opName, "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion)),
                return ge::GRAPH_FAILED);
    info.socVersion = socVersion;

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    const int64_t *topkTokens = attrs->GetInt(ATTR_TOPK_TOKENS_INDEX);
    const int64_t *kpool = attrs->GetInt(ATTR_KPOOL_INDEX);
    const int64_t *headDim = attrs->GetInt(ATTR_HEAD_DIM_INDEX);
    const int64_t *maxPoolSeqLen = attrs->GetInt(ATTR_MAX_POOL_SEQ_LEN_INDEX);
    const int64_t *outputMode = attrs->GetInt(ATTR_OUTPUT_MODE_INDEX);
    OP_CHECK_IF(topkTokens == nullptr || kpool == nullptr || headDim == nullptr || maxPoolSeqLen == nullptr,
                OP_LOGE(opName, "Glm5KpoolIndexer required attrs missing."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*headDim != HEAD_DIM_LIMIT, OP_LOGE(opName, "head_dim must be %u, got %ld.", HEAD_DIM_LIMIT, *headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*topkTokens == 0 || *topkTokens > TOPK_TOKENS_LIMIT || *kpool <= 0 || *topkTokens % *kpool != 0,
                OP_LOGE(opName, "topk_tokens(%ld)/kpool(%ld) invalid.", *topkTokens, *kpool),
                return ge::GRAPH_FAILED);

    const gert::Shape &qbarShape = *context->GetInputShape(QBAR_INDEX);
    const gert::Shape &cacheShape = *context->GetInputShape(INDEXER_CACHE_INDEX);
    const gert::Shape &cumShape = *context->GetInputShape(CUM_QUERY_LENS_INDEX);
    const gert::Shape &seqShape = *context->GetInputShape(INDEXER_SEQ_LENS_INDEX);
    const gert::Shape &blockTableShape = *context->GetInputShape(INDEXER_BLOCK_TABLE_INDEX);
    const gert::Shape &positionsShape = *context->GetInputShape(POSITIONS_INDEX);
    OP_CHECK_IF(qbarShape.GetDimNum() != 2 || cacheShape.GetDimNum() != 4 || cumShape.GetDimNum() != 1 ||
                    seqShape.GetDimNum() != 1 || blockTableShape.GetDimNum() != 2 || positionsShape.GetDimNum() != 1,
                OP_LOGE(opName, "Glm5KpoolIndexer input ranks invalid (expect 2/4/1/1/2/1)."),
                return ge::GRAPH_FAILED);

    info.tSize = static_cast<uint32_t>(qbarShape.GetDim(0));
    info.bSize = static_cast<uint32_t>(cumShape.GetDim(0));
    info.poolsPerBlock = static_cast<uint32_t>(cacheShape.GetDim(1));
    info.numCacheBlocks = static_cast<uint32_t>(cacheShape.GetDim(0));
    info.blockTableStride = static_cast<uint32_t>(blockTableShape.GetDim(1));
    info.maxPoolSeqLen = static_cast<uint32_t>(*maxPoolSeqLen);
    info.topkTokens = static_cast<uint32_t>(*topkTokens);
    info.kpool = static_cast<uint32_t>(*kpool);
    info.poolTopk = info.topkTokens / info.kpool;
    info.outputWidth = info.topkTokens + info.kpool - 1;
    info.outputMode = (outputMode == nullptr) ? 0U : static_cast<uint32_t>(*outputMode);
    OP_CHECK_IF(info.bSize == 0 || seqShape.GetDim(0) != static_cast<int64_t>(info.bSize),
                OP_LOGE(opName, "batch size invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.poolsPerBlock == 0 || info.blockTableStride == 0,
                OP_LOGE(opName, "poolsPerBlock/blockTableStride invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint32_t>(qbarShape.GetDim(1)) != HEAD_DIM_LIMIT ||
                    cacheShape.GetDim(2) != 1 || static_cast<uint32_t>(cacheShape.GetDim(3)) != HEAD_DIM_LIMIT,
                OP_LOGE(opName, "qbar/cache inner dims must be [*,128]/[*,*,1,128]."), return ge::GRAPH_FAILED);

    info.qbarType = context->GetInputDesc(QBAR_INDEX)->GetDataType();
    info.cacheType = context->GetInputDesc(INDEXER_CACHE_INDEX)->GetDataType();
    OP_CHECK_IF(info.qbarType != info.cacheType, OP_LOGE(opName, "qbar/cache dtype must match."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.qbarType != ge::DT_BF16 && info.qbarType != ge::DT_FLOAT16,
                OP_LOGE(opName, "qbar dtype must be bf16/fp16."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetInputDesc(POSITIONS_INDEX)->GetDataType() != ge::DT_INT64,
                OP_LOGE(opName, "positions must be int64."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForGlm5KpoolIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGlm5KpoolIndexer(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Glm5KpoolIndexer", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetRawTilingData() == nullptr,
                OP_LOGE(context->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);

    Glm5KpoolTilingInfo info;
    if (ParseAndCheckGlm5Kpool(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(info.platformInfo);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(info.aivNum, info.aicNum, info.aivNum);
    context->SetBlockDim(blockDim);

    // Workspace: lib API scratch only in M1 (the stub kernel writes no score GM).
    // M2 grows this by the scoreGm/LD terms and flips isLDOpen when the total
    // exceeds the 192MB budget (see plan §tiling).
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    Glm5KpoolTilingData tilingData;
    tilingData.set_tSize(info.tSize);
    tilingData.set_bSize(info.bSize);
    tilingData.set_maxPoolSeqLen(info.maxPoolSeqLen);
    tilingData.set_poolsPerBlock(info.poolsPerBlock);
    tilingData.set_numCacheBlocks(info.numCacheBlocks);
    tilingData.set_blockTableStride(info.blockTableStride);
    tilingData.set_topkTokens(info.topkTokens);
    tilingData.set_kpool(info.kpool);
    tilingData.set_poolTopk(info.poolTopk);
    tilingData.set_outputWidth(info.outputWidth);
    tilingData.set_outputMode(info.outputMode);
    tilingData.set_usedCoreNum(blockDim);
    tilingData.set_isLDOpen(0);
    tilingData.set_s2SplitNum(1);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // Tiling key: single dtype axis (bf16/fp16) via the template registry.
    uint32_t qbarType = static_cast<uint32_t>(info.qbarType);
    uint64_t tilingKey = GET_TPL_TILING_KEY(qbarType);
    context->SetTilingKey(tilingKey);
    context->SetScheduleMode(1); // 1: batch mode

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Glm5KpoolIndexer)
    .Tiling(TilingForGlm5KpoolIndexer)
    .TilingParse<Glm5KpoolCompileInfo>(TilingPrepareForGlm5KpoolIndexer);

} // namespace optiling
