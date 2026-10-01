#include <cstdint>
#include "graph/utils/type_utils.h"
#include "glm5_kpool_split_aic_tiling.h"
#include "../op_kernel/glm5_kpool_split_aic_template_tiling_key.h"

using namespace ge;

namespace optiling {

struct Glm5KpoolSplitCompileInfo {};

static ge::graphStatus TilingPrepareGlm5KpoolSplit(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGlm5KpoolSplitAic(gert::TilingContext *context)
{
    const char *opName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(opName, "attrs is nullptr"), return ge::GRAPH_FAILED);
    const int64_t *topkTokens = attrs->GetInt(0);
    const int64_t *kpool = attrs->GetInt(1);
    const int64_t *headDim = attrs->GetInt(2);
    const int64_t *maxPoolSeqLen = attrs->GetInt(3);
    const int64_t *splitBatch = attrs->GetInt(4);
    OP_CHECK_IF(topkTokens == nullptr || kpool == nullptr || headDim == nullptr || maxPoolSeqLen == nullptr ||
                    splitBatch == nullptr,
                OP_LOGE(opName, "required attrs missing."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*headDim != 128, OP_LOGE(opName, "head_dim must be 128, got %ld.", *headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*kpool != 4, OP_LOGE(opName, "kpool must be 4, got %ld.", *kpool), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*splitBatch < 0 || (*splitBatch) * 4096 >= *maxPoolSeqLen,
                OP_LOGE(opName, "split_batch %ld beyond max_pool_seq_len %ld.", *splitBatch, *maxPoolSeqLen),
                return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName, "num of core obtained is 0."),
                return ge::GRAPH_FAILED);

    // The wrapper mirrors request lengths to stride 8 so each scalar GM read
    // lands on a 32-byte boundary. Match the validated fused-op contract:
    // the physical tensor length is 8 * the logical request count.
    const int64_t cumDim0 = context->GetInputShape(2)->GetStorageShape().GetDim(0);
    const int64_t seqDim0 = context->GetInputShape(3)->GetStorageShape().GetDim(0);
    OP_CHECK_IF(cumDim0 <= 0 || cumDim0 % 8 != 0 || seqDim0 != cumDim0,
                OP_LOGE(opName,
                        "cum_query_lens/indexer_seq_lens dim0 must match and be a positive multiple of 8, got %ld/%ld.",
                        cumDim0, seqDim0),
                return ge::GRAPH_FAILED);

    Glm5KpoolTilingData tilingData;
    tilingData.set_tSize(static_cast<uint32_t>(context->GetInputShape(0)->GetStorageShape().GetDim(0)));
    tilingData.set_bSize(static_cast<uint32_t>(cumDim0 / 8));
    tilingData.set_maxPoolSeqLen(static_cast<uint32_t>(*maxPoolSeqLen));
    tilingData.set_poolsPerBlock(static_cast<uint32_t>(context->GetInputShape(1)->GetStorageShape().GetDim(1)));
    tilingData.set_numCacheBlocks(static_cast<uint32_t>(context->GetInputShape(1)->GetStorageShape().GetDim(0)));
    tilingData.set_blockTableStride(static_cast<uint32_t>(context->GetInputShape(4)->GetStorageShape().GetDim(1)));
    tilingData.set_topkTokens(static_cast<uint32_t>(*topkTokens));
    tilingData.set_kpool(static_cast<uint32_t>(*kpool));
    tilingData.set_poolTopk(static_cast<uint32_t>(*topkTokens) / static_cast<uint32_t>(*kpool));
    tilingData.set_outputWidth(static_cast<uint32_t>(*topkTokens + *kpool - 1));
    tilingData.set_outputMode(5); // OUTPUT_MODE_SPLIT_AIC
    tilingData.set_usedCoreNum(aicNum);
    tilingData.set_splitBatch(static_cast<uint32_t>(*splitBatch));
    tilingData.set_isLDOpen(0);
    tilingData.set_s2SplitNum(1);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    context->SetBlockDim(aicNum); // pure AIC launch: one block per cube core
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    const uint32_t qbarType = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
    const uint64_t tilingKey = GET_TPL_TILING_KEY(qbarType);
    context->SetTilingKey(tilingKey);
    OP_LOGI(opName, "Glm5KpoolSplitAic tilingKey=%llu blockDim=%u batch=%u", (unsigned long long)tilingKey,
            aicNum, tilingData.get_splitBatch());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Glm5KpoolSplitAic)
    .Tiling(TilingGlm5KpoolSplitAic)
    .TilingParse<Glm5KpoolSplitCompileInfo>(TilingPrepareGlm5KpoolSplit);

} // namespace optiling
