#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QBAR_INDEX = 0;
constexpr uint32_t POSITIONS_INDEX = 5;
constexpr uint32_t ATTR_TOPK_TOKENS_INDEX = 0;
constexpr uint32_t ATTR_KPOOL_INDEX = 1;
constexpr uint32_t ATTR_HEAD_DIM_INDEX = 2;
constexpr uint32_t ATTR_MAX_POOL_SEQ_LEN_INDEX = 3;
constexpr uint32_t ATTR_SPLIT_BATCH_INDEX = 4;

static ge::graphStatus InferShapeGlm5KpoolSplitAic(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Glm5KpoolSplitAic", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *qbarShape = context->GetInputShape(QBAR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qbarShape);
    gert::Shape *scoresShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoresShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *topkTokens = attrs->GetInt(ATTR_TOPK_TOKENS_INDEX);
    const int64_t *kpool = attrs->GetInt(ATTR_KPOOL_INDEX);
    const int64_t *maxPoolSeqLen = attrs->GetInt(ATTR_MAX_POOL_SEQ_LEN_INDEX);
    const int64_t *splitBatch = attrs->GetInt(ATTR_SPLIT_BATCH_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkTokens);
    OP_CHECK_NULL_WITH_CONTEXT(context, kpool);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxPoolSeqLen);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitBatch);
    OP_CHECK_IF(*splitBatch < 0 || (*splitBatch) * 4096 >= *maxPoolSeqLen,
                OP_LOGE(context, "split_batch (%ld) beyond max_pool_seq_len (%ld)!", *splitBatch, *maxPoolSeqLen),
                return ge::GRAPH_FAILED);
    // Fixed [T_pad, 4096]: one tensor reused across batches; the tail batch
    // leaves trailing columns unwritten.
    constexpr int64_t kSplitBatchPools = 4096;
    scoresShape->SetDimNum(2);
    scoresShape->SetDim(0, qbarShape->GetDim(0));
    scoresShape->SetDim(1, kSplitBatchPools);
    (void)POSITIONS_INDEX;
    (void)ATTR_HEAD_DIM_INDEX;
    OP_LOGI(context->GetNodeName(), "Glm5KpoolSplitAic InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGlm5KpoolSplitAic(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Glm5KpoolSplitAic", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Glm5KpoolSplitAic)
    .InferShape(InferShapeGlm5KpoolSplitAic)
    .InferDataType(InferDataTypeGlm5KpoolSplitAic);
} // namespace ops
