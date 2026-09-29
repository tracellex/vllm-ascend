#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t POSITIONS_INDEX = 2;
constexpr uint32_t ATTR_TOPK_TOKENS_INDEX = 0;
constexpr uint32_t ATTR_KPOOL_INDEX = 1;

static ge::graphStatus InferShapeGlm5KpoolSplitAiv(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Glm5KpoolSplitAiv", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *positionsShape = context->GetInputShape(POSITIONS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, positionsShape);
    gert::Shape *indicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *topkTokens = attrs->GetInt(ATTR_TOPK_TOKENS_INDEX);
    const int64_t *kpool = attrs->GetInt(ATTR_KPOOL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkTokens);
    OP_CHECK_NULL_WITH_CONTEXT(context, kpool);
    OP_CHECK_IF(*topkTokens <= 0 || *kpool <= 0 || *topkTokens % *kpool != 0,
                OP_LOGE(context, "topk_tokens (%ld) must be divisible by kpool (%ld)!", *topkTokens, *kpool),
                return ge::GRAPH_FAILED);
    // True token count: positions are the unpadded [T] view.
    indicesShape->SetDimNum(3);
    indicesShape->SetDim(0, positionsShape->GetDim(0));
    indicesShape->SetDim(1, 1);
    indicesShape->SetDim(2, *topkTokens / *kpool);
    OP_LOGI(context->GetNodeName(), "Glm5KpoolSplitAiv InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGlm5KpoolSplitAiv(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Glm5KpoolSplitAiv", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(Glm5KpoolSplitAiv)
    .InferShape(InferShapeGlm5KpoolSplitAiv)
    .InferDataType(InferDataTypeGlm5KpoolSplitAiv);
} // namespace ops
