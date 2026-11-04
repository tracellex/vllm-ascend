/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {
static ge::graphStatus InferShape4RmsNormCastQuant(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape* y_fp32_shape = context->GetOutputShape(1);
    gert::Shape* y_int8_shape = context->GetOutputShape(2);
    gert::Shape* y_scale_shape = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_fp32_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_int8_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_scale_shape);
    *y_shape = *x_shape;
    *y_fp32_shape = *x_shape;
    *y_int8_shape = *x_shape;
    y_scale_shape->SetDim(0, x_shape->GetDim(0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4RmsNormCastQuant(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_INT8);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormCastQuant)
    .InferShape(InferShape4RmsNormCastQuant)
    .InferDataType(InferDataType4RmsNormCastQuant);
}  // namespace ops
