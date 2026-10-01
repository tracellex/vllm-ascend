/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_template_tiling_key.h
 * \brief Single dtype axis: bf16/fp16 qbar+cache. Pool cache layout is fixed
 *        PA [blocks, poolsPerBlock, 1, headDim]; only the element dtype varies.
 */

#ifndef TEMPLATE_TILING_KEY_GLMKPOOL_H_
#define TEMPLATE_TILING_KEY_GLMKPOOL_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define GLMK_TPL_FP16 1
#define GLMK_TPL_BF16 27

ASCENDC_TPL_ARGS_DECL(Glm5KpoolIndexer, // 算子OpType
                      ASCENDC_TPL_DTYPE_DECL(DT_Q, GLMK_TPL_FP16, GLMK_TPL_BF16), );

// 支持的模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, GLMK_TPL_FP16), ),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, GLMK_TPL_BF16), ), );

#endif
