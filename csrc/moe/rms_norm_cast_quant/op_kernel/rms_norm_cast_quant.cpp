/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#include "rms_norm_cast_quant.h"

using namespace AscendC;

#define RUN_RMS_NORM_CAST_QUANT(TYPE)                       \
    do {                                                    \
        KernelRmsNormCastQuant<TYPE> op(&pipe);             \
        op.Init(x, gamma, y, y_fp32, y_int8, y_scale, &tiling_data); \
        op.Process();                                       \
    } while (0)

extern "C" __global__ __aicore__ void rms_norm_cast_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR y_fp32, GM_ADDR y_int8,
    GM_ADDR y_scale, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        RUN_RMS_NORM_CAST_QUANT(half);
    } else if (TILING_KEY_IS(3)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        RUN_RMS_NORM_CAST_QUANT(bfloat16_t);
#endif
    }
}
