/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef VLLM_ASCEND_RMS_NORM_CAST_QUANT_TORCH_ADPT_H
#define VLLM_ASCEND_RMS_NORM_CAST_QUANT_TORCH_ADPT_H

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_rms_norm_cast_quant(
    const at::Tensor& x, const at::Tensor& gamma, double epsilon)
{
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "npu_rms_norm_cast_quant only supports float16 and bfloat16 input");
    TORCH_CHECK(x.scalar_type() == gamma.scalar_type(),
                "x and gamma must have the same dtype");
    TORCH_CHECK(gamma.dim() == 1 && x.dim() >= 1 &&
                    x.size(-1) == gamma.size(0),
                "gamma must be 1D and match the last dimension of x");
    // The kernel writes one scale per row (num_row = numel / num_col); flatten
    // leading dims so rank>2 inputs cannot overflow the scale output.
    const int64_t num_col = gamma.size(0);
    const int64_t num_row = x.numel() / num_col;
    at::Tensor x2 = x.contiguous().view({num_row, num_col});
    // Scale shape mirrors aclnnDynamicQuantV2: all leading dims of x (rank-1
    // input degenerates to a single scale element).
    std::vector<int64_t> scale_shape;
    if (x.dim() == 1) {
        scale_shape = {1};
    } else {
        scale_shape = std::vector<int64_t>(x.sizes().begin(), x.sizes().end() - 1);
    }
    at::Tensor y = at::empty(x.sizes(), x.options());
    at::Tensor y_fp32 = at::empty(x.sizes(), x.options().dtype(at::kFloat));
    at::Tensor y_int8 = at::empty(x.sizes(), x.options().dtype(at::kChar));
    at::Tensor y_scale = at::empty(scale_shape, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnRmsNormCastQuant, x2, gamma, epsilon, y, y_fp32, y_int8, y_scale);
    return {y, y_fp32, y_int8, y_scale};
}
}  // namespace vllm_ascend
#endif
