/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer.cpp
 * \brief M1 skeleton: the kernel only fills the indices output with -1 so the
 *        full def/tiling/aclnn/torch-binding chain can be built and invoked.
 *        M2 replaces the stub body with the arch35/arch22 MIX_AIC_1_2
 *        implementation (cube paged-gather matmul + vector radix-topk).
 *
 * The int template parameters mirror ASCENDC_TPL_ARGS_DECL in
 * glm5_kpool_indexer_template_tiling_key.h (must be included here!): ccec
 * instantiates one kernel variant per ASCENDC_TPL_SEL combination and stamps
 * its tilingKey into the binary kernelList. A non-template entry — or a
 * missing tiling-key header — leaves kernelList[].tilingKey at 0 and the
 * executor fails at launch (NnopbaseExecutorGetCoreTypeAndTaskRation).
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "glm5_kpool_indexer_template_tiling_key.h"

using namespace AscendC;

namespace Glm5KpoolStub {
constexpr int32_t INVALID_IDX = -1;
constexpr uint32_t GM_ALIGN_BYTES = 512;

template <typename Q_T>
struct Glm5KType {
    using queryType = Q_T;
};

template <typename T>
__aicore__ inline T StubAlign(T num, T rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd);
}

// M1 stub kernel: AIV cores sweep the indices output and fill -1; cube cores
// return immediately (the task type stays MIX so the M2 tiling is unchanged).
template <typename LIT>
class Glm5KpoolStubKernel {
public:
    __aicore__ inline Glm5KpoolStubKernel() = default;
    __aicore__ inline ~Glm5KpoolStubKernel() = default;

    __aicore__ inline void Init(__gm__ uint8_t *indices, const Glm5KpoolTilingData *__restrict tiling)
    {
        tiling_ = tiling;
        if ASCEND_IS_AIV {
            indicesGm_.SetGlobalBuffer((__gm__ int32_t *)indices);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
            uint64_t totalSize = static_cast<uint64_t>(tiling_->tSize) * tiling_->outputWidth;
            uint64_t singleCoreSize =
                StubAlign<uint64_t>((totalSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(int32_t));
            uint64_t baseSize = static_cast<uint64_t>(GetBlockIdx()) * singleCoreSize;
            if (baseSize >= totalSize) {
                return;
            }
            uint64_t dealSize = (baseSize + singleCoreSize <= totalSize) ? singleCoreSize : totalSize - baseSize;
            GlobalTensor<int32_t> output = indicesGm_[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, INVALID_IDX);
        }
    }

private:
    const Glm5KpoolTilingData *__restrict tiling_ = nullptr;
    GlobalTensor<int32_t> indicesGm_;
};

#define INVOKE_GLMK_STUB_IMPL(templateClass, ...)                                                                       \
    do {                                                                                                                \
        templateClass<Glm5KType<__VA_ARGS__>> op;                                                                       \
        GET_TILING_DATA_WITH_STRUCT(Glm5KpoolTilingData, tiling_data_in, tiling);                                       \
        const Glm5KpoolTilingData *__restrict tiling_data = &tiling_data_in;                                            \
        op.Init(indices, tiling_data);                                                                                  \
        op.Process();                                                                                                   \
    } while (0)
} // namespace Glm5KpoolStub
using namespace Glm5KpoolStub;

template <int DT_Q>
__global__ __aicore__ void glm5_kpool_indexer(__gm__ uint8_t *qbar, __gm__ uint8_t *indexerCache,
                                              __gm__ uint8_t *cumQueryLens, __gm__ uint8_t *indexerSeqLens,
                                              __gm__ uint8_t *indexerBlockTable, __gm__ uint8_t *positions,
                                              __gm__ uint8_t *indices, __gm__ uint8_t *scoresDebug,
                                              __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    // 引入未使用参数避免告警；M2 实现会消费全部输入。
    (void)qbar;
    (void)indexerCache;
    (void)cumQueryLens;
    (void)indexerSeqLens;
    (void)indexerBlockTable;
    (void)positions;
    (void)scoresDebug;
    (void)workspace;
    TPipe tPipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
    if (ORIG_DTYPE_QBAR == DT_BF16) {
        INVOKE_GLMK_STUB_IMPL(Glm5KpoolStubKernel, bfloat16_t);
    } else {
        INVOKE_GLMK_STUB_IMPL(Glm5KpoolStubKernel, half);
    }
#else
    if constexpr (DT_Q == GLMK_TPL_FP16) {
        INVOKE_GLMK_STUB_IMPL(Glm5KpoolStubKernel, half);
    } else {
        INVOKE_GLMK_STUB_IMPL(Glm5KpoolStubKernel, bfloat16_t);
    }
#endif
    (void)tPipe;
}
