/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer.cpp
 * \brief M1 skeleton: the kernel only fills the indices output with -1 so the
 *        full def/tiling/aclnn/torch-binding chain can be built and invoked.
 *        M2 replaces this with the arch35/arch22 MIX_AIC_1_2 implementation
 *        (cube paged-gather matmul + vector radix-topk + pool expansion).
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace Glm5KpoolStub {
constexpr int32_t INVALID_IDX = -1;
constexpr uint32_t GM_ALIGN_BYTES = 512;

template <typename T>
__aicore__ inline T StubAlign(T num, T rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd);
}

// M1 stub kernel: AIV cores sweep the indices output and fill -1; cube cores
// return immediately (the task type stays MIX so the M2 tiling is unchanged).
class Glm5KpoolStubKernel {
public:
    __aicore__ inline Glm5KpoolStubKernel() = default;
    __aicore__ inline ~Glm5KpoolStubKernel() = default;

    __aicore__ inline void Init(__gm__ uint8_t *indices, const optiling::Glm5KpoolTilingData *__restrict tiling)
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
    const optiling::Glm5KpoolTilingData *__restrict tiling_ = nullptr;
    GlobalTensor<int32_t> indicesGm_;
};
} // namespace Glm5KpoolStub

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
    GET_TILING_DATA_WITH_STRUCT(optiling::Glm5KpoolTilingData, tilingDataIn, tiling);
    const optiling::Glm5KpoolTilingData *__restrict tilingData = &tilingDataIn;
    Glm5KpoolStub::Glm5KpoolStubKernel op;
    op.Init(indices, tilingData);
    op.Process();
    (void)tPipe;
}
