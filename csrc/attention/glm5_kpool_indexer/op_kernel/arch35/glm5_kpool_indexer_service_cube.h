/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_service_cube.h
 * \brief Cube side: qbar[128,128] x poolK[128,128] Mmad per base block, K
 *        gathered page-by-page from the [blocks, poolsPerBlock, 1, 128] cache
 *        via block_table. Fixed 128x128x128 blocks (no g axis, no tail-block
 *        branches); row-level causality is masked later on the vector side.
 */

#ifndef GLM5_KPOOL_INDEXER_SERVICE_CUBE_H
#define GLM5_KPOOL_INDEXER_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../glm5_kpool_indexer_common.h"

namespace Glm5KpoolKernel {
using namespace Glm5KpoolCommon;

template <typename Q_T>
class Glm5KpoolServiceCube {
public:
    __aicore__ inline Glm5KpoolServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGlobalTensor(const GlobalTensor<int32_t> &blockTableGm,
                                            const GlobalTensor<Q_T> &cacheGm, const GlobalTensor<Q_T> &qbarGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &runInfo);

    static constexpr uint64_t KEY_BUF_NUM = 2;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID3;

    static constexpr uint64_t M_BLOCK = M_TILE;    // 128
    static constexpr uint64_t D_BLOCK = HEAD_DIM;  // 128
    static constexpr uint64_t S2_BLOCK = S2_TILE;  // 128

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BLOCK * D_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BLOCK * D_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BLOCK * D_BLOCK;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BLOCK * S2_BLOCK;

    static constexpr uint64_t FP16_BLOCK_CUBE = 16;
    static constexpr FixpipeConfig GLM5_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

protected:
    __aicore__ inline void Fixp(const RunInfo &runInfo);
    __aicore__ inline void ComputeL0c(const RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(const RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(const RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(const RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(const RunInfo &runInfo);

    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<Q_T> cacheGm_;
    GlobalTensor<Q_T> qbarGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<Q_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<Q_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<float> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t kl0BufIdx_ = 0;

    ConstInfo constInfo_{};
};

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::InitBuffers(TPipe *pipe)
{
    // AIV-consumed fp32 scores: dual-dst gives each paired AIV half of M rows,
    // so 64 rows x 128 pools x 4B x 2 buffers = 64KB per AIC pair.
    pipe->InitBuffer(bufUB_, 2 * (M_BLOCK / 2) * S2_BLOCK * sizeof(float));
    mm1ResUB_ = bufUB_.Get<float>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * QUERY_BUFFER_OFFSET * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * KEY_BUFFER_OFFSET * sizeof(Q_T));
    keyL1_ = bufKeyL1_.Get<Q_T>();
    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * L0AB_BUFFER_OFFSET * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BLOCK * S2_BLOCK * sizeof(Q_T));
    keyL0_ = bufKeyL0_.Get<Q_T>();
    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * L0C_BUFFER_OFFSET * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::InitGlobalTensor(const GlobalTensor<int32_t> &blockTableGm,
                                                                  const GlobalTensor<Q_T> &cacheGm,
                                                                  const GlobalTensor<Q_T> &qbarGm)
{
    blockTableGm_ = blockTableGm;
    cacheGm_ = cacheGm;
    qbarGm_ = qbarGm;
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::QueryNd2Nz(const RunInfo &runInfo)
{
    // qbar is padded to a multiple of M_TILE rows by the wrapper, so full
    // 128-row tiles are always in bounds.
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = M_BLOCK;
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             qbarGm_[runInfo.tensorQueryOffset], nd2nzPara);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::KeyNd2NzForPA(const RunInfo &runInfo)
{
    // S2_TILE=128 pools == 4 cache blocks of poolsPerBlock=32 (asserted by
    // tiling); each block is one contiguous 32x128 Nd2Nz copy. Out-of-range
    // pools read a clamped stale block whose lanes are masked on the AIV side.
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < S2_BLOCK) {
        uint64_t logicalPool = runInfo.s2Start + s2L1Offset;
        uint64_t s2BlkId = logicalPool / constInfo_.poolsPerBlock;
        uint64_t s2BlkOffset = logicalPool % constInfo_.poolsPerBlock;
        int32_t physBlock = blockTableGm_.GetValue(runInfo.reqIdx * constInfo_.blockTableStride + s2BlkId);
        physBlock = Max(Min(physBlock, static_cast<int32_t>(constInfo_.numCacheBlocks) - 1), 0);
        uint64_t keyGmOffset = static_cast<uint64_t>(physBlock) * constInfo_.poolsPerBlock * constInfo_.headDim +
                               s2BlkOffset * constInfo_.headDim;

        uint64_t s2Mte2Size = S2_BLOCK - s2L1Offset;
        s2Mte2Size = (s2BlkOffset + s2Mte2Size >= constInfo_.poolsPerBlock)
                         ? constInfo_.poolsPerBlock - s2BlkOffset
                         : s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(S2_BLOCK, (uint64_t)BLOCK_CUBE);
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * FP16_BLOCK_CUBE],
                 cacheGm_[keyGmOffset], nd2nzPara);
        s2L1Offset += s2Mte2Size;
    }
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::LoadQueryToL0a(const RunInfo &runInfo)
{
    (void)runInfo;
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = 0;
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(M_BLOCK, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(constInfo_.headDim, FP16_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(M_BLOCK, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(M_BLOCK, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;
    LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::LoadKeyToL0b(const RunInfo &runInfo)
{
    (void)runInfo;
    LoadData2DParamsV2 loadData2DParamsV2;
    loadData2DParamsV2.mStartPosition = 0;
    loadData2DParamsV2.kStartPosition = 0;
    loadData2DParamsV2.mStep = CeilDiv(S2_BLOCK, BLOCK_CUBE);
    loadData2DParamsV2.kStep = CeilDiv(constInfo_.headDim, FP16_BLOCK_CUBE);
    loadData2DParamsV2.srcStride = CeilDiv(S2_BLOCK, BLOCK_CUBE);
    loadData2DParamsV2.dstStride = CeilDiv(S2_BLOCK, BLOCK_CUBE);
    loadData2DParamsV2.ifTranspose = false;
    LoadData(keyL0_[(kl0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParamsV2);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::ComputeL0c(const RunInfo &runInfo)
{
    (void)runInfo;
    MmadParams mmadParams;
    mmadParams.m = M_BLOCK;
    mmadParams.n = S2_BLOCK;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(kl0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::Fixp(const RunInfo &runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT + l0BufIdx_ % L0_BUF_NUM);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.mSize = M_BLOCK;
    fixpipeParams.srcStride = M_BLOCK;
    fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(float); // keep one UB bank
    fixpipeParams.dualDstCtl = 1; // split M across the two paired AIVs
    // N = 128 floats = 512B: two ND chunks within one bank depth.
    fixpipeParams.nSize = S2_BLOCK / 2;
    fixpipeParams.params.ndNum = 2;
    fixpipeParams.params.srcNdStride = ((fixpipeParams.mSize + 15) / 16) * fixpipeParams.nSize;
    fixpipeParams.params.dstNdStride = S2_BLOCK * M_BLOCK / 2;
    Fixpipe<float, float, GLM5_CFG_ROW_MAJOR_UB>(mm1ResUB_[(runInfo.loop % 2) * (M_BLOCK / 2) * S2_BLOCK / 2],
                                                 cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], fixpipeParams);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::ComputeMm1(const RunInfo &runInfo)
{
    // Wait until both paired AIVs consumed the previous Mmad result.
    CrossCoreWaitFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT + runInfo.loop % 2);
    CrossCoreWaitFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_FIX>(
        ConstInfo::CROSS_VC_EVENT + runInfo.loop % 2 + ConstInfo::AIV0_AIV1_OFFSET);

    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
    KeyNd2NzForPA(runInfo);
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

    queryL1Mte2BufIdx_++;
    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
    QueryNd2Nz(runInfo);
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
    LoadQueryToL0a(runInfo);
    LoadKeyToL0b(runInfo);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
    ComputeL0c(runInfo);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

    Fixp(runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
    l0BufIdx_++;
    kl0BufIdx_++;

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
    keyL1BufIdx_++;

    // Notify both paired AIVs that fresh scores landed in their UB halves.
    CrossCoreSetFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_CV_EVENT + runInfo.loop % 2);
    CrossCoreSetFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_FIX>(
        ConstInfo::CROSS_CV_EVENT + runInfo.loop % 2 + ConstInfo::AIV0_AIV1_OFFSET);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_SERVICE_CUBE_H
