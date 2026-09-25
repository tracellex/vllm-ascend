/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_service_cube.h
 * \brief arch22 cube side: qbar[128,128] x poolK[128,128] Mmad per base
 *        block, K gathered page-by-page via block_table, result fixed-piped
 *        NZ2ND to the per-AIC mm1Res GM strip (double-buffered) where the
 *        paired AIVs pick it up. reluPre stays 0: GLM5 scoring is linear.
 */

#ifndef GLM5_KPOOL_INDEXER_SERVICE_CUBE_H
#define GLM5_KPOOL_INDEXER_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "glm5_kpool_indexer_common.h"

namespace Glm5KpoolKernel {
using namespace Glm5KpoolCommon;

template <typename Q_T>
class Glm5KpoolServiceCube {
public:
    __aicore__ inline Glm5KpoolServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGlobalTensor(const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<Q_T> &cacheGm,
                                            const GlobalTensor<Q_T> &qbarGm, const GlobalTensor<float> &mm1ResGm);
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

    static constexpr uint64_t M_BLOCK = M_TILE;    // 128
    static constexpr uint64_t D_BLOCK = HEAD_DIM;  // 128
    static constexpr uint64_t S2_BLOCK = S2_TILE;  // 128

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BLOCK * D_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BLOCK * D_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BLOCK * D_BLOCK;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BLOCK * S2_BLOCK;

    static constexpr uint64_t FP16_BLOCK_CUBE = 16;
    static constexpr AscendC::IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;

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
    GlobalTensor<float> mm1ResGm_; // per-AIC [2][M_TILE][S2_TILE] fp32

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
                                                                  const GlobalTensor<Q_T> &qbarGm,
                                                                  const GlobalTensor<float> &mm1ResGm)
{
    blockTableGm_ = blockTableGm;
    cacheGm_ = cacheGm;
    qbarGm_ = qbarGm;
    mm1ResGm_ = mm1ResGm;
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
    // arch22 A0 loads MUST use the 3D (fmatrix) form: the arch35-style
    // LoadData2DParamsV2 path hangs the MTE1 queue on this core generation.
    (void)runInfo;
    LoadData3DParamsV2<Q_T> loadData3DParams;
    loadData3DParams.l1H = CeilDiv(M_BLOCK, BLOCK_CUBE); // Hin = M1 blocks
    loadData3DParams.l1W = BLOCK_CUBE;                   // Win = M0
    loadData3DParams.channelSize = constInfo_.headDim;   // Cin = K
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;
    loadData3DParams.mExtension = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    loadData3DParams.kExtension = constInfo_.headDim;
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    LoadData<Q_T, LOAD3DV2_CONFIG>(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
                                   queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
                                   loadData3DParams);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::LoadKeyToL0b(const RunInfo &runInfo)
{
    // arch22 B0 uses the legacy 1D LoadData2DParams form (v1 arch22 pattern).
    (void)runInfo;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(S2_BLOCK, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_[(kl0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParams);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::ComputeL0c(const RunInfo &runInfo)
{
    (void)runInfo;
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    mmadParams.n = S2_BLOCK;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
         keyL0_[(kl0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::Fixp(const RunInfo &runInfo)
{
    // NZ2ND fixpipe straight to the per-AIC GM strip; the paired AIVs read
    // their row halves from there (arch22 has no UB dual-dst fixpipe).
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = M_BLOCK;
    intriParams.nSize = runInfo.actS2SizeAlign;
    intriParams.dstStride = runInfo.actS2SizeAlign;
    intriParams.srcStride = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11;
    intriParams.reluPre = 0; // linear scoring: NO relu (GLM5 differs from NSA)
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    uint64_t dstOffset = (runInfo.loop % 2) * M_BLOCK * S2_BLOCK;
    AscendC::DataCopy(mm1ResGm_[dstOffset], cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceCube<Q_T>::ComputeMm1(const RunInfo &runInfo)
{
#define GLMK_CUBE_STAGE 4 // 2=Nd2Nz only, 3=+Load/Mmad, 4=+Fixp (full)
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
    KeyNd2NzForPA(runInfo);
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    AscendC::PRINTF("CUBE-KEY loop=%u\n", runInfo.loop);

    queryL1Mte2BufIdx_++;
    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
    QueryNd2Nz(runInfo);
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    AscendC::PRINTF("CUBE-QUERY loop=%u\n", runInfo.loop);

#if GLMK_CUBE_STAGE >= 3
    // The AllocEventID pre-set of M_MTE1 never fires on this stack when the M
    // queue has no activity yet; the first L0_BUF_NUM blocks own a clean L0,
    // so skipping the wait there removes the dependency on that pre-set.
    if (l0BufIdx_ >= L0_BUF_NUM) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
    }
    LoadQueryToL0a(runInfo);
    AscendC::PRINTF("CUBE-LQA loop=%u\n", runInfo.loop);
    LoadKeyToL0b(runInfo);
    AscendC::PRINTF("CUBE-LKB loop=%u\n", runInfo.loop);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    AscendC::PRINTF("CUBE-SETM1 loop=%u\n", runInfo.loop);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    AscendC::PRINTF("CUBE-LOAD loop=%u\n", runInfo.loop);

    ComputeL0c(runInfo);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
#endif

#if GLMK_CUBE_STAGE >= 4
    Fixp(runInfo);
    AscendC::PRINTF("CUBE-FIXP loop=%u\n", runInfo.loop);
#endif
    l0BufIdx_++;
    kl0BufIdx_++;

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
    keyL1BufIdx_++;
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
}
} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_SERVICE_CUBE_H
