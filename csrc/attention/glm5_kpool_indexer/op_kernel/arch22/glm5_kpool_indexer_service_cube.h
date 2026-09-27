/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_service_cube.h
 * \brief arch22 cube side: qbar[32,256] x poolK[128,256] Mmad per base
 *        block, K gathered page-by-page via block_table, result fixed-piped
 *        NZ2ND to the per-AIC mm1Res GM strip (quad-buffered) where the
 *        paired AIVs pick it up. reluPre stays 0: GLM5 scoring is linear.
 *        H9 precision: qbar rows are FP32-split [q_hi | q_lo] BF16 halves
 *        and the B operand is duplicated [K | K], so ONE k=256 Mmad computes
 *        q_hi@K + q_lo@K directly — ~16 mantissa bits of the head-weighted
 *        query survive with a single C tile / fixpipe drain and no
 *        vector-side add. (Accumulating dual-Mmads deadlock the cube next to
 *        the fixpipe on this stack; the two-C + AIV-Add form hit a Sort
 *        visibility hazard — both retired.)
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

template <typename Q_T, uint32_t M_TILE_SIZE>
class Glm5KpoolServiceCube {
public:
    __aicore__ inline Glm5KpoolServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGlobalTensor(const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<Q_T> &cacheGm,
                                            const GlobalTensor<Q_T> &qbarGm, const GlobalTensor<float> &mm1ResGm,
                                            const GlobalTensor<float> &scoresOutGm);
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
    // FIX->M: the Mmad pair zeroes its L0C tiles (cmatrixInitVal=true) while
    // the doubled fixpipe drain of the same-parity pair two tiles back can
    // still be reading them; without this flag the drain catches freshly
    // zeroed lanes (deterministic 7-lane lo-strip loss on probe row 4080).
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID4;

    static constexpr uint64_t M_BLOCK = M_TILE_SIZE;
    static constexpr uint64_t D_BLOCK = HEAD_DIM;  // 128
    static constexpr uint64_t S2_BLOCK = S2_TILE;  // 128
    // H9 k=256 form: one Mmad over the packed [q_hi | q_lo] x [K | K] inner
    // product (== q_hi@K + q_lo@K) restores FP32-level qbar fidelity with a
    // single C tile, a single fixpipe drain and no vector-side add. The
    // earlier dual-Mmad variants are retired: same-region accumulate
    // (cmatrixInitVal=false) deadlocks the cube next to the fixpipe, and the
    // two-C-tile + AIV Add form hit a c220 hazard where Sort reads scoreUb_
    // before the Add's ALU writes retire (deterministic rank-511/512 flips
    // on real replay data; only an MTE3 read of the Add's output closed it).
    static constexpr uint64_t QK_BLOCK = 2 * D_BLOCK; // packed [hi|lo] k width

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BLOCK * QK_BLOCK;
    static constexpr uint64_t KEY_K_BLOCK = S2_BLOCK * D_BLOCK; // one k=128 NZ block
    static constexpr uint64_t KEY_BUFFER_OFFSET = 2 * KEY_K_BLOCK; // [K | K]
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BLOCK * QK_BLOCK;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BLOCK * S2_BLOCK;

    static constexpr uint64_t FP16_BLOCK_CUBE = 16;
    static constexpr AscendC::IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;

protected:
    __aicore__ inline void Fixp(const RunInfo &runInfo);
    __aicore__ inline void ComputeL0c(const RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(const RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(const RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(const RunInfo &runInfo, uint64_t l1Slot);
    __aicore__ inline void KeyNd2NzForPA(const RunInfo &runInfo);

    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<Q_T> cacheGm_;
    GlobalTensor<Q_T> qbarGm_;
    GlobalTensor<float> mm1ResGm_;
    GlobalTensor<float> scoresOutGm_;

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

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * QUERY_BUFFER_OFFSET * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * KEY_BUFFER_OFFSET * sizeof(Q_T));
    keyL1_ = bufKeyL1_.Get<Q_T>();
    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * L0AB_BUFFER_OFFSET * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, 2 * D_BLOCK * S2_BLOCK * sizeof(Q_T)); // single [K|K] slot
    keyL0_ = bufKeyL0_.Get<Q_T>();
    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * L0C_BUFFER_OFFSET * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::InitGlobalTensor(
    const GlobalTensor<int32_t> &blockTableGm, const GlobalTensor<Q_T> &cacheGm,
    const GlobalTensor<Q_T> &qbarGm, const GlobalTensor<float> &mm1ResGm,
    const GlobalTensor<float> &scoresOutGm)
{
    blockTableGm_ = blockTableGm;
    cacheGm_ = cacheGm;
    qbarGm_ = qbarGm;
    mm1ResGm_ = mm1ResGm;
    scoresOutGm_ = scoresOutGm;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::QueryNd2Nz(const RunInfo &runInfo, uint64_t l1Slot)
{
    // qbar is padded to a multiple of M_TILE rows by the wrapper, so full
    // 128-row tiles are always in bounds. Rows are [q_hi | q_lo] packed and
    // copied as one contiguous 256-wide ND->NZ block (H9 k=256 form).
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = M_BLOCK;
    nd2nzPara.dValue = 2 * constInfo_.headDim;
    nd2nzPara.srcDValue = 2 * constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1_[l1Slot * QUERY_BUFFER_OFFSET], qbarGm_[runInfo.tensorQueryOffset], nd2nzPara);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::KeyNd2NzForPA(const RunInfo &runInfo)
{
    // S2_TILE=128 pools == 4 cache blocks of poolsPerBlock=32 (asserted by
    // tiling); each block is one contiguous 32x128 Nd2Nz copy. Out-of-range
    // pools read a clamped stale block whose lanes are masked on the AIV side.
    // H9 k=256: every chunk is written TWICE, at NZ-block offsets 0 and
    // KEY_K_BLOCK, so the L1 slot holds [K | K] matching the packed
    // [q_hi | q_lo] A operand.
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
        uint64_t slotBase = (keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * FP16_BLOCK_CUBE;
        DataCopy(keyL1_[slotBase], cacheGm_[keyGmOffset], nd2nzPara);
        DataCopy(keyL1_[slotBase + KEY_K_BLOCK], cacheGm_[keyGmOffset], nd2nzPara);
        s2L1Offset += s2Mte2Size;
    }
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::LoadQueryToL0a(const RunInfo &runInfo)
{
    // arch22 A0 loads MUST use the 3D (fmatrix) form: the arch35-style
    // LoadData2DParamsV2 path hangs the MTE1 queue on this core generation.
    // H9 k=256: the packed [q_hi | q_lo] A tile loads as one fmatrix with
    // Cin = 2*headDim into the parity ping-pong L0A slot (original 2-lag
    // reuse pattern restored).
    (void)runInfo;
    LoadData3DParamsV2<Q_T> loadData3DParams;
    loadData3DParams.l1H = CeilDiv(M_BLOCK, BLOCK_CUBE); // Hin = M1 blocks
    loadData3DParams.l1W = BLOCK_CUBE;                   // Win = M0
    loadData3DParams.channelSize = 2 * constInfo_.headDim; // Cin = 2K
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;
    loadData3DParams.mExtension = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    loadData3DParams.kExtension = 2 * constInfo_.headDim;
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

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::LoadKeyToL0b(const RunInfo &runInfo)
{
    // arch22 B0 uses the legacy 1D LoadData2DParams form (v1 arch22 pattern).
    // H9 k=256: 128 fractals load the whole [K | K] block into the single
    // 64KB L0B slot (B0 has no room to double-buffer at k=256); the 1-lag
    // M_MTE1 guard in ComputeMm1 covers the every-tile rewrite.
    (void)runInfo;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(S2_BLOCK, BLOCK_CUBE) * CeilDiv(2 * constInfo_.headDim, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_[0], keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParams);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::ComputeL0c(const RunInfo &runInfo)
{
    (void)runInfo;
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    mmadParams.n = S2_BLOCK;
    mmadParams.k = 2 * constInfo_.headDim; // H9: packed [q_hi|q_lo] x [K|K]
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], keyL0_[0], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::Fixp(const RunInfo &runInfo)
{
    AscendC::DataCopyCO12DstParams intriParams;
    const bool groupTopk = IsGroupTopkMode(constInfo_.outputMode);
    intriParams.mSize = M_BLOCK;
    intriParams.nSize = runInfo.actS2SizeAlign;
    intriParams.dstStride = groupTopk ? POOL_GROUP : runInfo.actS2SizeAlign;
    intriParams.srcStride = CeilAlign(M_BLOCK, (uint64_t)BLOCK_CUBE);
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11;
    intriParams.reluPre = 0; // linear scoring: NO relu (GLM5 differs from NSA)
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    if (groupTopk) {
        uint32_t groupOffset = runInfo.s2Start % POOL_GROUP;
        AscendC::DataCopy(mm1ResGm_[groupOffset], cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
                          intriParams);
    } else {
        uint64_t dstOffset = (runInfo.loop % 4) * M_BLOCK * S2_BLOCK;
        AscendC::DataCopy(mm1ResGm_[dstOffset], cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
    }
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::ComputeMm1(const RunInfo &runInfo)
{
#define GLMK_CUBE_STAGE 4 // 2=Nd2Nz only, 3=+Load/Mmad, 4=+Fixp (full)
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
    KeyNd2NzForPA(runInfo);
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

    // The packed [q_hi | q_lo] query loads as ONE 256-wide Nd2Nz per tile
    // into the parity ping-pong L1 slot (the original single-copy cadence).
    queryL1Mte2BufIdx_++;
    queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
    QueryNd2Nz(runInfo, queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
    SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

#if GLMK_CUBE_STAGE >= 3
    // L0B is a single [K | K] slot rewritten every tile, so the load must
    // wait on the immediately preceding Mmad's M_MTE1 flag; the M queue is
    // in order, so that wait also retires older Mmads and keeps their L0A /
    // L0C parity slots safe. Tile 0 owns a clean L0 and skips the wait.
    if (l0BufIdx_ >= 1) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + (l0BufIdx_ - 1) % L0_BUF_NUM);
    }
    LoadQueryToL0a(runInfo);
    LoadKeyToL0b(runInfo);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

#if GLMK_CUBE_STAGE >= 4
    // The Mmad re-initializes (zeroes) its L0C tile; hold it until the
    // fixpipe drain of the same-parity tile two back completed (arch35
    // FIX_M pattern — arch22 lacked it and raced under fixpipe pressure).
    // Tiles 0/1 own clean slots and skip the wait; nothing pre-arms it.
    if (l0BufIdx_ >= L0_BUF_NUM) {
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
    }
#endif
    ComputeL0c(runInfo);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
#endif

#if GLMK_CUBE_STAGE >= 4
    Fixp(runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0BufIdx_ % L0_BUF_NUM);
#endif
    l0BufIdx_++;
    kl0BufIdx_++;

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
    keyL1BufIdx_++;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceCube<Q_T, M_TILE_SIZE>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
    // Every Mmad pair's M_MTE1 flag is consumed by the next tile's load guard,
    // so only the final pair's flag is still outstanding at teardown; draining
    // anything older would wait on an already-consumed flag and hang.
    if (l0BufIdx_ > 0) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + (l0BufIdx_ - 1) % L0_BUF_NUM);
    }
    // FIX_M drain: pair L consumed Fixp(L-2)'s flag, so the final TWO fixpipe
    // flags (pairs n-2, n-1) are still outstanding.
    if (l0BufIdx_ > 0) {
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0BufIdx_ - 1) % L0_BUF_NUM);
    }
    if (l0BufIdx_ > 1) {
        WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0BufIdx_ - 2) % L0_BUF_NUM);
    }
}
} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_SERVICE_CUBE_H
