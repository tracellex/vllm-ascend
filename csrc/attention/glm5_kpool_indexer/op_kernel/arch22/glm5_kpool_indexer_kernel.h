/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_kernel.h
 * \brief arch22 main class (A3/ascend910_93 runtime path). The M axis is the
 *        concatenated token dimension (TND): split per request into M_TILE
 *        token tiles; each (req, mTile) unit owns ceil(visiblePools/S2_TILE)
 *        pool tiles. Units never span cores. AIC and its two AIVs advance in
 *        alternating lockstep over the same block sequence (vendored v1
 *        arch22 handshake): cube fix-pipes fp32 scores to a per-AIC GM strip,
 *        vectors fold them into per-row running top-k lists.
 */

#ifndef GLM5_KPOOL_INDEXER_KERNEL_ARCH22_H
#define GLM5_KPOOL_INDEXER_KERNEL_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "glm5_kpool_indexer_common.h"
#include "glm5_kpool_indexer_service_vector.h"
#include "glm5_kpool_indexer_service_cube.h"

namespace Glm5KpoolKernel {
using namespace Glm5KpoolCommon;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename Q_T>
class Glm5KpoolIndexerKernel {
public:
    __aicore__ inline Glm5KpoolIndexerKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *qbar, __gm__ uint8_t *indexerCache,
                                __gm__ uint8_t *cumQueryLens, __gm__ uint8_t *indexerSeqLens,
                                __gm__ uint8_t *indexerBlockTable, __gm__ uint8_t *positions,
                                __gm__ uint8_t *indicesOut, __gm__ uint8_t *scoresDebugOut,
                                __gm__ uint8_t *workspace, const Glm5KpoolTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitTilingData(const Glm5KpoolTilingData *__restrict tiling);
    __aicore__ inline uint32_t ReqTokenStart(uint32_t reqIdx) const;
    __aicore__ inline uint32_t ReqTokenLen(uint32_t reqIdx) const;
    __aicore__ inline uint32_t ReqPoolLen(uint32_t reqIdx) const;
    __aicore__ inline uint32_t UnitMTileNum(uint32_t reqIdx) const;
    __aicore__ inline uint32_t UnitS2TileNum(uint32_t reqIdx) const;
    __aicore__ inline uint32_t TotalUnits() const;
    __aicore__ inline uint32_t TotalS2Tiles() const;
    __aicore__ inline void LocateUnit(uint32_t unitIdx, uint32_t &reqIdx, uint32_t &mTileInReq) const;
    __aicore__ inline void SplitCore(uint32_t aiCoreIdx, uint32_t coreNum);
    __aicore__ inline void ProcessUnit(uint32_t unitIdx, uint32_t &loop);

    Glm5KpoolServiceCube<Q_T> matmulService;
    Glm5KpoolServiceVector<Q_T> vectorService;

    GlobalTensor<int32_t> cumQueryLensGm;
    GlobalTensor<int32_t> indexerSeqLensGm;
    GlobalTensor<float> mm1ResGm;

    ConstInfo constInfo_{};
    uint32_t aiCoreIdx_ = 0;
    uint32_t blockIdxRaw_ = 0;
    uint32_t unitFirst_ = 0;
    uint32_t unitLast_ = 0;
    bool coreEnable_ = false;
    // lockstep handshake events (arch22 mode2)
    uint32_t syncC1V1_ = 0;
    uint32_t syncV1C1_ = 0;
    TPipe *pipe_ = nullptr;
};

// ---------------------------------------------------------------- helpers

template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::InitTilingData(const Glm5KpoolTilingData *__restrict t)
{
    constInfo_.tSize = t->tSize;
    constInfo_.bSize = t->bSize;
    constInfo_.maxPoolSeqLen = t->maxPoolSeqLen;
    constInfo_.poolsPerBlock = t->poolsPerBlock;
    constInfo_.numCacheBlocks = t->numCacheBlocks;
    constInfo_.blockTableStride = t->blockTableStride;
    constInfo_.topkTokens = t->topkTokens;
    constInfo_.kpool = t->kpool;
    constInfo_.poolTopk = t->poolTopk;
    constInfo_.outputWidth = t->outputWidth;
    constInfo_.outputMode = t->outputMode;
    constInfo_.usedCoreNum = t->usedCoreNum;
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolIndexerKernel<Q_T>::ReqTokenStart(uint32_t reqIdx) const
{
    return reqIdx == 0 ? 0 : static_cast<uint32_t>(cumQueryLensGm.GetValue(reqIdx - 1));
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolIndexerKernel<Q_T>::ReqTokenLen(uint32_t reqIdx) const
{
    uint32_t end = static_cast<uint32_t>(cumQueryLensGm.GetValue(reqIdx));
    return end - ReqTokenStart(reqIdx);
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolIndexerKernel<Q_T>::ReqPoolLen(uint32_t reqIdx) const
{
    int32_t seqLen = indexerSeqLensGm.GetValue(reqIdx);
    uint32_t p = static_cast<uint32_t>(Max(seqLen, 0));
    return Min(p, constInfo_.maxPoolSeqLen);
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolIndexerKernel<Q_T>::UnitMTileNum(uint32_t reqIdx) const
{
    return CeilDiv(ReqTokenLen(reqIdx), M_TILE);
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolIndexerKernel<Q_T>::UnitS2TileNum(uint32_t reqIdx) const
{
    return CeilDiv(ReqPoolLen(reqIdx), S2_TILE);
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolIndexerKernel<Q_T>::TotalUnits() const
{
    uint32_t units = 0;
    for (uint32_t b = 0; b < constInfo_.bSize; b++) {
        units += UnitMTileNum(b);
    }
    return units;
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolIndexerKernel<Q_T>::TotalS2Tiles() const
{
    uint32_t tiles = 0;
    for (uint32_t b = 0; b < constInfo_.bSize; b++) {
        tiles += UnitMTileNum(b) * UnitS2TileNum(b);
    }
    return tiles;
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::LocateUnit(uint32_t unitIdx, uint32_t &reqIdx,
                                                               uint32_t &mTileInReq) const
{
    uint32_t acc = 0;
    for (uint32_t b = 0; b < constInfo_.bSize; b++) {
        uint32_t m = UnitMTileNum(b);
        if (unitIdx < acc + m) {
            reqIdx = b;
            mTileInReq = unitIdx - acc;
            return;
        }
        acc += m;
    }
    reqIdx = 0;
    mTileInReq = 0;
}

// Whole units to cores, contiguous in unit order, balanced by S2 tile count.
template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::SplitCore(uint32_t aiCoreIdx, uint32_t coreNum)
{
    uint32_t totalUnits = TotalUnits();
    coreEnable_ = false;
    unitFirst_ = 1;
    unitLast_ = 0;
    if (totalUnits == 0 || coreNum == 0) {
        return;
    }
    uint32_t totalS2 = TotalS2Tiles();
    uint32_t share = CeilDiv(totalS2, coreNum);

    uint32_t accS2 = 0;
    uint32_t coreCursor = 0;
    bool firstSet = false;
    for (uint32_t u = 0; u < totalUnits; u++) {
        uint32_t reqIdx, mTileInReq;
        LocateUnit(u, reqIdx, mTileInReq);
        uint32_t s2Num = UnitS2TileNum(reqIdx);
        if (coreCursor == aiCoreIdx && !firstSet) {
            unitFirst_ = u;
            firstSet = true;
        }
        accS2 += s2Num;
        bool shareDone = accS2 >= share;
        bool lastUnit = (u == totalUnits - 1);
        if (shareDone || lastUnit) {
            if (coreCursor == aiCoreIdx) {
                unitLast_ = u;
                coreEnable_ = true;
                return;
            }
            coreCursor++;
            accS2 = 0;
        }
    }
}

// ---------------------------------------------------------------- lifecycle

template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::Init(
    __gm__ uint8_t *qbar, __gm__ uint8_t *indexerCache, __gm__ uint8_t *cumQueryLens,
    __gm__ uint8_t *indexerSeqLens, __gm__ uint8_t *indexerBlockTable, __gm__ uint8_t *positions,
    __gm__ uint8_t *indicesOut, __gm__ uint8_t *scoresDebugOut, __gm__ uint8_t *workspace,
    const Glm5KpoolTilingData *__restrict tiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        blockIdxRaw_ = GetBlockIdx(); // vec: 0..2N-1
        aiCoreIdx_ = blockIdxRaw_ / 2;
    } else {
        blockIdxRaw_ = GetBlockIdx(); // cube: 0..N-1
        aiCoreIdx_ = blockIdxRaw_;
    }

    InitTilingData(tiling);
    cumQueryLensGm.SetGlobalBuffer((__gm__ int32_t *)cumQueryLens);
    indexerSeqLensGm.SetGlobalBuffer((__gm__ int32_t *)indexerSeqLens);
    SplitCore(aiCoreIdx_, constInfo_.usedCoreNum);
    syncC1V1_ = 0; // arch22 mode2 handshake: both directions share id 0 (v1 pattern)
    syncV1C1_ = 0;

    pipe_ = tPipe;

    // Workspace: per-AIC mm1Res double-buffered [2][M_TILE][S2_TILE] fp32.
    // The tiling sizes it as aicNum * 2 * M_TILE * S2_TILE * 4B.
    uint64_t stripSize = static_cast<uint64_t>(2) * M_TILE * S2_TILE * sizeof(float);
    mm1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + aiCoreIdx_ * stripSize));

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo_);
        vectorService.InitInputTensor(*reinterpret_cast<GlobalTensor<int32_t> *>(&indicesOut),
                                      *reinterpret_cast<GlobalTensor<float> *>(&scoresDebugOut),
                                      *reinterpret_cast<GlobalTensor<int32_t> *>(&positions), cumQueryLensGm,
                                      indexerSeqLensGm, mm1ResGm);
    } else {
        matmulService.InitParams(constInfo_);
        matmulService.InitGlobalTensor(*reinterpret_cast<GlobalTensor<int32_t> *>(&indexerBlockTable),
                                       *reinterpret_cast<GlobalTensor<Q_T> *>(&indexerCache),
                                       *reinterpret_cast<GlobalTensor<Q_T> *>(&qbar), mm1ResGm);
    }
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe_);
    } else {
        matmulService.InitBuffers(pipe_);
    }
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::ProcessUnit(uint32_t unitIdx, uint32_t &loop)
{
    uint32_t reqIdx, mTileInReq;
    LocateUnit(unitIdx, reqIdx, mTileInReq);
    uint32_t s2Num = UnitS2TileNum(reqIdx);

    RunInfo runInfo;
    runInfo.reqIdx = reqIdx;
    runInfo.mStart = ReqTokenStart(reqIdx) + mTileInReq * M_TILE;
    runInfo.actMSize = Min(M_TILE, ReqTokenLen(reqIdx) - mTileInReq * M_TILE);
    runInfo.tensorQueryOffset = static_cast<uint64_t>(runInfo.mStart) * HEAD_DIM;

    for (uint32_t s2Tile = 0; s2Tile < s2Num; s2Tile++) {
        runInfo.loop = loop;
        runInfo.s2TileIdx = s2Tile;
        runInfo.s2Start = s2Tile * S2_TILE;
        runInfo.actS2Size = Min(S2_TILE, ReqPoolLen(reqIdx) - runInfo.s2Start);
        runInfo.actS2SizeAlign = S2_TILE; // fixed width (F3-equivalent)
        runInfo.isFirstS2InnerLoop = (s2Tile == 0);
        runInfo.isLastS2InnerLoop = (s2Tile == s2Num - 1);
        runInfo.isValid = true;

        if ASCEND_IS_AIC {
            AscendC::PRINTF("AIC-PREWAIT loop=%u\n", runInfo.loop);
            CrossCoreWaitFlag(syncV1C1_);
            AscendC::PRINTF("AIC-MM1-BEGIN loop=%u\n", runInfo.loop);
            matmulService.ComputeMm1(runInfo);
            AscendC::PRINTF("AIC-MM1-DONE loop=%u\n", runInfo.loop);
            CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_FIX>(syncC1V1_);
            AscendC::PRINTF("AIC-SETFLAG loop=%u\n", runInfo.loop);
        } else {
            AscendC::PRINTF("AIV-PREWAIT loop=%u\n", runInfo.loop);
            CrossCoreWaitFlag(syncC1V1_);
            AscendC::PRINTF("AIV-POSTWAIT loop=%u\n", runInfo.loop);
            vectorService.ProcessVec(runInfo);
            AscendC::PRINTF("AIV-VEC-DONE loop=%u\n", runInfo.loop);
            if (runInfo.isLastS2InnerLoop) {
                vectorService.ProcessTopK(runInfo);
            }
            CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(syncV1C1_);
        }
        loop++;
    }
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::Process()
{
    if (!coreEnable_) {
        return;
    }

    if ASCEND_IS_AIV {
        AscendC::PRINTF("AIV-PROC-START aiv=%u\n", GetBlockIdx());
        vectorService.AllocEventID();
        AscendC::PRINTF("AIV-ALLOC-EV aiv=%u\n", GetBlockIdx());
        // prime the handshake: cube may compute block 0 immediately
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(syncV1C1_);
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(syncV1C1_);
        AscendC::PRINTF("AIV-PRIMED aiv=%u\n", GetBlockIdx());
    } else {
        matmulService.AllocEventID();
    }

    uint32_t loop = 0;
    for (uint32_t u = unitFirst_; u <= unitLast_; u++) {
        ProcessUnit(u, loop);
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(syncV1C1_);
        CrossCoreWaitFlag(syncV1C1_);
    }
}

} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_KERNEL_ARCH22_H
