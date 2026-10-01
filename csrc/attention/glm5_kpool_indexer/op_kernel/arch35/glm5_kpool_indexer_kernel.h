/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_kernel.h
 * \brief arch35 main class. The M axis is the concatenated token dimension
 *        (TND): it is split per request into 128-row token tiles, and each
 *        (req, mTile) unit owns ceil(visiblePools/128) pool tiles. Units are
 *        never split across cores (no LD merge in M2), so a core's top-k sees
 *        the whole S2 range of its rows in its private score strip.
 */

#ifndef GLM5_KPOOL_INDEXER_KERNEL_ARCH35_H
#define GLM5_KPOOL_INDEXER_KERNEL_ARCH35_H

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
    __aicore__ inline uint32_t ReqPoolLen(uint32_t reqIdx) const; // clamped to [0, maxPoolSeqLen]
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

    ConstInfo constInfo_{};
    uint32_t aiCoreIdx_ = 0;
    uint32_t blockIdxRaw_ = 0;
    uint32_t unitFirst_ = 0;
    uint32_t unitLast_ = 0; // 双闭; unitLast_ < unitFirst_ means disabled
    bool coreEnable_ = false;
    TPipe *pipe_ = nullptr;
};

// ---------------------------------------------------------------- helpers

template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::InitTilingData(const Glm5KpoolTilingData *__restrict t)
{
    constInfo_.tSize = t->tSize;
    constInfo_.bSize = t->bSize;
    constInfo_.maxPoolSeqLen = t->maxPoolSeqLen;
    constInfo_.maxPoolSeqLenAlign = Align(t->maxPoolSeqLen, S2_TILE);
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

// Map a global unit index to (reqIdx, mTileInReq) by sequential scan. Unit
// counts are small (<= ~64 at 8K-token chunks), the scan is cheap.
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

// Units are assigned contiguously in unit order; a unit's S2 tiles never span
// cores. Balance by S2 tile count: each core takes whole units until its
// share of the total S2 tiles is reached.
template <typename Q_T>
__aicore__ inline void Glm5KpoolIndexerKernel<Q_T>::SplitCore(uint32_t aiCoreIdx, uint32_t coreNum)
{
    uint32_t totalUnits = TotalUnits();
    if (totalUnits == 0 || coreNum == 0) {
        coreEnable_ = false;
        unitFirst_ = 1;
        unitLast_ = 0;
        return;
    }
    uint32_t totalS2 = TotalS2Tiles();
    uint32_t share = CeilDiv(totalS2, coreNum);

    uint32_t accS2 = 0;
    uint32_t unitFirst = 0;
    uint32_t coreCursor = 0;
    unitFirst_ = 1;
    unitLast_ = 0;
    for (uint32_t u = 0; u < totalUnits; u++) {
        uint32_t reqIdx, mTileInReq;
        LocateUnit(u, reqIdx, mTileInReq);
        uint32_t s2Num = UnitS2TileNum(reqIdx);
        if (coreCursor == aiCoreIdx && unitFirst_ > unitLast_) {
            unitFirst_ = u; // first unit of my range
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
    coreEnable_ = false;
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

    pipe_ = tPipe;

    // Per-core score strip: [aicIdx * M_TILE * maxPoolAlign, +M_TILE rows).
    GlobalTensor<uint32_t> scoreGm;
    uint64_t stripBytes = static_cast<uint64_t>(M_TILE) * constInfo_.maxPoolSeqLenAlign * sizeof(uint32_t);
    scoreGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + aiCoreIdx_ * stripBytes));

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo_);
        vectorService.InitInputTensor(
            *reinterpret_cast<GlobalTensor<int32_t> *>(&indicesOut),
            *reinterpret_cast<GlobalTensor<float> *>(&scoresDebugOut),
            *reinterpret_cast<GlobalTensor<int32_t> *>(&positions), cumQueryLensGm, indexerSeqLensGm);
        vectorService.InitWorkspaceTensor(scoreGm);
    } else {
        matmulService.InitParams(constInfo_);
        matmulService.InitGlobalTensor(*reinterpret_cast<GlobalTensor<int32_t> *>(&indexerBlockTable),
                                       *reinterpret_cast<GlobalTensor<Q_T> *>(&indexerCache),
                                       *reinterpret_cast<GlobalTensor<Q_T> *>(&qbar));
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
    runInfo.scoreRowOffset = static_cast<uint64_t>(aiCoreIdx_) * M_TILE * constInfo_.maxPoolSeqLenAlign;

    for (uint32_t s2Tile = 0; s2Tile < s2Num; s2Tile++) {
        runInfo.loop = loop;
        runInfo.s2TileIdx = s2Tile;
        runInfo.s2Start = s2Tile * S2_TILE;
        runInfo.actS2Size = Min(S2_TILE, ReqPoolLen(reqIdx) - runInfo.s2Start);
        runInfo.actS2SizeAlign = Align(runInfo.actS2Size, BUFFER_SIZE_BYTE_32B / sizeof(float));
        runInfo.isFirstS2InnerLoop = (s2Tile == 0);
        runInfo.isLastS2InnerLoop = (s2Tile == s2Num - 1);
        runInfo.isValid = true;

        if ASCEND_IS_AIC {
            matmulService.ComputeMm1(runInfo);
        } else {
            vectorService.ProcessVec1(runInfo);
            if (runInfo.isLastS2InnerLoop) {
                vectorService.ProcessTopK(runInfo);
            }
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
        vectorService.AllocEventID();
        CrossCoreSetFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_VC_EVENT + 1);
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
        CrossCoreWaitFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_FIX>(ConstInfo::CROSS_VC_EVENT + 1);
    }
}

} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_KERNEL_ARCH35_H
