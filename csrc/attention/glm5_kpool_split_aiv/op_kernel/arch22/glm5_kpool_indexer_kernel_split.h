/**
 * SPDX-License-Identifier: Apache-2.0
 * \file glm5_kpool_indexer_kernel_split.h
 * \brief arch22 de-mixed launch pair (see experiments.md 2026-09-28):
 *        the fused MIX kernel excludes the HCCL-AIV communication stream for
 *        its whole 30-50ms engine-side duration (measured overlap 0.24ms over
 *        26 launches), which stalls the TP8 collectives and inflates chunk
 *        latency ~33% despite the kernel itself being ~50x faster than the
 *        Triton reference. Splitting into PURE kernels (AIC_ONLY scoring +
 *        AIV_ONLY folding, driven per pool batch by the wrapper) restores
 *        block-level co-scheduling with communication, matching the Triton
 *        path's scheduling shape while keeping the device-side top-k.
 *
 * Contract per batch b (pools [b*SPLIT_BATCH_POOLS, (b+1)*SPLIT_BATCH_POOLS)):
 *   AIC launch  — for every unit, mm1 every S2 tile of the batch window into
 *                 a per-AIC [M_TILE, SPLIT_BATCH_POOLS] GM strip (workspace).
 *   AIV launch  — for every unit: restore/load the per-row running top-512,
 *                 fold each 1024-pool group of the batch, then either save
 *                 the strip back to GM (mid batches) or emit (last batch).
 * Kernel boundaries are full barriers, so no cross-core flags remain: the
 * wrapper's stream order AIC(b) -> AIV(b) -> AIC(b+1) -> ... is the only
 * synchronization. Zero-work units still emit -1 rows on the AIV side.
 */

#ifndef GLM5_KPOOL_INDEXER_KERNEL_SPLIT_H
#define GLM5_KPOOL_INDEXER_KERNEL_SPLIT_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "glm5_kpool_indexer_common.h"
#include "glm5_kpool_indexer_service_vector.h"
#include "glm5_kpool_indexer_service_cube.h"

namespace Glm5KpoolKernel {
using namespace Glm5KpoolCommon;

// Shared geometry helpers (mirrored from the fused kernel; units never span
// requests, split batches never span the unit's pool range semantics).
template <typename Q_T, uint32_t M_TILE_SIZE>
class Glm5KpoolSplitBase {
public:
    __aicore__ inline Glm5KpoolSplitBase(){};

protected:
    __aicore__ inline void InitGeo(const Glm5KpoolTilingData *__restrict tiling,
                                   __gm__ uint8_t *cumQueryLens, __gm__ uint8_t *indexerSeqLens);
    __aicore__ inline uint32_t ReqTokenStart(uint32_t reqIdx) const;
    __aicore__ inline uint32_t ReqTokenLen(uint32_t reqIdx) const;
    __aicore__ inline uint32_t ReqPoolLen(uint32_t reqIdx) const;
    __aicore__ inline uint32_t UnitMTileNum(uint32_t reqIdx) const;
    __aicore__ inline uint32_t UnitS2TileNum(uint32_t reqIdx) const;
    __aicore__ inline uint32_t TotalUnits() const;
    __aicore__ inline uint32_t TotalS2Tiles() const;
    __aicore__ inline void LocateUnit(uint32_t unitIdx, uint32_t &reqIdx, uint32_t &mTileInReq) const;
    // Whole units to cores, contiguous, balanced by S2 tile count (fused rule).
    __aicore__ inline void SplitCore(uint32_t coreIdx, uint32_t coreNum);
    // Batch window (global pool range) this launch covers.
    __aicore__ inline uint32_t BatchFirstPool() const;
    __aicore__ inline uint32_t BatchPoolCount() const;

    ConstInfo constInfo_{};
    GlobalTensor<int32_t> cumQueryLensGm;
    GlobalTensor<int32_t> indexerSeqLensGm;
    uint32_t unitFirst_ = 0;
    uint32_t unitLast_ = 0;
    bool coreEnable_ = false;
};

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::InitGeo(const Glm5KpoolTilingData *__restrict t,
                                                                     __gm__ uint8_t *cumQueryLens,
                                                                     __gm__ uint8_t *indexerSeqLens)
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
    constInfo_.splitBatch = t->splitBatch;
    cumQueryLensGm.SetGlobalBuffer((__gm__ int32_t *)cumQueryLens);
    indexerSeqLensGm.SetGlobalBuffer((__gm__ int32_t *)indexerSeqLens);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::ReqTokenStart(uint32_t reqIdx) const
{
    return reqIdx == 0 ? 0 : static_cast<uint32_t>(cumQueryLensGm.GetValue((reqIdx - 1) * 8));
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::ReqTokenLen(uint32_t reqIdx) const
{
    uint32_t end = static_cast<uint32_t>(cumQueryLensGm.GetValue(reqIdx * 8));
    return end - ReqTokenStart(reqIdx);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::ReqPoolLen(uint32_t reqIdx) const
{
    int32_t seqLen = indexerSeqLensGm.GetValue(reqIdx * 8);
    uint32_t p = static_cast<uint32_t>(Max(seqLen, 0));
    return Min(p, constInfo_.maxPoolSeqLen);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::UnitMTileNum(uint32_t reqIdx) const
{
    return CeilDiv(ReqTokenLen(reqIdx), M_TILE_SIZE);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::UnitS2TileNum(uint32_t reqIdx) const
{
    return CeilDiv(ReqPoolLen(reqIdx), S2_TILE);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::TotalUnits() const
{
    uint32_t units = 0;
    for (uint32_t b = 0; b < constInfo_.bSize; b++) {
        units += UnitMTileNum(b);
    }
    return units;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::TotalS2Tiles() const
{
    uint32_t tiles = 0;
    for (uint32_t b = 0; b < constInfo_.bSize; b++) {
        tiles += UnitMTileNum(b) * UnitS2TileNum(b);
    }
    return tiles;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::LocateUnit(uint32_t unitIdx, uint32_t &reqIdx,
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

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::SplitCore(uint32_t coreIdx, uint32_t coreNum)
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
        if (coreCursor == coreIdx && !firstSet) {
            unitFirst_ = u;
            firstSet = true;
        }
        accS2 += s2Num;
        bool shareDone = accS2 >= share;
        bool lastUnit = (u == totalUnits - 1);
        if (shareDone || lastUnit) {
            if (coreCursor == coreIdx) {
                unitLast_ = u;
                coreEnable_ = true;
                return;
            }
            coreCursor++;
            accS2 = 0;
        }
    }
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::BatchFirstPool() const
{
    return constInfo_.splitBatch * SPLIT_BATCH_POOLS;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::BatchPoolCount() const
{
    return SPLIT_BATCH_POOLS;
}

// ---------------------------------------------------------------------------
// AIC_ONLY scoring kernel: mm1 for every unit x every S2 tile of the batch
// window, fix-piped into the per-AIC batch strip.
// ---------------------------------------------------------------------------
template <typename Q_T, uint32_t M_TILE_SIZE>
class Glm5KpoolSplitAicKernel : public Glm5KpoolSplitBase<Q_T, M_TILE_SIZE> {
public:
    __aicore__ inline void Init(__gm__ uint8_t *qbar, __gm__ uint8_t *indexerCache,
                                __gm__ uint8_t *cumQueryLens, __gm__ uint8_t *indexerSeqLens,
                                __gm__ uint8_t *indexerBlockTable, __gm__ uint8_t *positions,
                                __gm__ uint8_t *scoresOut, const Glm5KpoolTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::constInfo_;
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::cumQueryLensGm;
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::unitFirst_;
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::unitLast_;
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::coreEnable_;
    Glm5KpoolServiceCube<Q_T, M_TILE_SIZE> matmulService;
    __gm__ float *scoresBase_ = nullptr; // output [T_pad, SPLIT_BATCH_POOLS] row-major
};

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolSplitAicKernel<Q_T, M_TILE_SIZE>::Init(
    __gm__ uint8_t *qbar, __gm__ uint8_t *indexerCache, __gm__ uint8_t *cumQueryLens,
    __gm__ uint8_t *indexerSeqLens, __gm__ uint8_t *indexerBlockTable, __gm__ uint8_t *positions,
    __gm__ uint8_t *scoresOut, const Glm5KpoolTilingData *__restrict tiling, TPipe *tPipe)
{
    (void)positions;
    uint32_t aiCoreIdx = GetBlockIdx();
    this->InitGeo(tiling, cumQueryLens, indexerSeqLens);
    this->SplitCore(aiCoreIdx, constInfo_.usedCoreNum);
    // Output strip is a caller-owned row-major [T_pad, SPLIT_BATCH_POOLS]
    // tensor: each unit relocates to its mStart row before its tile loop.
    scoresBase_ = (__gm__ float *)scoresOut;
    GlobalTensor<float> mm1ResGm;
    mm1ResGm.SetGlobalBuffer(scoresBase_);
    GlobalTensor<int32_t> blockTableGm;
    blockTableGm.SetGlobalBuffer((__gm__ int32_t *)indexerBlockTable);
    GlobalTensor<Q_T> cacheGm;
    cacheGm.SetGlobalBuffer((__gm__ Q_T *)indexerCache);
    GlobalTensor<Q_T> qbarGm;
    qbarGm.SetGlobalBuffer((__gm__ Q_T *)qbar);
    GlobalTensor<float> scoresGm; // unused on this path
    matmulService.InitParams(constInfo_);
    matmulService.InitGlobalTensor(blockTableGm, cacheGm, qbarGm, mm1ResGm, scoresGm);
    matmulService.InitBuffers(tPipe);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolSplitAicKernel<Q_T, M_TILE_SIZE>::Process()
{
    if (!coreEnable_) {
        return;
    }
    matmulService.AllocEventID();
    uint32_t loop = 0;
    const uint32_t batchFirst = this->BatchFirstPool();
    for (uint32_t u = unitFirst_; u <= unitLast_; u++) {
        uint32_t reqIdx, mTileInReq;
        this->LocateUnit(u, reqIdx, mTileInReq);
        const uint32_t poolLen = this->ReqPoolLen(reqIdx);
        // S2 tile window of this unit covered by the batch: tiles are global
        // per request ([0, ceil(poolLen/S2_TILE))); intersect with the batch.
        const uint32_t firstTile = batchFirst / S2_TILE;
        const uint32_t endPool = Min(poolLen, batchFirst + SPLIT_BATCH_POOLS);
        const uint32_t lastTileExcl = CeilDiv(endPool, S2_TILE);
        RunInfo runInfo;
        runInfo.reqIdx = reqIdx;
        runInfo.mStart = this->ReqTokenStart(reqIdx) + mTileInReq * M_TILE_SIZE;
        runInfo.actMSize = Min(M_TILE_SIZE, this->ReqTokenLen(reqIdx) - mTileInReq * M_TILE_SIZE);
        runInfo.reqPoolLen = poolLen;
        runInfo.posBase = mTileInReq * M_TILE_SIZE;
        runInfo.tensorQueryOffset = static_cast<uint64_t>(runInfo.mStart) * QBAR_ROW_ELEMS;
        matmulService.RelocateMm1Res(scoresBase_ + static_cast<uint64_t>(runInfo.mStart) * SPLIT_BATCH_POOLS);
        for (uint32_t tile = firstTile; tile < lastTileExcl; tile++) {
            runInfo.loop = loop++;
            runInfo.s2TileIdx = tile;
            runInfo.s2Start = tile * S2_TILE;
            runInfo.actS2Size = Min(S2_TILE, poolLen - runInfo.s2Start);
            runInfo.actS2SizeAlign = S2_TILE;
            matmulService.ComputeMm1(runInfo);
        }
    }
    matmulService.FreeEventID();
}

// ---------------------------------------------------------------------------
// AIV_ONLY folding kernel: per unit restore (or first-init) the running
// top-512, fold the batch's 1024-pool groups, save the strip back or emit.
// ---------------------------------------------------------------------------
template <typename Q_T, uint32_t M_TILE_SIZE>
class Glm5KpoolSplitAivKernel : public Glm5KpoolSplitBase<Q_T, M_TILE_SIZE> {
public:
    __aicore__ inline void Init(__gm__ uint8_t *indicesOut, __gm__ uint8_t *cumQueryLens,
                                __gm__ uint8_t *indexerSeqLens, __gm__ uint8_t *positions,
                                __gm__ uint8_t *scoresIn, __gm__ uint8_t *runningStrip,
                                const Glm5KpoolTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::constInfo_;
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::unitFirst_;
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::unitLast_;
    using Glm5KpoolSplitBase<Q_T, M_TILE_SIZE>::coreEnable_;
    Glm5KpoolServiceVector<Q_T, M_TILE_SIZE> vectorService;
    __gm__ float *scoresBase_ = nullptr;  // input [T_pad, SPLIT_BATCH_POOLS]
    uint32_t aivHalf_ = 0;
};

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolSplitAivKernel<Q_T, M_TILE_SIZE>::Init(
    __gm__ uint8_t *indicesOut, __gm__ uint8_t *cumQueryLens, __gm__ uint8_t *indexerSeqLens,
    __gm__ uint8_t *positions, __gm__ uint8_t *scoresIn, __gm__ uint8_t *runningStrip,
    const Glm5KpoolTilingData *__restrict tiling, TPipe *tPipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    aivHalf_ = blockIdx % 2;
    const uint32_t aiCoreIdx = blockIdx / 2;
    this->InitGeo(tiling, cumQueryLens, indexerSeqLens);
    this->SplitCore(aiCoreIdx, constInfo_.usedCoreNum);
    // scoresIn mirrors the AIC op's output tensor (row-major, per-unit rows);
    // runningStrip is the wrapper-owned [T_pad, poolTopk*2] fp32 staging pair
    // tensor shared across the batch launches.
    scoresBase_ = (__gm__ float *)scoresIn;
    GlobalTensor<float> mm1ResGm;
    mm1ResGm.SetGlobalBuffer(scoresBase_);
    GlobalTensor<float> stripGm;
    stripGm.SetGlobalBuffer((__gm__ float *)runningStrip);
    vectorService.InitParams(constInfo_);
    vectorService.SetSplitStripGm(stripGm);
    GlobalTensor<int32_t> outGm;
    outGm.SetGlobalBuffer((__gm__ int32_t *)indicesOut);
    GlobalTensor<float> scoresGm; // unused on this path
    GlobalTensor<int32_t> posGm;
    posGm.SetGlobalBuffer((__gm__ int32_t *)positions);
    vectorService.InitInputTensor(outGm, scoresGm, posGm, this->cumQueryLensGm, this->indexerSeqLensGm,
                                  mm1ResGm);
    vectorService.InitBuffers(tPipe);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolSplitAivKernel<Q_T, M_TILE_SIZE>::Process()
{
    if (!coreEnable_) {
        return;
    }
    vectorService.AllocEventID();
    const uint32_t batchFirst = this->BatchFirstPool();
    const bool firstBatch = batchFirst == 0;
    // Last batch for any request: the wrapper always launches the batch pair
    // up to ceil(maxPoolSeqLen / SPLIT_BATCH_POOLS); the emit batch is the
    // final one of the whole launch sequence (per-request tails are masked by
    // visibility, so a global last-batch check suffices).
    const uint32_t maxPools = constInfo_.maxPoolSeqLen;
    const bool lastBatch = batchFirst + SPLIT_BATCH_POOLS >= maxPools;
    uint32_t loop = 0;
    for (uint32_t u = unitFirst_; u <= unitLast_; u++) {
        uint32_t reqIdx, mTileInReq;
        this->LocateUnit(u, reqIdx, mTileInReq);
        RunInfo runInfo;
        runInfo.reqIdx = reqIdx;
        runInfo.mStart = this->ReqTokenStart(reqIdx) + mTileInReq * M_TILE_SIZE;
        runInfo.actMSize = Min(M_TILE_SIZE, this->ReqTokenLen(reqIdx) - mTileInReq * M_TILE_SIZE);
        runInfo.reqPoolLen = this->ReqPoolLen(reqIdx);
        runInfo.posBase = mTileInReq * M_TILE_SIZE;

        const uint32_t poolLen = runInfo.reqPoolLen;
        const uint32_t endPool = Min(poolLen, batchFirst + SPLIT_BATCH_POOLS);
        const uint32_t batchIdx = batchFirst / SPLIT_BATCH_POOLS;
        if (poolLen == 0 || endPool <= batchFirst) {
            // No pools in this batch. poolLen == 0: only the global last
            // batch owns the -1 emission. poolLen > 0 but this batch is past
            // the request's whole pool range: the strip staged by the
            // request's last fold batch (which was not the global last
            // batch, else it emitted inline) must still be emitted exactly
            // once — on the first batch past the request's pool range.
            if (poolLen == 0) {
                if (lastBatch) {
                    vectorService.ProcessEmptyUnit(runInfo);
                }
            } else if (!firstBatch && batchIdx == CeilDiv(poolLen, SPLIT_BATCH_POOLS)) {
                vectorService.LoadRowMeta(runInfo, false);
                vectorService.RestoreRunningStrip(runInfo);
                vectorService.ProcessTopK(runInfo);
            }
            continue;
        }
        vectorService.RelocateMm1Res(scoresBase_ + static_cast<uint64_t>(runInfo.mStart) * SPLIT_BATCH_POOLS);
        vectorService.LoadRowMeta(runInfo, firstBatch);
        if (!firstBatch) {
            vectorService.RestoreRunningStrip(runInfo);
        }
        constexpr uint32_t TILES_PER_GROUP = POOL_GROUP / S2_TILE;
        const uint32_t firstGroup = batchFirst / POOL_GROUP;
        const uint32_t endGroupExcl = CeilDiv(endPool, POOL_GROUP);
        for (uint32_t g = firstGroup; g < endGroupExcl; g++) {
            const uint32_t tileFirst = g * TILES_PER_GROUP;
            const uint32_t tileEndExcl = Min((g + 1) * TILES_PER_GROUP, CeilDiv(poolLen, S2_TILE));
            RunInfo groupInfo = runInfo;
            groupInfo.loop = loop;
            groupInfo.s2TileIdx = tileFirst;
            groupInfo.s2Start = g * POOL_GROUP;
            groupInfo.actS2Size = Min(POOL_GROUP, poolLen - groupInfo.s2Start);
            groupInfo.actS2SizeAlign = S2_TILE;
            // LoadRowMeta (+ Restore) already ran above; true here would
            // re-init the strip and wipe the restored cross-batch state.
            groupInfo.isFirstS2InnerLoop = false;
            groupInfo.isLastS2InnerLoop = false;
            groupInfo.isValid = true;
            (void)tileEndExcl;
            vectorService.ProcessPoolGroup(groupInfo);
            loop += TILES_PER_GROUP;
        }
        if (lastBatch) {
            vectorService.ProcessTopK(runInfo);
        } else {
            vectorService.SaveRunningStrip(runInfo);
        }
    }
    vectorService.FreeEventID();
}

} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_KERNEL_SPLIT_H
