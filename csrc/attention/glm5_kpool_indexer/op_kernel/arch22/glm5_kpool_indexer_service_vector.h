/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_service_vector.h
 * \brief arch22 vector side. Per base block each AIV reads its half of the
 *        cube's mm1Res GM strip (fp32 scores), folds the block into a
 *        per-row running top-512 (Sort32 + MrgSort, descending), and when a
 *        token tile's whole S2 range is covered, expands pool ids to token
 *        ids (x kpool), pads with -1 and appends the causal tail.
 *
 * M_TILE is 32 so one AIV owns 16 rows. Group mode stores only the 512 live
 * value/index pairs per row; fused mode retains its 1024-pair merge headroom.
 */

#ifndef GLM5_KPOOL_INDEXER_SERVICE_VECTOR_H
#define GLM5_KPOOL_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "glm5_kpool_indexer_common.h"
#include "glm5_kpool_indexer_vector.h"

namespace Glm5KpoolKernel {
using namespace Glm5KpoolCommon;
using namespace Glm5KpoolVec;

template <typename Q_T, uint32_t M_TILE_SIZE>
class Glm5KpoolServiceVector {
public:
    __aicore__ inline Glm5KpoolServiceVector(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitInputTensor(const GlobalTensor<int32_t> &indicesOutGm,
                                           const GlobalTensor<float> &scoresDebugGm,
                                           const GlobalTensor<int32_t> &positionsGm,
                                           const GlobalTensor<int32_t> &cumQueryLensGm,
                                           const GlobalTensor<int32_t> &indexerSeqLensGm,
                                           const GlobalTensor<float> &mm1ResGm);
    __aicore__ inline void ProcessVec(const RunInfo &runInfo);
    __aicore__ inline void ProcessPoolGroup(const RunInfo &runInfo);
    __aicore__ inline void LoadRowMeta(const RunInfo &runInfo, bool initStrip = true);
    __aicore__ inline void ProcessEmptyUnit(const RunInfo &runInfo);
    __aicore__ inline void SetSplitStripGm(const GlobalTensor<float> &stripGm);
    __aicore__ inline void RelocateMm1Res(__gm__ float *base);
    // Split mode spans multiple kernel launches (one per pool batch): the
    // per-row running top-512 lives in UB inside one launch and must be
    // staged through GM between batches. Kernel boundaries are full barriers,
    // so plain DataCopy is safe across launches.
    __aicore__ inline void SaveRunningStrip(const RunInfo &runInfo);
    __aicore__ inline void RestoreRunningStrip(const RunInfo &runInfo);
    __aicore__ inline void ProcessTopK(const RunInfo &runInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t ROWS_PER_AIV = M_TILE_SIZE / 2;

protected:
    __aicore__ inline uint32_t RowVisiblePools(int32_t pos, int32_t reqPoolLen) const;
    __aicore__ inline void FlattenStripRuns(const LocalTensor<float> &strip, uint32_t runCount);
    __aicore__ inline void FoldBlockIntoRowTopk(uint32_t rowLocal, const RunInfo &runInfo);
    __aicore__ inline void EmitRow(uint32_t rowLocal, const RunInfo &runInfo);

    ConstInfo constInfo_{};
    GlobalTensor<float> mm1ResGm;
    GlobalTensor<int32_t> indicesOutGm; // [T, 1, outputWidth]
    GlobalTensor<float> scoresDebugGm;  // [T, maxPoolSeqLen] or empty
    GlobalTensor<int32_t> positionsGm; // [T]
    GlobalTensor<int32_t> cumQueryLensGm;
    GlobalTensor<int32_t> indexerSeqLensGm;

    // running top-k: [ROWS_PER_AIV][poolTopk * 2] fp32 (value,index pairs)
    TBuf<TPosition::VECCALC> globalTopkBuf_;
    LocalTensor<float> globalTopkUb_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    LocalTensor<float> scoreUb_;       // [S2_TILE]
    LocalTensor<uint32_t> scoreIdxUb_; // [S2_TILE]
    TBuf<TPosition::VECCALC> sortDstBuf_;
    LocalTensor<float> sortDstUb_; // [S2_TILE * 2]
    TBuf<TPosition::VECCALC> mrgTmpBuf_;
    LocalTensor<float> mrgTmpUb_;   // SortFull1024 scratch (3072 words)
    TBuf<TPosition::VECCALC> idxTmpBuf_;
    LocalTensor<uint32_t> idxTmpUb_; // SortFull1024 index split (1024 words)
    // Output rides the TPipe VECOUT queue: EnQue/DeQue carry the hardware
    // V->MTE3 pipe sync (lightning_indexer pattern). Hand-rolled
    // SetFlag/WaitFlag pairs on a bare buffer race under multi-core load
    // (packed multi-request hangs bisected to exactly that).
    TQue<QuePosition::VECOUT, 1> outQue_;

    int32_t posCache_[ROWS_PER_AIV] = {0};
    uint32_t visibleCache_[ROWS_PER_AIV] = {0};
    // live pair count / sorted-run count of each row's running strip
    // (lazy merge-fold state: runs are appended untouched while live <
    // poolTopk, flattened on first overflow or at emit)
    uint32_t liveCache_[ROWS_PER_AIV] = {0};
    uint32_t runCountCache_[ROWS_PER_AIV] = {0};
    uint32_t blockId_ = 0;
    uint32_t aivHalf_ = 0;
    uint32_t poolTopk_ = 0;
    // Fused mode needs 2*poolTopk pairs for the incoming block and padding.
    // Group mode merges two already-truncated poolTopk prefixes, so it needs
    // no extra per-row headroom.
    uint32_t topkStride_ = 0;
    GlobalTensor<float> splitStripGm_; // split mode: cross-batch running strip
};

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    blockId_ = GetBlockIdx();
    aivHalf_ = blockId_ % 2;
    poolTopk_ = constInfo.poolTopk;
    topkStride_ = (IsGroupTopkMode(constInfo.outputMode) || IsSplitMode(constInfo.outputMode))
                       ? poolTopk_
                       : 2 * poolTopk_;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(globalTopkBuf_, ROWS_PER_AIV * topkStride_ * 2 * sizeof(float));
    globalTopkUb_ = globalTopkBuf_.Get<float>();
    const uint32_t scoreWidth =
        (IsGroupTopkMode(constInfo_.outputMode) || IsSplitMode(constInfo_.outputMode)) ? POOL_GROUP : S2_TILE;
    pipe->InitBuffer(scoreBuf_, 2 * scoreWidth * sizeof(float));
    scoreUb_ = scoreBuf_.Get<float>();
    scoreIdxUb_ = scoreBuf_.Get<uint32_t>()[scoreWidth];
    pipe->InitBuffer(sortDstBuf_, scoreWidth * 2 * sizeof(float));
    sortDstUb_ = sortDstBuf_.Get<float>();
    pipe->InitBuffer(mrgTmpBuf_, 3072 * sizeof(float));
    mrgTmpUb_ = mrgTmpBuf_.Get<float>();
    pipe->InitBuffer(idxTmpBuf_, 1024 * sizeof(float));
    idxTmpUb_ = idxTmpBuf_.Get<uint32_t>();
    pipe->InitBuffer(outQue_, 1, (constInfo_.poolTopk + 64) * sizeof(int32_t));
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::InitInputTensor(
    const GlobalTensor<int32_t> &indicesOutGm, const GlobalTensor<float> &scoresDebugGm,
    const GlobalTensor<int32_t> &positionsGm, const GlobalTensor<int32_t> &cumQueryLensGm,
    const GlobalTensor<int32_t> &indexerSeqLensGm, const GlobalTensor<float> &mm1ResGm)
{
    this->indicesOutGm = indicesOutGm;
    this->scoresDebugGm = scoresDebugGm;
    this->positionsGm = positionsGm;
    this->cumQueryLensGm = cumQueryLensGm;
    this->indexerSeqLensGm = indexerSeqLensGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::AllocEventID()
{
    // Only the scoreUb_ ping-pong needs a pre-armed flag (first Fold waits on
    // it). The MTE3 output direction is owned by outQue_'s EnQue/DeQue.
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline uint32_t Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::RowVisiblePools(int32_t pos,
                                                                                     int32_t reqPoolLen) const
{
    uint32_t causalPools = static_cast<uint32_t>((static_cast<int64_t>(pos) + 1) >> 2); // kpool == 4 (asserted in tiling)
    uint32_t vis = Min(causalPools, static_cast<uint32_t>(Max(reqPoolLen, 0)));
    vis = Min(vis, constInfo_.maxPoolSeqLen);
    return vis;
}

// On the first S2 tile of a token tile: cache per-row positions/visibility
// and reset the running top-k strips. No GM scalar reads: the AIV scalar
// load compiles to a vector-granularity access that faults at non-32B-aligned
// offsets (positions were the original offender; the request pool length
// rides in RunInfo and pos is derived arithmetically — a unit never spans
// requests, so pos == posBase + rowInTile).
template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::LoadRowMeta(const RunInfo &runInfo, bool initStrip)
{
    int32_t reqPoolLen = static_cast<int32_t>(runInfo.reqPoolLen);
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + r;
        posCache_[r] = (rowInTile < runInfo.actMSize)
                           ? static_cast<int32_t>(runInfo.posBase + rowInTile)
                           : -1; // padded row marker
        visibleCache_[r] = RowVisiblePools(posCache_[r], reqPoolLen);
    }
    // A later split batch restores the strip from GM. Re-initializing the same
    // UB immediately before MTE2 writes it creates a cross-pipe write race.
    if (initStrip) {
        InitSortOutBuf(globalTopkUb_, ROWS_PER_AIV * topkStride_ * 2);
    }
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        liveCache_[r] = 0;
        runCountCache_[r] = 0;
    }
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::SetSplitStripGm(const GlobalTensor<float> &stripGm)
{
    splitStripGm_ = stripGm;
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::RelocateMm1Res(__gm__ float *base)
{
    mm1ResGm.SetGlobalBuffer(base);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::SaveRunningStrip(const RunInfo &runInfo)
{
    const uint32_t words = poolTopk_ * VALUE_AND_INDEX_NUM;
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue;
        }
        const uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + r;
        const uint32_t rowGlobal = runInfo.mStart + rowInTile;
        DataCopy(splitStripGm_[static_cast<uint64_t>(rowGlobal) * words],
                 globalTopkUb_[r * topkStride_ * 2], words);
    }
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::RestoreRunningStrip(const RunInfo &runInfo)
{
    const uint32_t words = poolTopk_ * VALUE_AND_INDEX_NUM;
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue;
        }
        const uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + r;
        const uint32_t rowGlobal = runInfo.mStart + rowInTile;
        DataCopy(globalTopkUb_[r * topkStride_ * 2],
                 splitStripGm_[static_cast<uint64_t>(rowGlobal) * words], words);
    }
    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::ProcessEmptyUnit(const RunInfo &runInfo)
{
    LoadRowMeta(runInfo);
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue;
        }
        const uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + r;
        const uint32_t rowGlobal = runInfo.mStart + rowInTile;
        // InitGlobalMemory needs a GlobalTensor lvalue and brings its own
        // V->MTE3 event pair; reuse the proven EmitRow output lane instead
        // (VEC duplicate -> MTE3 copy out), keeping the event ledger of a
        // regular emit.
        LocalTensor<int32_t> outUb = outQue_.AllocTensor<int32_t>();
        AscendC::Duplicate<int32_t>(outUb, -1, constInfo_.poolTopk);
        PipeBarrier<PIPE_V>();
        outQue_.EnQue<int32_t>(outUb);
        outUb = outQue_.DeQue<int32_t>();
        Glm5KpoolVec::CopyOut(indicesOutGm[static_cast<uint64_t>(rowGlobal) * constInfo_.poolTopk],
                              outUb, constInfo_.poolTopk);
        outQue_.FreeTensor(outUb);
    }
}

// k sorted 128-runs sit contiguously at strip[0]; merge them into one
// sorted run (k is 2..4 — a single run needs no work). Lengths stay
// 32-multiples per the M6 granularity lesson.
template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::FlattenStripRuns(const LocalTensor<float> &strip,
                                                                                  uint32_t runCount)
{
    AscendC::MrgSort4Info p;
    p.elementLengths[MRG_QUE_0] = (runCount > 0) ? S2_TILE : 0;
    p.elementLengths[MRG_QUE_1] = (runCount > 1) ? S2_TILE : 0;
    p.elementLengths[MRG_QUE_2] = (runCount > 2) ? S2_TILE : 0;
    p.elementLengths[MRG_QUE_3] = (runCount > 3) ? S2_TILE : 0;
    p.ifExhaustedSuspension = false;
    p.validBit = static_cast<uint8_t>((1u << runCount) - 1u);
    p.repeatTimes = 1;
    AscendC::MrgSortSrcList<float> s;
    s.src1 = strip[0];
    s.src2 = strip[S2_TILE * VALUE_AND_INDEX_NUM];
    s.src3 = strip[2 * S2_TILE * VALUE_AND_INDEX_NUM];
    s.src4 = strip[3 * S2_TILE * VALUE_AND_INDEX_NUM];
    AscendC::MrgSort<float>(mrgTmpUb_, s, p);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(strip, mrgTmpUb_, runCount * S2_TILE * VALUE_AND_INDEX_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::FoldBlockIntoRowTopk(uint32_t rowLocal,
                                                                                      const RunInfo &runInfo)
{
    uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + rowLocal;
    // mm1Res strips: quad-buffered [4][M_TILE][S2_TILE]; the AIC's single
    // k=256 Mmad already emits the combined [q_hi|q_lo] fp32 score (H9), so
    // the fold is a plain single-buffer read again. Selector matches the
    // AIC's Fixp (loop % 4).
    uint64_t bufBase = (runInfo.loop % 4) * M_TILE_SIZE * S2_TILE;
    uint64_t rowBase = bufBase + rowInTile * S2_TILE;

    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams inParams;
    inParams.blockCount = 1;
    inParams.blockLen = S2_TILE * sizeof(float);
    inParams.srcStride = 0;
    inParams.dstStride = 0;
    inParams.rsv = 0;
    AscendC::DataCopyPad(scoreUb_, mm1ResGm[rowBase], inParams, padParams);
    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT);
    uint32_t visible = visibleCache_[rowLocal];
    uint32_t validLanes = (runInfo.s2Start < visible) ? Min(S2_TILE, visible - runInfo.s2Start) : 0;
    PipeBarrier<PIPE_V>();

    // Pools beyond the row's causal/visible bound must never win: force them
    // to -inf via an exact bitmap mask (plain Duplicate needs 32B-aligned count).
    if (validLanes < S2_TILE) {
        PipeBarrier<PIPE_V>();
        for (uint32_t repeat = 0; repeat < 2; repeat++) {
            uint32_t repeatBase = repeat * B32_VEC_ELM_NUM;
            uint32_t validInRepeat =
                (validLanes > repeatBase) ? Min(B32_VEC_ELM_NUM, validLanes - repeatBase) : 0;
            if (validInRepeat < B32_VEC_ELM_NUM) {
                uint64_t invalidMask[2] = {
                    (validInRepeat == 0) ? ~0ULL : (~0ULL << validInRepeat), 0};
                AscendC::Duplicate(scoreUb_.ReinterpretCast<int32_t>()[repeatBase], Glm5KpoolVec::NEG_INF,
                                   invalidMask, 1, 1, B32_VEC_REPEAT_STRIDE);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    // debug mode: persist raw fp32 scores through the output queue. MUST run
    // before SortAll: Sort consumes scoreUb_ as its source and leaves it
    // reordered, so a post-sort copy ships a descending run instead of the
    // pool-ordered scores (M5c diagnosis was briefly misled by exactly that).
    if (constInfo_.outputMode == 1 && validLanes > 0) {
        uint32_t rowGlobal = runInfo.mStart + rowInTile;
        LocalTensor<float> dbgUb = outQue_.AllocTensor<float>();
        AscendC::DataCopy(dbgUb, scoreUb_, CeilDiv(validLanes, 8) * 8);
        outQue_.EnQue<float>(dbgUb);
        dbgUb = outQue_.DeQue<float>();
        Glm5KpoolVec::CopyOut(scoresDebugGm[static_cast<uint64_t>(rowGlobal) * constInfo_.maxPoolSeqLen +
                                            runInfo.s2Start],
                              dbgUb, validLanes);
        outQue_.FreeTensor(dbgUb);
    }

    // indices: absolute pool ids
    PipeBarrier<PIPE_V>();
    ArithProgression<int32_t>(scoreIdxUb_.ReinterpretCast<int32_t>(),
                              static_cast<int32_t>(runInfo.s2Start), 1, S2_TILE);
    PipeBarrier<PIPE_V>();

    // descending full sort of the 128-lane block
    SortAll(sortDstUb_, scoreUb_, scoreIdxUb_, mrgTmpUb_, S2_TILE);

    // Lazy fold: while the strip is below capacity the sorted 128-run is
    // merely APPENDED (strip = concatenation of up to 4 sorted runs, zero
    // merges); only the first overflow fold flattens the runs (one
    // MrgSort4) and from then on every fold is a single 2-way merge of two
    // sorted runs (O(640)) — still 32-multiple lengths throughout (the M6
    // eviction-granularity lesson). The evicted tail is re-padded to the
    // -inf/-1 sentinel so no stale real pair re-enters a merge or emit
    // window; EmitRow flattens any still-lazy runs before reading top-512.
    LocalTensor<float> strip = globalTopkUb_[rowLocal * topkStride_ * 2];
    uint32_t live = liveCache_[rowLocal];
    if (live + S2_TILE <= poolTopk_) {
        AscendC::DataCopy(strip[live * 2], sortDstUb_, S2_TILE * 2);
        PipeBarrier<PIPE_V>();
        if (live == 0) {
            InitSortOutBuf(strip[S2_TILE * 2],
                           static_cast<int64_t>((topkStride_ - S2_TILE) * 2));
        }
        liveCache_[rowLocal] = live + S2_TILE;
        runCountCache_[rowLocal]++;
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
        return;
    }
    if (runCountCache_[rowLocal] > 1) {
        FlattenStripRuns(strip, runCountCache_[rowLocal]);
        runCountCache_[rowLocal] = 1;
    }
    {
        AscendC::MrgSort4Info p;
        p.elementLengths[MRG_QUE_0] = poolTopk_; // liveAlign == poolTopk_ here
        p.elementLengths[MRG_QUE_1] = S2_TILE;
        p.elementLengths[MRG_QUE_2] = 0;
        p.elementLengths[MRG_QUE_3] = 0;
        p.ifExhaustedSuspension = false;
        p.validBit = 0b0011;
        p.repeatTimes = 1;
        AscendC::MrgSortSrcList<float> s;
        s.src1 = strip[0];
        s.src2 = sortDstUb_[0];
        AscendC::MrgSort<float>(mrgTmpUb_, s, p);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(strip, mrgTmpUb_, (poolTopk_ + S2_TILE) * 2);
        AscendC::PipeBarrier<PIPE_V>();
        InitSortOutBuf(strip[poolTopk_ * 2], static_cast<int64_t>(S2_TILE * 2));
    }
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::ProcessVec(const RunInfo &runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        LoadRowMeta(runInfo);
    }
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue; // padded row: nothing to fold or emit
        }
        FoldBlockIntoRowTopk(r, runInfo);
    }
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::ProcessPoolGroup(const RunInfo &runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        LoadRowMeta(runInfo);
    }

    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue;
        }
        const uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + r;
        // Mix mode: the strip is [M_TILE, POOL_GROUP] and the AIC rewrites it
        // per group (1-lag). Split AIV: the strip is [M_TILE, SPLIT_BATCH_POOLS]
        // and each group is a column window inside the batch.
        const uint64_t rowBase = IsSplitMode(constInfo_.outputMode)
            ? static_cast<uint64_t>(rowInTile) * SPLIT_BATCH_POOLS +
              (runInfo.s2Start % SPLIT_BATCH_POOLS)
            : static_cast<uint64_t>(rowInTile) * POOL_GROUP;

        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.blockLen = POOL_GROUP * sizeof(float);
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        AscendC::DataCopyPad(scoreUb_, mm1ResGm[rowBase], inParams, padParams);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT);

        const uint32_t visible = visibleCache_[r];
        const uint32_t validLanes = (runInfo.s2Start < visible) ? Min(POOL_GROUP, visible - runInfo.s2Start) : 0;
        if (validLanes < POOL_GROUP) {
            for (uint32_t repeat = 0; repeat < POOL_GROUP / B32_VEC_ELM_NUM; repeat++) {
                const uint32_t repeatBase = repeat * B32_VEC_ELM_NUM;
                const uint32_t validInRepeat =
                    (validLanes > repeatBase) ? Min(B32_VEC_ELM_NUM, validLanes - repeatBase) : 0;
                if (validInRepeat < B32_VEC_ELM_NUM) {
                    uint64_t invalidMask[2] = {
                        (validInRepeat == 0) ? ~0ULL : (~0ULL << validInRepeat), 0};
                    AscendC::Duplicate(scoreUb_.ReinterpretCast<int32_t>()[repeatBase], Glm5KpoolVec::NEG_INF,
                                       invalidMask, 1, 1, B32_VEC_REPEAT_STRIDE);
                }
            }
            PipeBarrier<PIPE_V>();
        }

        ArithProgression<int32_t>(scoreIdxUb_.ReinterpretCast<int32_t>(),
                                  static_cast<int32_t>(runInfo.s2Start), 1, POOL_GROUP);
        PipeBarrier<PIPE_V>();
        SortFull1024(sortDstUb_, mrgTmpUb_, scoreUb_, scoreIdxUb_);

        LocalTensor<float> strip = globalTopkUb_[r * topkStride_ * 2];
        if (runInfo.isFirstS2InnerLoop) {
            Glm5KpoolVec::CopyWordsVec(strip, sortDstUb_, poolTopk_ * VALUE_AND_INDEX_NUM);
        } else {
            AscendC::MrgSort4Info p;
            p.elementLengths[MRG_QUE_0] = poolTopk_;
            p.elementLengths[MRG_QUE_1] = poolTopk_;
            p.elementLengths[MRG_QUE_2] = 0;
            p.elementLengths[MRG_QUE_3] = 0;
            p.ifExhaustedSuspension = false;
            p.validBit = 0b0011;
            p.repeatTimes = 1;
            AscendC::MrgSortSrcList<float> src;
            src.src1 = strip;
            src.src2 = sortDstUb_;
            AscendC::MrgSort<float>(mrgTmpUb_, src, p);
            PipeBarrier<PIPE_V>();
            Glm5KpoolVec::CopyWordsVec(strip, mrgTmpUb_, poolTopk_ * VALUE_AND_INDEX_NUM);
        }
        liveCache_[r] = poolTopk_;
        runCountCache_[r] = 1;
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
    }
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::EmitRow(uint32_t rowLocal,
                                                                         const RunInfo &runInfo)
{
    // Vector-only pool-id emission: scalar GetValue/SetValue on UB compiles
    // to vector-granular accesses and faults at unaligned offsets, so the
    // x4 expansion and causal tail move to the python wrapper. Output row =
    // poolTopk int32 pool ids (sentinel lanes stay -1).
    uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + rowLocal;
    uint32_t rowGlobal = runInfo.mStart + rowInTile;
    uint32_t visible = visibleCache_[rowLocal];
    // lazy append leaves the strip as several sorted runs; the top-512 read
    // below needs one sorted prefix.
    if (runCountCache_[rowLocal] > 1) {
        FlattenStripRuns(globalTopkUb_[rowLocal * topkStride_ * 2], runCountCache_[rowLocal]);
        runCountCache_[rowLocal] = 1;
    }

    // Extract the interleaved index half of the running top-k strip and ship
    // it raw through the VECOUT queue: out-of-range / sentinel lanes are
    // filtered in the python wrapper (it owns `positions`, hence per-row
    // visibility).
    LocalTensor<int32_t> outUb = outQue_.AllocTensor<int32_t>();
    Glm5KpoolVec::ExtractIndex(outUb.ReinterpretCast<uint32_t>(),
                               globalTopkUb_[rowLocal * topkStride_ * 2].ReinterpretCast<uint32_t>(),
                               static_cast<int64_t>(poolTopk_));
    PipeBarrier<PIPE_V>();
    outQue_.EnQue<int32_t>(outUb);
    outUb = outQue_.DeQue<int32_t>();
    Glm5KpoolVec::CopyOut(indicesOutGm[static_cast<uint64_t>(rowGlobal) * constInfo_.poolTopk],
                          outUb, constInfo_.poolTopk);
    outQue_.FreeTensor(outUb);
}

template <typename Q_T, uint32_t M_TILE_SIZE>
__aicore__ inline void Glm5KpoolServiceVector<Q_T, M_TILE_SIZE>::ProcessTopK(const RunInfo &runInfo)
{
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue;
        }
        EmitRow(r, runInfo);
    }
}

} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_SERVICE_VECTOR_H
