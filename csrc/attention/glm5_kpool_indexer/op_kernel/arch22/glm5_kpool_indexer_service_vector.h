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
 * M_TILE is 32 so one AIV owns 16 rows; the running top-k strip for all rows
 * (16 * 512 * 2 * 4B = 64KB) fits in UB alongside sort scratch.
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

template <typename Q_T>
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
    __aicore__ inline void ProcessTopK(const RunInfo &runInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t ROWS_PER_AIV = M_TILE / 2; // 16

protected:
    __aicore__ inline uint32_t RowVisiblePools(int32_t pos, int32_t reqPoolLen) const;
    __aicore__ inline void LoadRowMeta(const RunInfo &runInfo);
    __aicore__ inline void FoldBlockIntoRowTopk(uint32_t rowLocal, const RunInfo &runInfo);
    __aicore__ inline void EmitRow(uint32_t rowLocal, const RunInfo &runInfo);

    ConstInfo constInfo_{};
    GlobalTensor<float> mm1ResGm;       // per-AIC [2][M_TILE][S2_TILE]
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
    uint32_t blockId_ = 0;
    uint32_t aivHalf_ = 0;
    uint32_t poolTopk_ = 0;
    // Running-strip entry stride in (value,index) pairs: poolTopk live pairs,
    // then the incoming S2_TILE block, then -inf/-1 pad up to 2*poolTopk.
    // The fold appends + pads + runs SortFull1024 (Sort32 + fixed merge
    // tree); the single-shot MrgSort strip recursion it replaces silently
    // dropped the last pairs of every 32-pair run whenever the strip had to
    // evict live values (M6 eviction bug).
    uint32_t topkStride_ = 0;
};

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    blockId_ = GetBlockIdx();
    aivHalf_ = blockId_ % 2;
    poolTopk_ = constInfo.poolTopk;
    topkStride_ = 2 * poolTopk_; // 1024 pairs: poolTopk live + new block + pad
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(globalTopkBuf_, ROWS_PER_AIV * topkStride_ * 2 * sizeof(float));
    globalTopkUb_ = globalTopkBuf_.Get<float>();
    pipe->InitBuffer(scoreBuf_, 2 * S2_TILE * sizeof(float));
    scoreUb_ = scoreBuf_.Get<float>();
    scoreIdxUb_ = scoreBuf_.Get<uint32_t>()[S2_TILE];
    pipe->InitBuffer(sortDstBuf_, S2_TILE * 2 * sizeof(float));
    sortDstUb_ = sortDstBuf_.Get<float>();
    pipe->InitBuffer(mrgTmpBuf_, 3072 * sizeof(float));
    mrgTmpUb_ = mrgTmpBuf_.Get<float>();
    pipe->InitBuffer(idxTmpBuf_, 1024 * sizeof(float));
    idxTmpUb_ = idxTmpBuf_.Get<uint32_t>();
    pipe->InitBuffer(outQue_, 1, (constInfo_.poolTopk + 64) * sizeof(int32_t));
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitInputTensor(
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

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::AllocEventID()
{
    // Only the scoreUb_ ping-pong needs a pre-armed flag (first Fold waits on
    // it). The MTE3 output direction is owned by outQue_'s EnQue/DeQue.
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolServiceVector<Q_T>::RowVisiblePools(int32_t pos, int32_t reqPoolLen) const
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
template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::LoadRowMeta(const RunInfo &runInfo)
{
    int32_t reqPoolLen = static_cast<int32_t>(runInfo.reqPoolLen);
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + r;
        posCache_[r] = (rowInTile < runInfo.actMSize)
                           ? static_cast<int32_t>(runInfo.posBase + rowInTile)
                           : -1; // padded row marker
        visibleCache_[r] = RowVisiblePools(posCache_[r], reqPoolLen);
    }
    // value = -inf, index = -1 interleaved, for all owned rows (live + tail)
    InitSortOutBuf(globalTopkUb_, ROWS_PER_AIV * topkStride_ * 2);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::FoldBlockIntoRowTopk(uint32_t rowLocal, const RunInfo &runInfo)
{
    uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + rowLocal;
    uint64_t bufBase = (runInfo.loop % 2) * M_TILE * S2_TILE; // fixed S2_TILE stride
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

    // Pools beyond the row's causal/visible bound must never win: force them
    // to -inf via an exact bitmap mask (plain Duplicate needs 32B-aligned count).
    uint32_t visible = visibleCache_[rowLocal];
    uint32_t validLanes = (runInfo.s2Start < visible) ? Min(S2_TILE, visible - runInfo.s2Start) : 0;
    if (validLanes < S2_TILE) {
        uint64_t laneMask[2] = {0, 0};
        for (uint32_t lane = validLanes; lane < S2_TILE; lane++) {
            laneMask[lane / 64] |= (1ULL << (lane % 64));
        }
        PipeBarrier<PIPE_V>();
        AscendC::Duplicate(scoreUb_.ReinterpretCast<int32_t>(), Glm5KpoolVec::NEG_INF, laneMask, 2, 1,
                           B32_VEC_REPEAT_STRIDE);
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

    // fold into the running top-k: append the sorted block after the live
    // pairs, arm the -inf pad, and full-resort the whole 1024-pair strip.
    // (Replaces the MrgSort strip recursion, which silently dropped the
    // trailing pairs of every 32-pair run once the strip was full.)
    LocalTensor<float> strip = globalTopkUb_[rowLocal * topkStride_ * 2];
    AscendC::DataCopy(strip[poolTopk_ * 2], sortDstUb_, S2_TILE * 2);
    PipeBarrier<PIPE_V>();
    InitSortOutBuf(strip[(poolTopk_ + S2_TILE) * 2],
                   static_cast<int64_t>((topkStride_ - poolTopk_ - S2_TILE) * 2));
    SortFull1024(strip, mrgTmpUb_, idxTmpUb_);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::ProcessVec(const RunInfo &runInfo)
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

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::EmitRow(uint32_t rowLocal, const RunInfo &runInfo)
{
    // Vector-only pool-id emission: scalar GetValue/SetValue on UB compiles
    // to vector-granular accesses and faults at unaligned offsets, so the
    // x4 expansion and causal tail move to the python wrapper. Output row =
    // poolTopk int32 pool ids (sentinel lanes stay -1).
    uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + rowLocal;
    uint32_t rowGlobal = runInfo.mStart + rowInTile;
    uint32_t visible = visibleCache_[rowLocal];

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

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::ProcessTopK(const RunInfo &runInfo)
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
