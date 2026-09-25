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

// Trustworthy content gates (top of file, before ANY use):
#define GLMK_LRM 1 
#define GLMK_LRM_GM 1 // scalar GM reads inside LoadRowMeta
#define GLMK_LRM_SORT 1 // InitSortOutBuf inside LoadRowMeta  // LoadRowMeta (GM scalar reads + InitSortOutBuf)
#define GLMK_COPY 1  // per-row DataCopyPad GM->UB
#define GLMK_VOPS 1  // mask Duplicate + ArithProgression + SortAll + MergeSort
#define GLMK_EMIT 1  // EmitRow (Duplicate + scalar SetValue + CopyOut)

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
                                           const GlobalTensor<int64_t> &positionsGm,
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
    __aicore__ inline uint32_t RowVisiblePools(int64_t pos, int32_t reqPoolLen) const;
    __aicore__ inline void LoadRowMeta(const RunInfo &runInfo); // cache pos/visible per row
    __aicore__ inline void FoldBlockIntoRowTopk(uint32_t rowLocal, const RunInfo &runInfo);
    __aicore__ inline void EmitRow(uint32_t rowLocal, const RunInfo &runInfo);

    ConstInfo constInfo_{};
    GlobalTensor<float> mm1ResGm;       // per-AIC [2][M_TILE][S2_TILE]
    GlobalTensor<int32_t> indicesOutGm; // [T, 1, outputWidth]
    GlobalTensor<float> scoresDebugGm;  // [T, maxPoolSeqLen] or empty
    GlobalTensor<int64_t> positionsGm;  // [T padded to M_TILE]
    GlobalTensor<int32_t> cumQueryLensGm;
    GlobalTensor<int32_t> indexerSeqLensGm;

    // running top-k: [ROWS_PER_AIV][poolTopk * 2] fp32 (value,index pairs)
    TBuf<TPosition::VECCALC> globalTopkBuf_;
    LocalTensor<float> globalTopkUb_;
    // block score + index, sort dst, merge tmp
    TBuf<TPosition::VECCALC> scoreBuf_;
    LocalTensor<float> scoreUb_;       // [S2_TILE]
    LocalTensor<uint32_t> scoreIdxUb_; // [S2_TILE]
    TBuf<TPosition::VECCALC> sortDstBuf_;
    LocalTensor<float> sortDstUb_; // [S2_TILE * 2]
    TBuf<TPosition::VECCALC> mrgTmpBuf_;
    LocalTensor<float> mrgTmpUb_; // [(poolTopk + S2_TILE) * 2]
    // row output staging
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<int32_t> outUb_; // [outputWidth + 64]

    int64_t posCache_[ROWS_PER_AIV] = {0};
    uint32_t visibleCache_[ROWS_PER_AIV] = {0};
    uint32_t blockId_ = 0;
    uint32_t aivHalf_ = 0;
    uint32_t poolTopk_ = 0;
};

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    blockId_ = GetBlockIdx();
    aivHalf_ = blockId_ % 2;
    poolTopk_ = constInfo.poolTopk;
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(globalTopkBuf_, ROWS_PER_AIV * poolTopk_ * 2 * sizeof(float));
    globalTopkUb_ = globalTopkBuf_.Get<float>();
    pipe->InitBuffer(scoreBuf_, 2 * S2_TILE * sizeof(float));
    scoreUb_ = scoreBuf_.Get<float>();
    scoreIdxUb_ = scoreBuf_.Get<uint32_t>()[S2_TILE];
    pipe->InitBuffer(sortDstBuf_, S2_TILE * 2 * sizeof(float));
    sortDstUb_ = sortDstBuf_.Get<float>();
    pipe->InitBuffer(mrgTmpBuf_, (poolTopk_ + S2_TILE) * 2 * sizeof(float));
    mrgTmpUb_ = mrgTmpBuf_.Get<float>();
    pipe->InitBuffer(outBuf_, (constInfo_.outputWidth + 64) * sizeof(int32_t));
    outUb_ = outBuf_.Get<int32_t>();
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitInputTensor(
    const GlobalTensor<int32_t> &indicesOutGm, const GlobalTensor<float> &scoresDebugGm,
    const GlobalTensor<int64_t> &positionsGm, const GlobalTensor<int32_t> &cumQueryLensGm,
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
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolServiceVector<Q_T>::RowVisiblePools(int64_t pos, int32_t reqPoolLen) const
{
    uint32_t causalPools = static_cast<uint32_t>((pos + 1) >> 2); // kpool == 4 (asserted in tiling)
    uint32_t vis = Min(causalPools, static_cast<uint32_t>(Max(reqPoolLen, 0)));
    vis = Min(vis, constInfo_.maxPoolSeqLen);
    return vis;
}

// On the first S2 tile of a token tile: cache per-row positions/visibility
// and reset the running top-k strips.
template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::LoadRowMeta(const RunInfo &runInfo)
{
#if GLMK_LRM_GM
    int32_t reqPoolLen = indexerSeqLensGm.GetValue(runInfo.reqIdx);
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        uint32_t rowGlobal = runInfo.mStart + aivHalf_ * ROWS_PER_AIV + r;
        if (rowGlobal < constInfo_.tSize) {
            posCache_[r] = positionsGm.GetValue(rowGlobal);
        } else {
            posCache_[r] = -1; // padded row marker
        }
        visibleCache_[r] = RowVisiblePools(posCache_[r], reqPoolLen);
    }
#else
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        posCache_[r] = r; // fake in-range positions
        visibleCache_[r] = constInfo_.maxPoolSeqLen;
    }
#endif
#if GLMK_LRM_SORT
    // value = -inf, index = -1 interleaved, for all owned rows
    InitSortOutBuf(globalTopkUb_, ROWS_PER_AIV * poolTopk_ * 2);
#endif
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::FoldBlockIntoRowTopk(uint32_t rowLocal, const RunInfo &runInfo)
{
    // Pull the row's S2_TILE scores from the cube's mm1Res strip.
    uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + rowLocal;
    uint64_t bufBase = (runInfo.loop % 2) * M_TILE * runInfo.actS2SizeAlign;
    uint64_t rowBase = bufBase + rowInTile * runInfo.actS2SizeAlign;

#else
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
#endif
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
#endif


    // Lane masks: pools beyond the row's causal/visible bound (and beyond the
    // request bound) must never win: force them to -inf. The plain Duplicate
    // requires a 32B-aligned count, so mask off exact lanes bitmap-style
    // (validLanes is rarely a multiple of 8).
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
#endif

    // indices: absolute pool ids; Sort needs them as float-reinterpret uint32.
    PipeBarrier<PIPE_V>();
    ArithProgression<int32_t>(scoreIdxUb_.ReinterpretCast<int32_t>(),
                              static_cast<int32_t>(runInfo.s2Start), 1, S2_TILE);
    PipeBarrier<PIPE_V>();

    // descending full sort of the 128-lane block
    SortAll(sortDstUb_, scoreUb_, scoreIdxUb_, mrgTmpUb_, S2_TILE);
#endif

    // debug mode: also persist the raw fp32 scores
    if (constInfo_.outputMode == 1 && validLanes > 0) {
        uint32_t rowGlobal = runInfo.mStart + rowInTile;
        WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
        Glm5KpoolVec::CopyOut(scoresDebugGm[static_cast<uint64_t>(rowGlobal) * constInfo_.maxPoolSeqLen +
                                            runInfo.s2Start],
                              scoreUb_, validLanes);
        SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    }

    // fold into the running top-k: MergeSort keeps the best poolTopk pairs.
    MergeSort(globalTopkUb_[rowLocal * poolTopk_ * 2], static_cast<int32_t>(poolTopk_), sortDstUb_,
              static_cast<int32_t>(S2_TILE), mrgTmpUb_);
#endif
#if GLMK_FOLD_STAGE < 9
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT);
#endif
}

#define GLMK_DEBUG_STAGE 3 // 1=handshake only, 2=+fold, 3=full
template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::ProcessVec(const RunInfo &runInfo)
{
    // V-pipe liveness probe: one plain (maskless) Duplicate, nothing else.
    {
        LocalTensor<float> probe = globalTopkUb_;
        AscendC::Duplicate(probe, 0.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
    }

#if GLMK_DEBUG_STAGE < 2
    (void)runInfo;
    return;
#endif
    if (runInfo.isFirstS2InnerLoop) {
        LoadRowMeta(runInfo);
#endif
    }
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue; // padded row: nothing to fold or emit
        }
#if GLMK_COPY || GLMK_VOPS
        FoldBlockIntoRowTopk(r, runInfo);
#endif
    }
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::EmitRow(uint32_t rowLocal, const RunInfo &runInfo)
{
    uint32_t rowInTile = aivHalf_ * ROWS_PER_AIV + rowLocal;
    uint32_t rowGlobal = runInfo.mStart + rowInTile;
    int64_t pos = posCache_[rowLocal];
    uint32_t visible = visibleCache_[rowLocal];

    // Stage the whole output row as -1, then poke the expanded history.
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate(outUb_, static_cast<int32_t>(INVALID_IDX), constInfo_.outputWidth);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);

    LocalTensor<float> rowTopk = globalTopkUb_[rowLocal * poolTopk_ * 2];
    uint32_t topkSel = Min(poolTopk_, visible);
    // value/index pairs live interleaved: index bits at odd lanes. Lanes whose
    // index is the -1 init sentinel (0xFFFFFFFF) or beyond the row's visible
    // pools stay -1 in the output. Scalar walk (<=512*4): vectorising is an
    // M3 tuning item; correctness first.
    constexpr uint32_t IDX_SENTINEL = 0xFFFFFFFFu;
    for (uint32_t j = 0; j < topkSel; j++) {
        uint32_t pid = rowTopk[j * 2 + 1].ReinterpretCast<uint32_t>().GetValue(0);
        if (pid == IDX_SENTINEL || pid >= visible) {
            continue;
        }
        uint32_t base = j * constInfo_.kpool;
        for (uint32_t c = 0; c < constInfo_.kpool; c++) {
            outUb_.SetValue(base + c, static_cast<int32_t>(pid * constInfo_.kpool + c));
        }
    }

    // causal tail: <= kpool-1 trailing tokens at min(causal_len, topk)+c
    uint64_t causalTokens = static_cast<uint64_t>(((pos + 1) / constInfo_.kpool) * constInfo_.kpool);
    uint32_t tailCount = static_cast<uint32_t>(pos + 1 - static_cast<int64_t>(causalTokens));
    if (tailCount > 0) {
        uint64_t tailStart = Min(causalTokens, static_cast<uint64_t>(constInfo_.topkTokens));
        for (uint32_t c = 0; c < tailCount; c++) {
            outUb_.SetValue(tailStart + c, static_cast<int32_t>(causalTokens + c));
        }
    }

    // scalar SetValue lanes are V-pipe writes too; re-sync before MTE3 reads
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
    Glm5KpoolVec::CopyOut(indicesOutGm[static_cast<uint64_t>(rowGlobal) * constInfo_.outputWidth],
                          outUb_, constInfo_.outputWidth);
    // release the staging buffer: MTE3 done, V may overwrite it next row
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::ProcessTopK(const RunInfo &runInfo)
{
#if GLMK_DEBUG_STAGE < 3
    (void)runInfo;
    return;
#endif
    for (uint32_t r = 0; r < ROWS_PER_AIV; r++) {
        if (posCache_[r] < 0) {
            continue;
        }
        EmitRow(r, runInfo);
    }
}

} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_SERVICE_VECTOR_H
