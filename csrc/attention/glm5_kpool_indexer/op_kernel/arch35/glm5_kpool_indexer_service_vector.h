/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_service_vector.h
 * \brief Vector side: per base block, convert the cube's fp32 pool scores to
 *        monotonic uint32 keys and append them to the per-core score strip in
 *        GM; once a row's whole S2 range is covered, run the radix top-k over
 *        the strip, expand pool ids to token ids (x kpool), pad with -1 and
 *        place the causal tail.
 */

#ifndef GLM5_KPOOL_INDEXER_SERVICE_VECTOR_H
#define GLM5_KPOOL_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "glm5_kpool_indexer_common.h"
#include "vf/glm5_kpool_indexer_vector1.h"
#include "vf/vf_topk.h"
#include "vf/glm5_kpool_indexer_topk.h"

namespace Glm5KpoolKernel {
using namespace Glm5KpoolCommon;

template <typename Q_T>
class Glm5KpoolServiceVector {
public:
    using SCORE_KEY_T = uint32_t;

    __aicore__ inline Glm5KpoolServiceVector(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitInputTensor(const GlobalTensor<int32_t> &indicesOutGm,
                                           const GlobalTensor<float> &scoresDebugGm,
                                           const GlobalTensor<int32_t> &positionsGm,
                                           const GlobalTensor<int32_t> &cumQueryLensGm,
                                           const GlobalTensor<int32_t> &indexerSeqLensGm);
    __aicore__ inline void InitWorkspaceTensor(const GlobalTensor<SCORE_KEY_T> &scoreGm);
    __aicore__ inline void ProcessVec1(const RunInfo &runInfo);
    __aicore__ inline void ProcessTopK(const RunInfo &runInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID2;
    // Rows per AIV per base block (dual-dst halves of M_TILE).
    static constexpr uint32_t AIV_ROWS = M_TILE / 2;

protected:
    __aicore__ inline uint32_t RowVisiblePools(uint32_t reqIdx, int64_t pos, int32_t reqPoolLen) const;
    __aicore__ inline void ProcessTopKRow(uint32_t rowGlobal, uint32_t rowInTile, const RunInfo &runInfo);
    __aicore__ inline void WriteExpandedOutput(LocalTensor<uint32_t> &poolIdLocal, uint32_t validPools,
                                               uint32_t topkSel, int64_t pos, uint64_t indicesRowOffset,
                                               uint32_t tailCount);

    ConstInfo constInfo_{};
    GlobalTensor<SCORE_KEY_T> scoreGm;       // [aicNum * M_TILE, maxPoolAlign]
    GlobalTensor<int32_t> indicesOutGm;      // [T, 1, outputWidth]
    GlobalTensor<float> scoresDebugGm;       // [T, maxPoolSeqLen] or empty
    GlobalTensor<int32_t> positionsGm;       // [T pad 128]
    GlobalTensor<int32_t> cumQueryLensGm;    // [B]
    GlobalTensor<int32_t> indexerSeqLensGm;  // [B]

    // Vec1: fp32 scores landed by fixpipe (AIV half rows x S2_TILE)
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<float> resMm1UB_;
    // Vec1: converted keys before GM store
    TBuf<TPosition::VECCALC> keyBuf_;
    LocalTensor<SCORE_KEY_T> keyUB_;
    // TopK: whole-row key strip (<= 128KB @128k ctx) + shared tmp
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_KEY_T> mrgValueLocal_;
    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<SCORE_KEY_T> topkSharedTmpLocal_;
    // TopK outputs: pool ids then expanded token ids
    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;
    TBuf<TPosition::VECCALC> poolIdBuf_;
    LocalTensor<uint32_t> poolIdLocal_;

    topk::Glm5KpoolTopk<SCORE_KEY_T> topkOp_;
    uint32_t blockId_ = 0;
    uint32_t aivHalf_ = 0; // 0/1: which half of M_TILE this AIV owns
};

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
    blockId_ = GetBlockIdx();
    aivHalf_ = blockId_ % 2;
    topkOp_.Init(constInfo.poolTopk);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, AIV_ROWS * S2_TILE * sizeof(float));
    resMm1UB_ = resMm1Buf_.Get<float>();
    pipe->InitBuffer(keyBuf_, AIV_ROWS * S2_TILE * sizeof(SCORE_KEY_T));
    keyUB_ = keyBuf_.Get<SCORE_KEY_T>();
    pipe->InitBuffer(mrgValueBuf_, constInfo_.maxPoolSeqLenAlign * sizeof(SCORE_KEY_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_KEY_T>();
    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize(constInfo_.poolTopk);
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<SCORE_KEY_T>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);
    pipe->InitBuffer(poolIdBuf_, (constInfo_.poolTopk + 64) * sizeof(uint32_t));
    poolIdLocal_ = poolIdBuf_.Get<uint32_t>();
    pipe->InitBuffer(indicesOutBuf_, constInfo_.outputWidth * sizeof(uint32_t));
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitInputTensor(const GlobalTensor<int32_t> &indicesOutGm,
                                                                    const GlobalTensor<float> &scoresDebugGm,
                                                                    const GlobalTensor<int32_t> &positionsGm,
                                                                    const GlobalTensor<int32_t> &cumQueryLensGm,
                                                                    const GlobalTensor<int32_t> &indexerSeqLensGm)
{
    this->indicesOutGm = indicesOutGm;
    this->scoresDebugGm = scoresDebugGm;
    this->positionsGm = positionsGm;
    this->cumQueryLensGm = cumQueryLensGm;
    this->indexerSeqLensGm = indexerSeqLensGm;
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::InitWorkspaceTensor(const GlobalTensor<SCORE_KEY_T> &scoreGm)
{
    this->scoreGm = scoreGm;
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
}

template <typename Q_T>
__aicore__ inline uint32_t Glm5KpoolServiceVector<Q_T>::RowVisiblePools(uint32_t reqIdx, int64_t pos,
                                                                        int32_t reqPoolLen) const
{
    uint32_t causalPools = static_cast<uint32_t>((pos + 1) >> 2); // kpool == 4 (asserted in tiling)
    uint32_t vis = Min(causalPools, static_cast<uint32_t>(Max(reqPoolLen, 0)));
    vis = Min(vis, constInfo_.maxPoolSeqLen);
    return vis;
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::ProcessVec1(const RunInfo &runInfo)
{
    auto pingpong = runInfo.loop % 2;
    // Wait for the cube's fixpipe: fp32 scores for this AIV's half rows are in UB.
    CrossCoreWaitFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_CV_EVENT + pingpong);

    // Per-row causal bound: pools beyond visible must read as key 0 (never
    // selected). The cube computed the full tile against the request bound,
    // so only the lane range [visible - s2Start, actS2Size) needs zeroing per row.
    for (uint32_t r = 0; r < AIV_ROWS; r++) {
        uint32_t rowGlobal = runInfo.mStart + aivHalf_ * AIV_ROWS + r;
        if (rowGlobal >= constInfo_.tSize) {
            continue; // padded row: no score store, no output later
        }
        int64_t pos = positionsGm.GetValue(rowGlobal);
        int32_t reqPoolLen = indexerSeqLensGm.GetValue(runInfo.reqIdx);
        uint32_t visible = RowVisiblePools(runInfo.reqIdx, pos, reqPoolLen);
        uint32_t validLanes = 0;
        if (runInfo.s2Start < visible) {
            validLanes = Min(S2_TILE, visible - runInfo.s2Start);
        }
        // Copy fp32 lane -> sortable key; invalid lanes collapse to key 0.
        // fp32 -> u32 key conversion is bit surgery; do it lane-wise via the
        // MicroAPI helper on the full 128-lane vector with a lane mask.
        glm5k_vector1::ConvertScoreRowToKey(keyUB_[r * S2_TILE], resMm1UB_[r * S2_TILE],
                                            resMm1UB_[r * S2_TILE + validLanes], S2_TILE, validLanes);
    }

    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT);
    // Store keys to the per-core score strip.
    for (uint32_t r = 0; r < AIV_ROWS; r++) {
        uint32_t rowGlobal = runInfo.mStart + aivHalf_ * AIV_ROWS + r;
        if (rowGlobal >= constInfo_.tSize) {
            continue;
        }
        uint64_t rowBase = runInfo.scoreRowOffset + (aivHalf_ * AIV_ROWS + r) * constInfo_.maxPoolSeqLenAlign;
        DataCopy(scoreGm[rowBase + runInfo.s2Start], keyUB_[r * S2_TILE], S2_TILE);
    }
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);

    if (constInfo_.outputMode == 1) {
        // Debug: also drop the raw fp32 scores (score strip is keyed, keep a
        // parallel fp32 view for accuracy diffing).
        WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
        for (uint32_t r = 0; r < AIV_ROWS; r++) {
            uint32_t rowGlobal = runInfo.mStart + aivHalf_ * AIV_ROWS + r;
            if (rowGlobal >= constInfo_.tSize) {
                continue;
            }
            int64_t pos = positionsGm.GetValue(rowGlobal);
            int32_t reqPoolLen = indexerSeqLensGm.GetValue(runInfo.reqIdx);
            uint32_t visible = RowVisiblePools(runInfo.reqIdx, pos, reqPoolLen);
            uint32_t validLanes = (runInfo.s2Start < visible) ? Min(S2_TILE, visible - runInfo.s2Start) : 0;
            if (validLanes == 0) {
                continue;
            }
            // fp32 scores are in UB (pre-mask); masked lanes already hold
            // garbage only beyond validLanes, so copy exactly the valid span.
            DataCopyPadParams padParams{false, 0, 0, 0};
            DataCopy(scoresDebugGm[static_cast<uint64_t>(rowGlobal) * constInfo_.maxPoolSeqLen +
                                   runInfo.s2Start],
                     resMm1UB_[r * S2_TILE], validLanes, padParams);
        }
    }

    // Release the pingpong slot so the cube can fixpipe the next block.
    CrossCoreSetFlag<ConstInfo::GLM5_SYNC_MODE4, PIPE_V>(ConstInfo::CROSS_VC_EVENT + pingpong);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::ProcessTopK(const RunInfo &runInfo)
{
    // Called on the LAST s2 tile of an m tile: run top-k for this AIV's rows.
    for (uint32_t r = 0; r < AIV_ROWS; r++) {
        uint32_t rowInTile = aivHalf_ * AIV_ROWS + r;
        uint32_t rowGlobal = runInfo.mStart + rowInTile;
        if (rowGlobal >= constInfo_.tSize) {
            continue;
        }
        ProcessTopKRow(rowGlobal, rowInTile, runInfo);
    }
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::ProcessTopKRow(uint32_t rowGlobal, uint32_t rowInTile,
                                                                   const RunInfo &runInfo)
{
    int64_t pos = positionsGm.GetValue(rowGlobal);
    int32_t reqPoolLen = indexerSeqLensGm.GetValue(runInfo.reqIdx);
    uint32_t visible = RowVisiblePools(runInfo.reqIdx, pos, reqPoolLen);

    // Pull the whole key strip for this row, zero-filled beyond `visible`.
    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    if (visible > 0) {
        WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT);
        uint64_t rowBase = runInfo.scoreRowOffset + rowInTile * constInfo_.maxPoolSeqLenAlign;
        DataCopy(mrgValueLocal_, scoreGm[rowBase], constInfo_.maxPoolSeqLenAlign);
        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
        // Zero the tail lanes beyond visible (align 256 for the histogram).
        uint32_t zeroLen = constInfo_.maxPoolSeqLenAlign - Align(visible, (uint32_t)BUFFER_SIZE_BYTE_256B / sizeof(SCORE_KEY_T));
        if (zeroLen > 0) {
            uint32_t zeroStart = Align(visible, (uint32_t)BUFFER_SIZE_BYTE_256B / sizeof(SCORE_KEY_T));
            Duplicate(mrgValueLocal_[zeroStart], (SCORE_KEY_T)0, zeroLen);
        }
        PipeBarrier<PIPE_V>();
    } else {
        Duplicate(mrgValueLocal_, (SCORE_KEY_T)0, constInfo_.maxPoolSeqLenAlign);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);

    uint32_t topkSel = Min(constInfo_.poolTopk, Align(visible, (uint32_t)BUFFER_SIZE_BYTE_256B / sizeof(SCORE_KEY_T)));
    if (visible == 0) {
        // No pool at all: whole row is -1 except the causal tail.
        WriteExpandedOutput(poolIdLocal_, 0, 0, pos,
                            static_cast<uint64_t>(rowGlobal) * constInfo_.outputWidth, 0);
        return;
    }

    topkOp_(poolIdLocal_, mrgValueLocal_, constInfo_.maxPoolSeqLenAlign);
    PipeBarrier<PIPE_V>();

    uint32_t tailCount = static_cast<uint32_t>(pos + 1 - ((pos + 1) / constInfo_.kpool) * constInfo_.kpool);
    WriteExpandedOutput(poolIdLocal_, visible, topkSel, pos,
                        static_cast<uint64_t>(rowGlobal) * constInfo_.outputWidth, tailCount);
}

template <typename Q_T>
__aicore__ inline void Glm5KpoolServiceVector<Q_T>::WriteExpandedOutput(LocalTensor<uint32_t> &poolIdLocal,
                                                                        uint32_t visible, uint32_t topkSel,
                                                                        int64_t pos, uint64_t indicesRowOffset,
                                                                        uint32_t tailCount)
{
    // History columns: pool p -> tokens p*kpool + {0..kpool-1}; pool ids whose
    // key is 0 (unselected / invalid) collapse to -1. Pool 0 can legitimately
    // win with a zero-score key only if ALL its lanes were masked; key 0 is
    // also what zero-filled lanes produce, so treat key==0 as unselected.
    // The radix top-k returns indices in descending key order.
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    int32_t neg = INVALID_IDX;
    Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, constInfo_.outputWidth);
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    if (topkSel > 0) {
        // scalar walk: 512 pools x 4 lanes — cheap next to the radix passes
        for (uint32_t j = 0; j < topkSel; j++) {
            uint32_t pid = poolIdLocal_.GetValue(j);
            // key==0 lanes were zero-filled; the topk still emits them when
            // fewer than poolTopk real candidates exist. Their pid points at
            // a zero lane; detect via scoreGm later would need another read,
            // so rely on the strip: unselected lanes sit beyond `visible`.
            uint32_t base = j * constInfo_.kpool;
            if (pid >= visible) {
                continue; // stays -1
            }
            for (uint32_t c = 0; c < constInfo_.kpool; c++) {
                indicesOutLocal_.GetValue(base + c) = pid * constInfo_.kpool + c;
            }
        }
        PipeBarrier<PIPE_V>();
    }

    // Causal tail (append_causal_tail semantics): up to kpool-1 trailing
    // tokens land at min(causal_len*kpool, topk_tokens) + {0..tailCount-1}.
    if (tailCount > 0) {
        uint64_t causalTokens = ((pos + 1) / constInfo_.kpool) * constInfo_.kpool;
        uint64_t tailStart = Min(causalTokens, static_cast<uint64_t>(constInfo_.topkTokens));
        for (uint32_t c = 0; c < tailCount; c++) {
            indicesOutLocal_.GetValue(tailStart + c) = static_cast<uint32_t>(causalTokens + c);
        }
    }

    // Persist: one row of outputWidth int32.
    DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = constInfo_.outputWidth * sizeof(int32_t);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyPad(indicesOutGm[indicesRowOffset], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

} // namespace Glm5KpoolKernel
#endif // GLM5_KPOOL_INDEXER_SERVICE_VECTOR_H
