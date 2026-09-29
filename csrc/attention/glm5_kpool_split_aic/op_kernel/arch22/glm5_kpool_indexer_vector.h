/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_vector.h
 * \brief Sort/merge helpers (Sort32 + MrgSort based, A3-verified pattern from
 *        the vendored lightning_indexer arch22 vector layer).
 */

#ifndef GLM5_KPOOL_INDEXER_VECTOR_H
#define GLM5_KPOOL_INDEXER_VECTOR_H

#include "kernel_operator.h"

namespace Glm5KpoolVec {
using namespace AscendC;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr int32_t INVALID_INDEX = -1;
constexpr uint8_t VEC_REPEAT_MAX = 127;
constexpr uint8_t B32_VEC_ELM_NUM = 64;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;
constexpr int64_t VALUE_AND_INDEX_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MRG_QUE_0 = 0;
constexpr int64_t MRG_QUE_1 = 1;
constexpr int64_t MRG_QUE_2 = 2;
constexpr int64_t MRG_QUE_3 = 3;
constexpr int64_t MRG_BLOCK_2 = 2;
constexpr int64_t MRG_BLOCK_3 = 3;
constexpr int64_t MRG_BLOCK_4 = 4;

// Bit-exact UB-to-UB copy on the vector pipe. A plain DataCopy may use an MTE
// pipe, so a following PIPE_V barrier does not make the result visible to the
// next Sort/MrgSort. Pair buffers are always copied in 64-word multiples.
__aicore__ inline void CopyWordsVec(const LocalTensor<float> &dst, const LocalTensor<float> &src, int32_t wordCount)
{
    LocalTensor<int32_t> dstI = dst.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> srcI = src.ReinterpretCast<int32_t>();
    for (int32_t off = 0; off < wordCount; off += B32_VEC_ELM_NUM) {
        AscendC::Adds(dstI[off], srcI[off], 0, B32_VEC_ELM_NUM);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}

/**
  src: 传入的初始化空间
  eleNum: 需要初始化的元素个数需为64整数倍，元素将被初始化为交错排布的-inf，-1
 */
__aicore__ inline void InitSortOutBuf(const LocalTensor<float> &src, int64_t eleNum)
{
    uint64_t mask1[2] = {0x5555555555555555, 0};
    uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
    int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
    int64_t repeatOffset = 0;
    while (repeatOffset < repeatNum) {
        int64_t remaining = repeatNum - repeatOffset;
        uint8_t repeatCount = static_cast<uint8_t>(remaining > VEC_REPEAT_MAX ? VEC_REPEAT_MAX : remaining);
        int64_t elementOffset = repeatOffset * B32_VEC_ELM_NUM;
        auto chunk = src.template ReinterpretCast<int32_t>()[elementOffset];
        AscendC::Duplicate(chunk, NEG_INF, mask1, repeatCount, 1, B32_VEC_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(chunk, INVALID_INDEX, mask0, repeatCount, 1, B32_VEC_REPEAT_STRIDE);
        repeatOffset += repeatCount;
    }
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  dst: 输出全排序的结果，排布方式为value，index 交错
  srcValue：输入的待排序浮点数
  srcIndex：浮点数的索引
  tmp: 计算使用到的临时空间（至少 logitsNum*2 words，兼作层级归并输出）
  logitsNum: 排序的元素个数（32 的倍数）
 */
__aicore__ inline void SortAll(LocalTensor<float> &dst, LocalTensor<float> &srcValue, LocalTensor<uint32_t> &srcIndex,
                               LocalTensor<float> &tmpTensor, int64_t logitsNum)
{
    // Step 1: intra-32 group sort. Sort only orders within each 32-element
    // repeat: the output is mrgGroups independently sorted runs, NOT a full
    // ordering. Skipping the hierarchical merge below fed pseudo-sorted input
    // to MrgSort and silently corrupted every fold where the running top-k
    // actually had to evict losers (visible_pools >= poolTopk); below
    // visible < poolTopk the -inf padding absorbed the damage.
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();

    // Step 2: one 4-way merge run -> full ordering (S2_TILE == 128 == 4 runs
    // of 32; lightining_indexer_quant_vector.h verified MrgSort4 pattern).
    if (sort32Repeats > 1) {
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = BLOCK_BYTES;
        params.elementLengths[MRG_QUE_1] = BLOCK_BYTES;
        params.elementLengths[MRG_QUE_2] = BLOCK_BYTES;
        params.elementLengths[MRG_QUE_3] = BLOCK_BYTES;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = dst[0];
        srcList.src2 = dst[MRG_QUE_1 * VALUE_AND_INDEX_NUM * BLOCK_BYTES];
        srcList.src3 = dst[MRG_QUE_2 * VALUE_AND_INDEX_NUM * BLOCK_BYTES];
        srcList.src4 = dst[MRG_QUE_3 * VALUE_AND_INDEX_NUM * BLOCK_BYTES];
        AscendC::MrgSort<float>(tmpTensor, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(dst, tmpTensor, logitsNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
  mrgDst: 合并进的Tensor（value,index 交错有序）
  mrgSrc: 待合并的Tensor（value,index 交错有序）
  tmpTensor：空间为 mrgDst+mrgSrc
 */
__aicore__ inline void MergeSort(const LocalTensor<float> &mrgDst, int32_t mrgDstNum, LocalTensor<float> &mrgSrc,
                                 int32_t mrgSrcNum, LocalTensor<float> &tmpTensor)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[MRG_QUE_0] = mrgSrcNum;
    params.elementLengths[MRG_QUE_1] = mrgDstNum;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = mrgSrc;
    srcList.src2 = mrgDst;

    AscendC::MrgSort<float>(tmpTensor, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
 * @brief 从排序结果(value,index 交错)中抽取 index 序列
 */
__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t> &idxULocal, const LocalTensor<uint32_t> &sortLocal,
                                    int64_t extractNum)
{
    AscendC::GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2; // 固定模式2,表示筛选出奇数索引的数
    AscendC::GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
 * srcValue/srcIndex: 1024 separate score/id lanes. dst: the sorted top-512
 * interleaved (value,index) prefix. tmp: >= 2048 words scratch. Full
 * descending sort via Sort32(32 runs) + fixed 4-way merge tree
 * (32x32 -> 8x128 -> 2x512 -> needed prefix of the final merge). The final
 * merge suspends when either 512-pair input queue is exhausted; it may emit
 * 512..1023 pairs, not a fixed top-512 count. Only its top-512 prefix is copied
 * because group-topk discards the remainder. Every queue
 * granularity on this path (32/128/512 pairs) matches a stage of lightning_indexer's
 * production-verified SortAll; the single-shot MrgSort over a partially
 * filled strip (our eviction bug class) is never used.
 */
__aicore__ inline void SortFull1024(const LocalTensor<float> &dst, LocalTensor<float> &tmpTensor,
                                    LocalTensor<float> &srcValue, LocalTensor<uint32_t> &srcIndex)
{
    constexpr int64_t PAIRS = 1024;
    constexpr int64_t TOPK_WORDS = 512 * VALUE_AND_INDEX_NUM;
    constexpr int64_t RUN = 32;
    // Sort consumes separate scores/ids and produces interleaved pairs.
    AscendC::Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, PAIRS / 32);
    AscendC::PipeBarrier<PIPE_V>();
    // Merge tree, alternating dst/tmp. Stage A: 32 runs of 32 -> 8 runs of 128
    {
        AscendC::MrgSort4Info p;
        p.elementLengths[0] = RUN; p.elementLengths[1] = RUN;
        p.elementLengths[2] = RUN; p.elementLengths[3] = RUN;
        p.ifExhaustedSuspension = false;
        p.validBit = 0b1111;
        p.repeatTimes = 8;
        AscendC::MrgSortSrcList<float> s;
        s.src1 = dst[0];
        s.src2 = dst[RUN * VALUE_AND_INDEX_NUM * 1];
        s.src3 = dst[RUN * VALUE_AND_INDEX_NUM * 2];
        s.src4 = dst[RUN * VALUE_AND_INDEX_NUM * 3];
        AscendC::MrgSort<float>(tmpTensor, s, p);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // Stage B: 8 runs of 128 -> 2 runs of 512 (repeat steps over run groups of 4)
    {
        AscendC::MrgSort4Info p;
        p.elementLengths[0] = 128; p.elementLengths[1] = 128;
        p.elementLengths[2] = 128; p.elementLengths[3] = 128;
        p.ifExhaustedSuspension = false;
        p.validBit = 0b1111;
        p.repeatTimes = 2;
        AscendC::MrgSortSrcList<float> s;
        s.src1 = tmpTensor[0];
        s.src2 = tmpTensor[128 * VALUE_AND_INDEX_NUM * 1];
        s.src3 = tmpTensor[128 * VALUE_AND_INDEX_NUM * 2];
        s.src4 = tmpTensor[128 * VALUE_AND_INDEX_NUM * 3];
        AscendC::MrgSort<float>(dst, s, p);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // Stage C: merge two 512-pair runs; consume only the needed top-512 prefix.
    {
        AscendC::MrgSort4Info p;
        p.elementLengths[0] = 512; p.elementLengths[1] = 512;
        p.elementLengths[2] = 0; p.elementLengths[3] = 0;
        p.ifExhaustedSuspension = true;
        p.validBit = 0b0011;
        p.repeatTimes = 1;
        AscendC::MrgSortSrcList<float> s;
        s.src1 = dst[0];
        s.src2 = dst[512 * VALUE_AND_INDEX_NUM];
        AscendC::MrgSort<float>(tmpTensor, s, p);
        AscendC::PipeBarrier<PIPE_V>();
        CopyWordsVec(dst, tmpTensor, TOPK_WORDS);
    }
}

} // namespace Glm5KpoolVec
#endif // GLM5_KPOOL_INDEXER_VECTOR_H
