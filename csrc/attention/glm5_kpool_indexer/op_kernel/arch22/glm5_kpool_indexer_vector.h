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
constexpr uint8_t VEC_REPEAT_MAX = 255;
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
    int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
    int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
    for (int i = 0; i < forLoop; i++) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), NEG_INF, mask1, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), INVALID_INDEX, mask0, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
    }
    if (forRemain > 0) {
        uint64_t maskAll[2] = {~0ULL, ~0ULL};
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM],
                           NEG_INF, maskAll, forRemain, 1, B32_VEC_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM],
                           INVALID_INDEX, mask0, forRemain, 1, B32_VEC_REPEAT_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  dst: 输出全排序的结果，排布方式为value，index 交错
  srcValue：输入的待排序浮点数
  srcIndex：浮点数的索引
  tmp: 计算使用到的临时空间
  logitsNum: 排序的元素个数（32 的倍数）
 */
__aicore__ inline void SortAll(LocalTensor<float> &dst, LocalTensor<float> &srcValue, LocalTensor<uint32_t> &srcIndex,
                               LocalTensor<float> &tmpTensor, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();
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

} // namespace Glm5KpoolVec
#endif // GLM5_KPOOL_INDEXER_VECTOR_H
