/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_topk.h
 * \brief Thin buffer/call wrapper around topkb32::LiTopKVF (uint32 keys).
 */

#ifndef GLM5_KPOOL_INDEXER_TOPK_H
#define GLM5_KPOOL_INDEXER_TOPK_H

#include "kernel_operator.h"
#include "vf_topk.h"

namespace topk {

template <typename T>
class Glm5KpoolTopk {
public:
    __aicore__ inline void operator()(LocalTensor<uint32_t> &outputIdxLocal, LocalTensor<T> &inputLocal,
                                      uint32_t s2SeqLen)
    {}
};

template <>
class Glm5KpoolTopk<uint32_t> {
public:
    static __aicore__ inline uint32_t GetSharedTmpBufferSize(uint32_t topK)
    {
        return 2 * topK * sizeof(uint32_t) + 5 * 256 * sizeof(uint32_t) + 64 * sizeof(uint32_t) +
               (topK + 64) * sizeof(uint32_t); // for output value tensor
    }

    __aicore__ inline void Init(uint32_t topK)
    {
        this->topK = topK;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t> &sharedTmpBuffer)
    {
        tmpIdxLocal = sharedTmpBuffer[0];
        tmpValueLocal = tmpIdxLocal[topK];
        histogramsLocal = tmpValueLocal[topK];
        idx0Local = histogramsLocal[256];
        idx1Local = idx0Local[256];
        idx2Local = idx1Local[256];
        idx3Local = idx2Local[256];
        nkValueLocal = idx3Local[256];
        outputValueLocal = nkValueLocal[64];
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t> &outputIdxLocal, LocalTensor<uint32_t> &inputLocal,
                                      uint32_t s2SeqLen)
    {
        topkb32::LiTopKVF(outputIdxLocal,  // filter阶段使用输出value Buf topK * 4B
                          outputValueLocal, // filter阶段使用输出 Idx Buf topK * 4B
                          inputLocal,       // 输入 s2SeqLen * 4B
                          tmpIdxLocal,      // filter阶段使用暂存index Buf topK * 4B
                          tmpValueLocal,    // filter阶段使用暂存value Buf topK * 4B
                          histogramsLocal,  // 直方图的临时Buf 256 * 4B
                          idx0Local,        // 输入数据第1个8位Buf 256 * 4B
                          idx1Local,        // 输入数据第2个8位Buf 256 * 4B
                          idx2Local,        // 输入数据第3个8位Buf 256 * 4B
                          idx3Local,        // 输入数据第4个8位Buf 256 * 4B
                          nkValueLocal,     // next_k 暂存Buf 64 * 4B
                          topK,             // topk数量
                          s2SeqLen);        // 输入元素总数
    }

private:
    LocalTensor<uint32_t> tmpIdxLocal;
    LocalTensor<uint32_t> tmpValueLocal;
    LocalTensor<uint32_t> histogramsLocal;
    LocalTensor<uint32_t> idx0Local;
    LocalTensor<uint32_t> idx1Local;
    LocalTensor<uint32_t> idx2Local;
    LocalTensor<uint32_t> idx3Local;
    LocalTensor<uint32_t> nkValueLocal;
    LocalTensor<uint32_t> outputValueLocal;
    uint32_t topK = 0;
};
} // namespace topk

#endif // GLM5_KPOOL_INDEXER_TOPK_H
