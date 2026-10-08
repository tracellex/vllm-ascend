/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_vector1.h
 * \brief fp32 score -> monotonic uint32 sort key (NaN-safe). Scores produced
 *        by the cube side (bf16 qbar x bf16 pool K accumulated in fp32) are
 *        converted here so the radix top-k sees a total order over floats,
 *        including negative weights and negative scores.
 */

#ifndef GLM5_KPOOL_INDEXER_VECTOR1_H
#define GLM5_KPOOL_INDEXER_VECTOR1_H

#include "kernel_operator.h"

namespace glm5k_vector1 {

template <typename T>
struct FloatSortTraits;

// fp32
template <>
struct FloatSortTraits<float> {
    using UInt = uint32_t;
    static constexpr UInt ZERO = 0x00000000;
    static constexpr UInt SIGN_MASK = 0x80000000;
    static constexpr UInt NAN_MASK = 0x7FC00000;
    static constexpr UInt ALL_ONE = 0xFFFFFFFF;
};

template <typename FloatT>
struct FloatSortConstCtx {
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;
    AscendC::MicroAPI::RegTensor<UInt> zeros;
    AscendC::MicroAPI::RegTensor<UInt> allOne;
    AscendC::MicroAPI::RegTensor<UInt> signMask;
    AscendC::MicroAPI::RegTensor<UInt> nan;
};

template <typename FloatT>
__simd_callee__ inline void InitFloatSortConstCtx(FloatSortConstCtx<FloatT> &ctx,
                                                  AscendC::MicroAPI::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    AscendC::MicroAPI::Duplicate(ctx.zeros, Traits::ZERO, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.allOne, Traits::ALL_ONE, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.signMask, Traits::SIGN_MASK, maskAll);
    AscendC::MicroAPI::Duplicate(ctx.nan, Traits::NAN_MASK, maskAll);
}

// IEEE float -> monotonic unsigned key: flip sign bit for positives, flip all
// bits for negatives, NaN maps to the largest key (can never outrank a real
// score's complement; NaN inputs are not expected from the matmul anyway).
template <typename FloatT>
__simd_callee__ inline void FloatToSortableKey(
    AscendC::MicroAPI::RegTensor<typename FloatSortTraits<FloatT>::UInt> &outKey,
    AscendC::MicroAPI::RegTensor<FloatT> &inVal, FloatSortConstCtx<FloatT> &ctx,
    AscendC::MicroAPI::MaskReg &maskAll)
{
    using Traits = FloatSortTraits<FloatT>;
    using UInt = typename Traits::UInt;

    AscendC::MicroAPI::RegTensor<UInt> regTemp;
    AscendC::MicroAPI::RegTensor<UInt> regMask;
    AscendC::MicroAPI::MaskReg regSelectNan;
    AscendC::MicroAPI::MaskReg regSelectSign;

    auto &inBits = (AscendC::MicroAPI::RegTensor<UInt> &)inVal;

    // 1. NaN check
    AscendC::MicroAPI::Compare<UInt, CMPMODE::EQ>(regSelectNan, inBits, ctx.nan, maskAll);
    // 2. NaN -> ALL_ONE
    AscendC::MicroAPI::Select(outKey, ctx.allOne, inBits, regSelectNan);
    // 3. sign bit
    AscendC::MicroAPI::And(regTemp, outKey, ctx.signMask, maskAll);
    AscendC::MicroAPI::Compare<UInt, CMPMODE::GT>(regSelectSign, regTemp, ctx.zeros, maskAll);
    // 4. xor mask: positive -> flip sign bit, negative -> flip all bits
    AscendC::MicroAPI::Select(regMask, ctx.allOne, ctx.signMask, regSelectSign);
    AscendC::MicroAPI::Xor(outKey, outKey, regMask, maskAll);
}

} // namespace glm5k_vector1

namespace glm5k_vector1 {

// Convert one row of fp32 scores to uint32 sort keys. Lanes >= validLanes
// (and the pad tail of the 128-lane tile) collapse to key 0 so they can never
// be selected by the radix top-k. Processed 64 lanes per MicroAPI repeat.
__aicore__ inline void ConvertScoreRowToKey(__ubuf__ uint32_t *outBuf, __ubuf__ float *inBuf,
                                            __ubuf__ float * /*maskBaseUnused*/, uint32_t totalLanes,
                                            uint32_t validLanes)
{
    (void)maskBaseUnused;
    using Reg = AscendC::MicroAPI;
    Reg::MaskReg maskAllB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    FloatSortConstCtx<float> ctx;
    InitFloatSortConstCtx(ctx, maskAllB32);

    const uint32_t repeatSize = 64; // u32 lanes per MicroAPI repeat
    uint32_t fullRepeats = validLanes / repeatSize;
    for (uint32_t i = 0; i < fullRepeats; i++) {
        Reg::RegTensor<float> inVal;
        Reg::RegTensor<uint32_t> outKey;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(inVal, inBuf + i * repeatSize);
        FloatToSortableKey<float>(outKey, inVal, ctx, maskAllB32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outBuf + i * repeatSize, outKey, maskAllB32);
    }
    // Partial repeat: convert all 64 lanes, store, then zero the invalid tail
    // of this repeat plus everything beyond it in the tile.
    uint32_t tailValid = validLanes % repeatSize;
    if (tailValid != 0) {
        Reg::RegTensor<float> inVal;
        Reg::RegTensor<uint32_t> outKey;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(inVal, inBuf + fullRepeats * repeatSize);
        FloatToSortableKey<float>(outKey, inVal, ctx, maskAllB32);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(outBuf + fullRepeats * repeatSize, outKey, maskAllB32);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t tailZero = repeatSize - tailValid;
        AscendC::Duplicate(outBuf[fullRepeats * repeatSize + tailValid], (uint32_t)0, tailZero);
    }
    // Zero the remainder of the lane tile beyond the last touched repeat.
    uint32_t zeroFrom = (tailValid == 0) ? validLanes : (fullRepeats + 1) * repeatSize;
    if (zeroFrom < totalLanes) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(outBuf[zeroFrom], (uint32_t)0, totalLanes - zeroFrom);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

} // namespace glm5k_vector1

#endif // GLM5_KPOOL_INDEXER_VECTOR1_H
