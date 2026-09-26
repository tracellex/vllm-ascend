/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_common.h
 * \brief Shared structs for the GLM5 KPool indexer arch35 kernel.
 */

#ifndef GLM5_KPOOL_INDEXER_COMMON_H
#define GLM5_KPOOL_INDEXER_COMMON_H

using namespace AscendC;

namespace Glm5KpoolCommon {

// NOTE (H9, 2026-09-27): the op contract now feeds qbar as [T, 2*headDim]
// packing [q_hi | q_lo] bf16 halves (see arch22 common.h / the torch
// wrapper). arch22 consumes them with dual init-Mmads into separate L0C
// tiles plus a vector-side fp32 add; this arch35 path still reads single
// [T, headDim] rows and was never device-validated. Port the dual-half
// treatment here before building any 950-class SOC.
// ------------------常量------------------
constexpr int32_t INVALID_IDX = -1;
constexpr uint32_t HEAD_DIM = 128;      // index_head_dim (fixed by def check)
constexpr uint32_t M_TILE = 128;        // token rows per cube base block
constexpr uint32_t S2_TILE = 128;       // pools per cube base block
constexpr uint32_t KPOOL_DEFAULT = 4;   // tokens per pool (GLM-5.3-Flash)
constexpr uint32_t OUT_TAIL = 3;        // kpool - 1 causal tail columns

// CUBE与VEC核间同步的模式与事件
constexpr uint32_t GLM5_SYNC_MODE4 = 4;
constexpr uint32_t AIV0_AIV1_OFFSET = 16;
constexpr uint32_t CROSS_VC_EVENT = 0;
constexpr uint32_t CROSS_CV_EVENT = 2;

// buffer 字节常量
constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;

// ------------------运行信息------------------
struct RunInfo {
    uint32_t loop = 0;           // global base-block counter (pingpong selector)
    uint32_t reqIdx = 0;         // request owning this m-tile
    uint32_t mTileIdx = 0;       // global token tile index (tile-local rows = actMSize)
    uint32_t mStart = 0;         // first token row (global) of this tile
    uint32_t actMSize = 0;       // valid rows in this m tile (tail clamp)
    uint32_t s2TileIdx = 0;      // pool tile index within the request
    uint32_t s2Start = 0;        // first pool (logical) of this tile
    uint32_t actS2Size = 0;      // valid pools in this s2 tile (per-request bound)
    uint32_t actS2SizeAlign = 0; // 32B-aligned copy width
    uint64_t tensorQueryOffset = 0; // qbar row offset (elements) == mStart * HEAD_DIM
    uint64_t scoreRowOffset = 0;    // scoreGm row offset (elements) for this tile
    uint64_t indicesOutOffset = 0;  // indices row offset (elements) == mStart * outputWidth
    bool isFirstS2InnerLoop = false;
    bool isLastS2InnerLoop = false;
    bool isValid = false;
};

// ------------------编译期常量信息------------------
struct ConstInfo {
    uint32_t tSize = 0;            // total tokens this launch
    uint32_t bSize = 0;            // requests
    uint32_t maxPoolSeqLen = 0;    // static S2 bound (pools)
    uint32_t maxPoolSeqLenAlign = 0; // align(maxPoolSeqLen, S2_TILE): scoreGm row stride
    uint32_t poolsPerBlock = 0;    // cache block size in pools (32)
    uint32_t numCacheBlocks = 0;   // cache dim0 (stale-entry clamp)
    uint32_t blockTableStride = 0; // block table columns per request
    uint32_t topkTokens = 0;       // 2048
    uint32_t kpool = 4;            // 4
    uint32_t poolTopk = 512;       // topkTokens / kpool
    uint32_t outputWidth = 2051;   // topkTokens + kpool - 1
    uint32_t outputMode = 0;       // 0 fused / 1 raw scores
    uint32_t usedCoreNum = 0;
    uint32_t mBaseSize = M_TILE;
    uint32_t s2BaseSize = S2_TILE;
    uint32_t headDim = HEAD_DIM;
};

// ------------------分核信息------------------
struct SplitCoreInfo {
    uint32_t mTileStart = 0;  // 双闭区间, over (req, mTile) flattened
    uint32_t mTileEnd = 0;
    uint32_t s2TileStart = 0; // per m-tile-range start (only first m-tile keeps >0)
    uint32_t s2TileEnd = 0;
    bool isCoreEnable = false;
};

// ------------------工具------------------
template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd * rnd);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (rnd == 0) ? 0 : ((num + rnd - 1) / rnd);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

} // namespace Glm5KpoolCommon

#endif // GLM5_KPOOL_INDEXER_COMMON_H
