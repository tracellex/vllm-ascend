/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_common.h
 * \brief Shared structs for the GLM5 KPool indexer arch22 kernel (A3 path).
 */

#ifndef GLM5_KPOOL_INDEXER_COMMON_H
#define GLM5_KPOOL_INDEXER_COMMON_H

using namespace AscendC;

namespace Glm5KpoolCommon {

// ------------------常量------------------
constexpr int32_t INVALID_IDX = -1;
constexpr uint32_t HEAD_DIM = 128;      // index_head_dim (fixed by def check)
constexpr uint32_t M_TILE = 32;         // token rows per cube base block; 16/AIV
                                        // rows keeps the running top-k strips in UB
constexpr uint32_t S2_TILE = 128;       // pools per cube base block
// qbar rows arrive FP32-split as [q_hi | q_lo] BF16 halves (H9): the cube
// accumulates q_hi@K + q_lo@K in FP32, keeping ~16 mantissa bits of the
// head-weighted query so top-k boundaries match the FP32 Triton reference.
constexpr uint32_t QBAR_ROW_ELEMS = 2 * HEAD_DIM;

// arch22 cross-core handshake (alternating lockstep, vendored v1 pattern).
constexpr uint32_t FIA_SYNC_MODE2 = 2;

// buffer 字节常量
constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;

// ------------------运行信息------------------
struct RunInfo {
    uint32_t loop = 0;              // global base-block counter (pingpong selector)
    uint32_t reqIdx = 0;            // request owning this m-tile
    uint32_t mTileIdx = 0;          // global token tile index
    uint32_t mStart = 0;            // first token row (global) of this tile
    uint32_t actMSize = 0;          // valid rows in this m tile (tail clamp)
    uint32_t s2TileIdx = 0;         // pool tile index within the request
    uint32_t s2Start = 0;           // first pool (logical) of this tile
    uint32_t actS2Size = 0;         // valid pools in this s2 tile (per-request bound)
    uint32_t actS2SizeAlign = 0;    // 32B-aligned mm1Res row width
    uint32_t reqPoolLen = 0;        // clamped pool length of the owning request
    uint32_t posBase = 0;           // pos of the tile's first row == mTileInReq * M_TILE
    uint64_t tensorQueryOffset = 0; // qbar row offset (elements) == mStart * QBAR_ROW_ELEMS
    uint64_t indicesOutOffset = 0;  // indices row offset (elements) == mStart * outputWidth
    bool isFirstS2InnerLoop = false;
    bool isLastS2InnerLoop = false;
    bool isValid = false;
};

// ------------------编译期常量信息------------------
struct ConstInfo {
    uint32_t tSize = 0;
    uint32_t bSize = 0;
    uint32_t maxPoolSeqLen = 0;
    uint32_t poolsPerBlock = 0;    // cache block size in pools (32)
    uint32_t numCacheBlocks = 0;   // cache dim0 (stale-entry clamp)
    uint32_t blockTableStride = 0; // block table columns per request
    uint32_t topkTokens = 0;       // 2048
    uint32_t kpool = 4;
    uint32_t poolTopk = 512;       // topkTokens / kpool
    uint32_t outputWidth = 2051;   // topkTokens + kpool - 1
    uint32_t outputMode = 0;
    uint32_t usedCoreNum = 0;
    uint32_t mBaseSize = M_TILE;
    uint32_t s2BaseSize = S2_TILE;
    uint32_t headDim = HEAD_DIM;
};

// ------------------分核信息------------------
struct SplitCoreInfo {
    uint32_t mTileStart = 0;  // 双闭区间, over (req, mTile) flattened
    uint32_t mTileEnd = 0;
    uint32_t s2TileStart = 0; // only the first m-tile keeps >0
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
