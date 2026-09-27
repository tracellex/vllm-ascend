/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_tiling.h
 * \brief
 */

#ifndef GLM5_KPOOL_INDEXER_TILING_H_
#define GLM5_KPOOL_INDEXER_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {
// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QBAR_INDEX = 0;
constexpr uint32_t INDEXER_CACHE_INDEX = 1;
constexpr uint32_t CUM_QUERY_LENS_INDEX = 2;
constexpr uint32_t INDEXER_SEQ_LENS_INDEX = 3;
constexpr uint32_t INDEXER_BLOCK_TABLE_INDEX = 4;
constexpr uint32_t POSITIONS_INDEX = 5;
// Outputs Index
constexpr uint32_t INDICES_INDEX = 0;
constexpr uint32_t SCORES_DEBUG_INDEX = 1;
// Attributes Index
constexpr uint32_t ATTR_TOPK_TOKENS_INDEX = 0;
constexpr uint32_t ATTR_KPOOL_INDEX = 1;
constexpr uint32_t ATTR_HEAD_DIM_INDEX = 2;
constexpr uint32_t ATTR_MAX_POOL_SEQ_LEN_INDEX = 3;
constexpr uint32_t ATTR_OUTPUT_MODE_INDEX = 4;
// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
// qbar rows are [q_hi | q_lo] packed (H9 FP32-split), so the input is twice
// as wide as the cache head dim.
constexpr uint32_t QBAR_WIDTH = 2 * HEAD_DIM_LIMIT;
constexpr uint32_t TOPK_TOKENS_LIMIT = 8192;
constexpr uint32_t OUTPUT_MODE_GROUP_TOPK = 3;
constexpr uint32_t OUTPUT_MODE_GROUP_TOPK_M64 = 4;
constexpr uint32_t POOL_GROUP = 1024;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(Glm5KpoolTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tSize)              // num tokens (M axis)
TILING_DATA_FIELD_DEF(uint32_t, bSize)              // num requests
TILING_DATA_FIELD_DEF(uint32_t, maxPoolSeqLen)      // S2 upper bound in pool units
TILING_DATA_FIELD_DEF(uint32_t, poolsPerBlock)      // cache block size in pool units
TILING_DATA_FIELD_DEF(uint32_t, numCacheBlocks)     // cache dim0 (for stale-entry clamping)
TILING_DATA_FIELD_DEF(uint32_t, blockTableStride)   // block_table stride(1) == max blocks per request
TILING_DATA_FIELD_DEF(uint32_t, topkTokens)         // total token top-k (2048)
TILING_DATA_FIELD_DEF(uint32_t, kpool)              // tokens per pool (4)
TILING_DATA_FIELD_DEF(uint32_t, poolTopk)           // topkTokens / kpool
TILING_DATA_FIELD_DEF(uint32_t, outputWidth)        // topkTokens + kpool - 1
TILING_DATA_FIELD_DEF(uint32_t, outputMode)         // 0 fused, 1/2 diagnostics, 3/4 grouped device top-k
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, isLDOpen)           // S2-split + LD merge enabled (M2+)
TILING_DATA_FIELD_DEF(uint32_t, s2SplitNum)         // S2 splits per (m-tile) when LD open
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(Glm5KpoolIndexer, Glm5KpoolTilingData)

// -----------算子CompileInfo定义-------------------
struct Glm5KpoolCompileInfo {};

// -----------算子Tiling入参信息类---------------
class Glm5KpoolTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t aicNum = 1;
    uint32_t aivNum = 1;

    uint32_t tSize = 0;            // num tokens
    uint32_t bSize = 0;            // num requests
    uint32_t maxPoolSeqLen = 0;    // S2 bound in pools
    uint32_t poolsPerBlock = 0;    // cache dim1
    uint32_t numCacheBlocks = 0;   // cache dim0
    uint32_t blockTableStride = 0; // block_table dim1
    uint32_t topkTokens = 0;
    uint32_t kpool = 0;
    uint32_t poolTopk = 0;
    uint32_t outputWidth = 0;
    uint32_t outputMode = 0;
    ge::DataType qbarType = ge::DT_BF16;
    ge::DataType cacheType = ge::DT_BF16;
};

ge::graphStatus TilingForGlm5KpoolIndexer(gert::TilingContext *context);

} // namespace optiling
#endif // GLM5_KPOOL_INDEXER_TILING_H_
