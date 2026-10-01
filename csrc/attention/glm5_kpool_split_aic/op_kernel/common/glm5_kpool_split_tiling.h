/**
 * Plain-struct mirror of the op_host TILING_DATA_FIELD_DEF layout for the
 * kernel TU (k2q_csr_meta pattern): field order and types must stay
 * byte-identical with op_host/glm5_kpool_split_*_tiling.h.
 */
#ifndef GLMK_SPLIT_KERNEL_TILING_H
#define GLMK_SPLIT_KERNEL_TILING_H
#include <cstdint>
struct Glm5KpoolTilingData {
    uint32_t tSize;
    uint32_t bSize;
    uint32_t maxPoolSeqLen;
    uint32_t poolsPerBlock;
    uint32_t numCacheBlocks;
    uint32_t blockTableStride;
    uint32_t topkTokens;
    uint32_t kpool;
    uint32_t poolTopk;
    uint32_t outputWidth;
    uint32_t outputMode;
    uint32_t usedCoreNum;
    uint32_t splitBatch;
    uint32_t isLDOpen;
    uint32_t s2SplitNum;
};
#endif
