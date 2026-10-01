/**
 * SPDX-License-Identifier: Apache-2.0
 * \file glm5_kpool_split_aiv.cpp
 * \brief PURE AIV folding half of the de-mixed GLM5 KPool indexer pair.
 *        Consumes the batch scores written by glm5_kpool_split_aic plus the
 *        wrapper-owned cross-batch running-strip tensor, folds each
 *        1024-pool group, saves the strip (mid batches) or emits indices
 *        (last batch). See glm5_kpool_indexer_kernel_split.h for the mode
 *        contract.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "glm5_kpool_split_aiv_template_tiling_key.h"
#include "common/glm5_kpool_split_tiling.h"
#include "arch22/glm5_kpool_indexer_kernel_split.h"

using namespace AscendC;

template <int DT_Q>
__global__ __aicore__ void glm5_kpool_split_aiv(__gm__ uint8_t *cumQueryLens, __gm__ uint8_t *indexerSeqLens,
                                                __gm__ uint8_t *positions, __gm__ uint8_t *scoresIn,
                                                __gm__ uint8_t *runningStrip, __gm__ uint8_t *indicesOut,
                                                __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(Glm5KpoolTilingData);
    TPipe tPipe;
    GET_TILING_DATA_WITH_STRUCT(Glm5KpoolTilingData, tiling_data_in, tiling);
    const Glm5KpoolTilingData *__restrict tiling_data = &tiling_data_in;
    if (DT_Q == GLMK_SPLIT_AIV_BF16) {
        Glm5KpoolKernel::Glm5KpoolSplitAivKernel<bfloat16_t, Glm5KpoolCommon::M_TILE> op;
        op.Init(indicesOut, cumQueryLens, indexerSeqLens, positions, scoresIn, runningStrip, tiling_data,
                &tPipe);
        op.Process();
    } else {
        Glm5KpoolKernel::Glm5KpoolSplitAivKernel<half, Glm5KpoolCommon::M_TILE> op;
        op.Init(indicesOut, cumQueryLens, indexerSeqLens, positions, scoresIn, runningStrip, tiling_data,
                &tPipe);
        op.Process();
    }
}
