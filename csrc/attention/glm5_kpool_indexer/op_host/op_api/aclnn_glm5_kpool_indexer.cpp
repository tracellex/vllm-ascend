/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_glm5_kpool_indexer.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerGlm5KpoolIndexerGetWorkspaceSize(
    const aclTensor *qbar, const aclTensor *indexerCache, const aclTensor *cumQueryLens,
    const aclTensor *indexerSeqLens, const aclTensor *indexerBlockTable, const aclTensor *positions,
    int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t outputMode,
    const aclTensor *indicesOut, const aclTensor *scoresDebugOut, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerGlm5KpoolIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               const aclrtStream stream);

aclnnStatus aclnnGlm5KpoolIndexerGetWorkspaceSize(
        const aclTensor *qbar, const aclTensor *indexerCache, const aclTensor *cumQueryLens,
        const aclTensor *indexerSeqLens, const aclTensor *indexerBlockTable, const aclTensor *positions,
        int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t outputMode,
        const aclTensor *indicesOut, const aclTensor *scoresDebugOut, uint64_t *workspaceSize,
        aclOpExecutor **executor)
{
    if (qbar == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "qbar pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (indicesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "indicesOut cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (scoresDebugOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "scoresDebugOut cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }

    return aclnnInnerGlm5KpoolIndexerGetWorkspaceSize(
        qbar, indexerCache, cumQueryLens, indexerSeqLens, indexerBlockTable, positions, topkTokens, kpool, headDim,
        maxPoolSeqLen, outputMode, indicesOut, scoresDebugOut, workspaceSize, executor);
}

aclnnStatus aclnnGlm5KpoolIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   const aclrtStream stream)
{
    return aclnnInnerGlm5KpoolIndexer(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
