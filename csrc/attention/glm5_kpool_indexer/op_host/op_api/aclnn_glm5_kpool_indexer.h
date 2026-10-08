/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

#ifndef ACLNN_GLM5_KPOOL_INDEXER_H
#define ACLNN_GLM5_KPOOL_INDEXER_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnGlm5KpoolIndexerGetWorkspaceSize calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGlm5KpoolIndexerGetWorkspaceSize(
    const aclTensor *qbar,
    const aclTensor *indexerCache,
    const aclTensor *cumQueryLens,
    const aclTensor *indexerSeqLens,
    const aclTensor *indexerBlockTable,
    const aclTensor *positions,
    int64_t topkTokens,
    int64_t kpool,
    int64_t headDim,
    int64_t maxPoolSeqLen,
    int64_t outputMode,
    const aclTensor *indicesOut,
    const aclTensor *scoresDebugOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnGlm5KpoolIndexer is used to perform calculations.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGlm5KpoolIndexer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_GLM5_KPOOL_INDEXER_H
