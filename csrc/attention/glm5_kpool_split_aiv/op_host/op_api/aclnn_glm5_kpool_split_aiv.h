/** SPDX-License-Identifier: Apache-2.0 */
#ifndef ACLNN_GLM5_KPOOL_SPLIT_AIV_H
#define ACLNN_GLM5_KPOOL_SPLIT_AIV_H
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
__attribute__((visibility("default")))
aclnnStatus aclnnGlm5KpoolSplitAivGetWorkspaceSize(
    const aclTensor *cumQueryLens, const aclTensor *indexerSeqLens, const aclTensor *positions,
    const aclTensor *scores, const aclTensor *runningStrip,
    int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t splitBatch,
    const aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
__attribute__((visibility("default")))
aclnnStatus aclnnGlm5KpoolSplitAiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    const aclrtStream stream);
#ifdef __cplusplus
}
#endif
#endif
