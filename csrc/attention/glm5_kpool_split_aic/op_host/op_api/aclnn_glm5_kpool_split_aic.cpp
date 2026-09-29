#include <string.h>
#include "graph/types.h"
#include "aclnn_glm5_kpool_split_aic.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerGlm5KpoolSplitAicGetWorkspaceSize(
    const aclTensor *qbar, const aclTensor *indexerCache, const aclTensor *cumQueryLens,
    const aclTensor *indexerSeqLens, const aclTensor *indexerBlockTable, const aclTensor *positions,
    int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t splitBatch,
    const aclTensor *scoresOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerGlm5KpoolSplitAic(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                const aclrtStream stream);

aclnnStatus aclnnGlm5KpoolSplitAicGetWorkspaceSize(
        const aclTensor *qbar, const aclTensor *indexerCache, const aclTensor *cumQueryLens,
        const aclTensor *indexerSeqLens, const aclTensor *indexerBlockTable, const aclTensor *positions,
        int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t splitBatch,
        const aclTensor *scoresOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (qbar == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "qbar pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    return aclnnInnerGlm5KpoolSplitAicGetWorkspaceSize(
        qbar, indexerCache, cumQueryLens, indexerSeqLens, indexerBlockTable, positions, topkTokens, kpool,
        headDim, maxPoolSeqLen, splitBatch, scoresOut, workspaceSize, executor);
}

aclnnStatus aclnnGlm5KpoolSplitAic(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    const aclrtStream stream)
{
    return aclnnInnerGlm5KpoolSplitAic(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
