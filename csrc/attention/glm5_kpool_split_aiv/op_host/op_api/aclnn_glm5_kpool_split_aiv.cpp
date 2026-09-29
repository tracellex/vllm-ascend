#include <string.h>
#include "graph/types.h"
#include "aclnn_glm5_kpool_split_aiv.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerGlm5KpoolSplitAivGetWorkspaceSize(
    const aclTensor *cumQueryLens, const aclTensor *indexerSeqLens, const aclTensor *positions,
    const aclTensor *scores, const aclTensor *runningStrip,
    int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t splitBatch,
    const aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerGlm5KpoolSplitAiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                const aclrtStream stream);

aclnnStatus aclnnGlm5KpoolSplitAivGetWorkspaceSize(
        const aclTensor *cumQueryLens, const aclTensor *indexerSeqLens, const aclTensor *positions,
        const aclTensor *scores, const aclTensor *runningStrip,
        int64_t topkTokens, int64_t kpool, int64_t headDim, int64_t maxPoolSeqLen, int64_t splitBatch,
        const aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (cumQueryLens == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "cum_query_lens pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    return aclnnInnerGlm5KpoolSplitAivGetWorkspaceSize(
        cumQueryLens, indexerSeqLens, positions, scores, runningStrip, topkTokens, kpool, headDim,
        maxPoolSeqLen, splitBatch, indicesOut, workspaceSize, executor);
}

aclnnStatus aclnnGlm5KpoolSplitAiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    const aclrtStream stream)
{
    return aclnnInnerGlm5KpoolSplitAiv(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
