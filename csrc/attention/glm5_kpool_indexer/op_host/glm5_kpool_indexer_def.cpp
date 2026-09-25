/**
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 */

/*!
 * \file glm5_kpool_indexer_def.cpp
 * \brief GLM-Next KPool lightning indexer: score head-weighted pooled queries
 *        against a paged compressed-K pool cache, select top-k pools, expand
 *        to token indices and append the causal tail, all in one operator.
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class Glm5KpoolIndexer : public OpDef {
public:
    explicit Glm5KpoolIndexer(const char *name) : OpDef(name)
    {
        // [numTokens, headDim], head-weighted query (sum_g w_g * q_g), bf16.
        this->Input("qbar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [numBlocks, poolsPerBlock, 1, headDim] paged pool cache, PA layout.
        this->Input("indexer_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [batch] inclusive query-end offsets, no leading zero.
        this->Input("cum_query_lens")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [batch] per-request cache length in POOL units (not tokens).
        this->Input("indexer_seq_lens")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [batch, maxBlocksPerReq] logical->physical block mapping.
        this->Input("indexer_block_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [numTokens] absolute token positions (int64 from vLLM).
        this->Input("positions")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [numTokens, 1, topkTokens + kpool - 1] selected token indices, -1 padded.
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        // output_mode=1: [numTokens, maxPoolSeqLen] raw fp32 pool scores (debug).
        this->Output("scores_debug")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("topk_tokens").AttrType(REQUIRED).Int();
        this->Attr("kpool").AttrType(REQUIRED).Int();
        this->Attr("head_dim").AttrType(REQUIRED).Int();
        this->Attr("max_pool_seq_len").AttrType(REQUIRED).Int();
        this->Attr("output_mode").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(Glm5KpoolIndexer);
} // namespace ops
