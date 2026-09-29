#include <cstdint>
#include "register/op_def.h"
#include "register/op_def_registry.h"

namespace ops {
class Glm5KpoolSplitAiv : public OpDef {
public:
    explicit Glm5KpoolSplitAiv(const char *name) : OpDef(name)
    {
        this->Input("cum_query_lens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("indexer_seq_lens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("positions").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // [T_pad, SPLIT_BATCH_POOLS] fp32 scores produced by Glm5KpoolSplitAic.
        this->Input("scores").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // [T_pad, poolTopk*2] fp32 cross-batch running top-512 staging pairs.
        this->Input("running_strip").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // [numTokens, 1, topkTokens / kpool] raw pool ids, -1 padded.
        this->Output("indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Attr("topk_tokens").AttrType(REQUIRED).Int();
        this->Attr("kpool").AttrType(REQUIRED).Int();
        this->Attr("head_dim").AttrType(REQUIRED).Int();
        this->Attr("max_pool_seq_len").AttrType(REQUIRED).Int();
        this->Attr("split_batch").AttrType(REQUIRED).Int();

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
OP_ADD(Glm5KpoolSplitAiv);
} // namespace ops
