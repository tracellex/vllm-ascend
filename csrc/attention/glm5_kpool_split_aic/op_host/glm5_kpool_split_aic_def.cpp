#include <cstdint>
#include "register/op_def.h"
#include "register/op_def_registry.h"

namespace ops {
class Glm5KpoolSplitAic : public OpDef {
public:
    explicit Glm5KpoolSplitAic(const char *name) : OpDef(name)
    {
        this->Input("qbar").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("indexer_cache").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cum_query_lens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("indexer_seq_lens").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("indexer_block_table").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("positions").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        // [T_pad, SPLIT_BATCH_POOLS] fp32 scores of this batch (row-major,
        // caller-owned tensor passed on to the AIV op).
        this->Output("scores").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
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
OP_ADD(Glm5KpoolSplitAic);
} // namespace ops
