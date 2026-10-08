# SPDX-License-Identifier: Apache-2.0
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
    glm5_kpool_split_aiv_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME Glm5KpoolSplitAiv
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE glm5_kpool_split_aiv ACLNNTYPE aclnn_inner)
endif()
