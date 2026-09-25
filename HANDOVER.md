# HANDOVER — Glm5KpoolIndexer（GLM5 lightning indexer AscendC 替换）

> 交接对象：codex（或任何接手者）。写于 2026-09-25。前置计划：
> `/home/ascend/.claude/plans/glm5-3-flash-profiling-glm5-next-lightn-giggly-gosling.md`
> （含完整背景/里程碑/验证口径，先读它）。

## 任务一句话

用 AscendC 算子 `Glm5KpoolIndexer`（fused 打分+topk+展开+tail）替换 glm5.3-flash 的
triton 版 `_glm5_next_lightning_indexer_score_kernel`（P 侧 prefill 占 49.6%，冷 TTFT 主犯）。
**用户明确要求：性能必须显著优于 triton 版（triton 特别慢），同时精度对齐。**

## 现场布局

| 项 | 值 |
|---|---|
| 源仓 | `~/repos/vllm-ascend`（clone tracellex/vllm-ascend，branch `glm5.3-flash-0.29` = 157fc66661，与挂载树零差异） |
| worktree | `/data1/ascend/myascend/code/vllm-ascend-wt/glm5-kpool-indexer`（branch `feat/glm5-kpool-ascendc-indexer`；git 身份已配仓内 xiangyongzh <ascend-operator@localhost>；third_party 已 rsync） |
| 构建容器（本机 .39） | `sudo docker exec glmk-build`（镜像 quay.io/ascend/vllm-ascend:glm53-a3-41run；设备 /dev/davinci0+manager+devmm_svm+hisi_hdc 全挂；-v worktree:/opt/src/vllm-ascend -v /home/ascend/repos:/home/ascend/repos） |
| 构建命令 | 容器内 `source /usr/local/Ascend/ascend-toolkit/set_env.sh; cd /opt/src/vllm-ascend && SOC_VERSION=ascend910_9391 bash csrc/build_aclnn.sh /opt/src/vllm-ascend ascend910_9391`（改 kernel 后必须先容器 root 清 `csrc/build/binary/ascend910_93/{src,bin,gen}/*glm5*`，否则 ninja 用旧拷贝） |
| 冒烟脚本 | 容器 `/tmp/smoke_ac_only.py`（ascendc 单测）、`/tmp/smoke_m2.py`（对拍 triton；pos 每请求重置已修）；跑法 `ASCEND_LAUNCH_BLOCKING=1 timeout 90 python3 /tmp/smoke_ac_only.py` |
| 主 .so | `python3 setup.py build_ext --inplace`（M1 时已构建过一次完整 .so+OPP；torch_binding 已注册 npu_glm5_kpool_indexer，meta 同步） |

## 已完成 ✅

1. **M1 全链（commit 125869d1f + 后续修复）**：def/infershape/tiling/aclnn/torch_binding(+meta)/python 分发层
   （`vllm_ascend/ops/glm5_kpool_indexer.py`，env `VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL`=auto/triton/ascendc）+ stub kernel。
   冒烟通过：[8,1,2051] int32。`sparse_attn_indexer_kpool.py` 已改为经分发层调用。
2. **M2 arch22 实现（编译全过，未提交）**：`op_kernel/arch22/` 五件（common/vector/sort 辅助/service_cube/service_vector/kernel）
   + 入口 `#if (__CCE_AICORE__==310)→arch35 else arch22`（**实测 A3 的 __CCE_AICORE__≠310，A3 走 arch22**；arch35 目录是早期按错误假设写的草案，可删或留作 950 兼容）。
   设计：M_TILE=32（每 AIV 16 行，运行中 top-k 条带 16×512×2×4B=64KB 进 UB）；S2_TILE=128；
   AIC 锁步（vendored v1 arch22 模式 2 握手）Mmad→NZ2ND fixpipe 到 per-AIC mm1Res GM 双缓冲（workspace 仅 ~1.5MB）；
   AIV 行级 Sort32+MrgSort 滚动 top-512 → 标量展开 ×4 + causal tail（tail 语义对齐 append_causal_tail，幂等）。
   tiling workspace 公式、qbar/positions 32 行 pad（python 侧 `_compute_qbar`/`_pad_positions`）、tSize 从输出 shape 读（pad 行 kernel 跳过）均已落地。

## 当前卡点 🔴（唯一）：AIC 全链跑通，AIV 一进 ProcessVec 行循环就 hang

二分矩阵（每轮 = 清缓存重编 + timeout 60-90s 冒烟；BUILD 均 OK）：

| 组合 | 结果 |
|---|---|
| AIC 完整（Nd2Nz+Mmad+Fixp）+ AIV ProcessVec 整体早退 | ✅ 过（shape 正确） |
| AIC 只 Nd2Nz / +Mmad | ✅ |
| AIC +Fixp 固定 128 宽 | ✅ |
| AIC +Fixp 动态宽 16 对齐 | ✅（**曾以为 16 对齐修复了**，见下） |
| AIV 进程进 ProcessVec（任何变体：仅事件、无事件、仅 copy、fold 全开、有无 LoadRowMeta） | ❌ 全部 hang |

关键数据点：
- AIV 行循环 `for r in 16` 内**逐项拆空**（只剩 Wait/Set 事件对）仍挂；把事件也删掉（V 计算裸跑）仍挂；
  LoadRowMeta（GM 标量读×17 + InitSortOutBuf 写 64KB）关掉也挂。
- 即：**hang 与行循环内容无关，与 ProcessVec 的执行时长/路径本身相关**。
- 容器重启后依旧（非设备脏）；dmesg 无 aicore fault；plog 未找到。
- 之前"16 对齐修复"的结论存疑：那轮通过的组合是 AIV 早退版（AIC 单侧跑），并非 Fixp 动态宽+AIV 双侧。

## 已确认的机制知识（不要重踩）

1. `csrc/build_aclnn.sh` 硬编码算子清单（910b/910_93 两分支），新算子必须登记（已加 glm5_kpool_indexer）。
2. kernel 入口必须 int 模板 + include 自己的 `*_template_tiling_key.h` + 实现宏裸名调用，否则 binary json
   `kernelList[].tilingKey=0` → 运行时 `NnopbaseExecutorGetCoreTypeAndTaskRation failed`。当前 tilingKey=1/27 正常。
3. 构建容器必须挂全 NPU 设备（否则 torch import stdout 污染 cmake 前缀路径 + torch_npu 初始化失败）。
4. `csrc/build` 为容器 root 属主：清缓存必须 `docker exec` 内做，宿主 rm 静默失败。
5. 9.1 API：tiling 的 `GetInputShape→StorageShape`；`graph/defs.h` 不存在；`Duplicate` 3 参要求 count 32B 对齐
   （行级 mask 用 128bit 位掩码版，见 `Glm5KpoolVec` 与 service_vector 的 laneMask 用法）；`GatherMask` 的 rsvdCnt 是
   uint64_t&；`LocalTensor` 标量写用 SetValue；`Align` 与 AscendC 同名需限定。
6. v1 arch22 参考实现在挂载树 `code/vllm-ascend/csrc/attention/lightning_indexer/op_kernel/arch22/`（2247 行四件），
   其 service_vector 用 **TPipe queue（VECOUT）+ PING/PONG 双事件** 组织 AIV 流程——我们的 fold 用 TBuf+手动事件简化，
   是当前 hang 的头号嫌疑结构差异。

## 下一步建议（优先级）

1. **定位 hang（建议三选一，先 a）**
   a. **照 v1 的 queue/pingpong 结构重构 `Glm5KpoolServiceVector`**（TQue<VECOUT> + 双份 score/tmp UB 乒乓 +
      仅 3 个事件），这是与 v1 已验证结构的最大差异，很可能绕开死锁；
   b. 心跳法：output_mode=1 的 scores_debug 输出当心跳区，AIV 在 ProcessVec 各阶段写标记，宿主后台线程定时
      `tensor.copy_` 抓快照定位卡点（需绕过 ASCEND_LAUNCH_BLOCKING 的同步挂）；
   c. ccec 单算子 dump（`csrc/build.sh` + ASCEND_OP_NAME 或 msdebug）看 AIV 核是否真死等事件。
2. **hang 解开后**：跑 `/tmp/smoke_m2.py` 对拍（sort 后集合精确 + tail/-1 逐元素）；若 Sort/Mrg 精度 tied 引起集合漂移，
   放宽为 top-256 精确+其余近似需回报（用户强调精度对齐）。
3. **性能（用户强调）**：当前锁步+标量展开肯定不达标。M3 清单：锁步→双缓冲流水（AIC 领先 2 块）；行批 Sort
   （64 行一次 Sort repeat）；展开向量化（4 次 strided Adds 或 Gather）；`output_mode=1` 分数对拍 triton。
   bench 口径：T=8192×128k 上下文，目标 ≤12ms/launch（现状 triton 33.6ms）。
4. **留档**：memory 已有 `glmk-indexer-ascendc-dev.md` / `vllm-ascend-csrc-build-gotchas.md`，里程碑更新随手做；
   M2 编译通过态先 commit（现在未提交，防丢）。

## 纪律提醒

- 主挂载树 `code/vllm-ascend` 零构建产物不动；一切改动在 worktree。
- 完成节点双轨留档（docs/glm-5.3-flash/model-profile.md + hwnpu MCP 项目 4）。
- AGENTS.md 规范：conventional commits + signoff；env 变量集中 envs.py。
