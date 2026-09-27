# HANDOVER — Glm5KpoolIndexer 性能优化战役交接

> 日期:2026-09-27 深夜。前史(M1-M7 精度闭环 + 两轮性能翻案)见主仓
> `research/active/glm5-kpool-indexer/{experiments,lessons-ascendc,hypotheses}.md`。
> **状态更正 (2026-09-27)**：下方 C' 设计已被后续 F1 分组 TopK 取代，
> 不可按 C' 步骤继续实现或部署。C' 的完整 `[T,P]` FP32 score 矩阵路径因
> 大规模临时写流量/内存成本被用户否决。当前代码实现的是 F1：保留
> `M_TILE=32,S2_TILE=128`，每 1024 pools 做设备侧分组 TopK，不生成完整
> score 矩阵。CANN 9.1 全量构建已通过，但 39 serving 容器在 tiling 阶段以
> `group-topk mode requires A3` 拒绝；**F1 的正确性与性能均未验证，不得宣称
> 达标或部署**。实验细节和当前验证门见主仓
> `research/active/glm5-kpool-indexer/experiments.md` 的 F1 段及 `hypotheses.md` H10。
> 后续 H15 M64 只证明隔离 CANN full-target 编译通过；无 UB/L0 资源报告，未做设备正确性或延迟验证。
> **终章 (2026-09-27 晚)**:用户授权 .39 隔离窗口后,F1+H11-15+零池修复在
> glmk-build(本机 .39)完成:**正确性门 100% 全绿**(probe/smoke 11 cases x
> 3 impl/qbar/replay414/acceptance x2,含 M32/M64 首次设备验证);microbench
> M32 2.4x/M64 3.6x vs triton;引擎内 profile kernel 对照快 ~50x。**但引擎
> 端到端不可用**:ascendc dispatcher 路径存在随使用持续劣化的病理(64k TTFT
> 34->65s+,triton 长跑稳定 15.7s),已按止损纪律回滚——029 现役 = triton
> 默认,gsm8k 95.00 复验通过。C' 早已撤销;关键破案:group-topk SoC guard 是
> 人为闸门且被枚举映射误挡(.39/.40 同为 A3=IT22HMDA_2_S,但 CANN
> GetSocVersion 对 die ascend910_9391 报 ASCEND910B,guard 按 ==910_93
> 精确匹配误拒;已放行 {910B, 910_93},全部验证均在 A3 完成);隔离 /tmp 树"编译通过"
> 多为 build/src 副本陈旧假阳性(逐文件 sha256 核对才算);41run 镜像固化了
> M7 旧 glmk 算子(serve.sh env=ascendc 会踩中,P serve.sh 现已改 triton)。
> 续战方向:修 dispatcher 热态病理(fp32 qbar 临时张量/allocator 嫌疑),
> kernel 侧无需再优化。全过程见主仓 experiments.md 2026-09-27 各节。
> 机器访问范围经用户明确限定为 `.39/.40`；不得访问其它主机。两台服务机上的
> `.40` 显示 A3 镜像/板卡但 16 个 NPU worker 在线、HBM 约 88% 占用；benchmark
> 须先确认 runtime SoC 兼容并获得隔离窗口，不能与在线推理竞争资源。

> 本战役验收目标仍是精度不降，并以正确性通过后同形状 Triton 对照的可复现
> kernel 延迟为准；此前 C' 的 1.7-2.5x 只是旧预测，不是当前目标或结果。

## 1. 旧方案 C' (已撤销,仅保留历史)

以下设计只用于理解历史决策，禁止继续实现：

现状问题(全部已实证):真实形状(T=7680/调用,pools=ctx/4≤48k)下 ascendc
159.6ms@16k pools vs triton 50.8ms。F0 分侧:AIC+锁步占 70%(110.9ms@16k),
根因 **M_TILE=32 的小 Mmad(利用率 15-24%)+ 每 tile 串行流水**;AIV 折叠占
30%;选择若用 aclnn TopKV2 每元素成本是 VEC 折叠的 1/4000。

**C' 设计**:
1. **删除 AIV 折叠链**(FoldBlockIntoRowTopk/SortAll/merge/emit 全链),
   AIV 仅保留 handshake 预放行(锁步退化为 AIC 自由深流水;可先让 AIV 每
   tile 只 SetFlag 放行)。
2. **M_TILE 32→128**(common.h;AIV 条带约束消失后 cube 效率 ×~4;注意
   mm1Res/四缓冲/FIX_M/1-lag 守卫的尺寸推导全部随 M_TILE 走,已参数化)。
3. **Fixp 直写 [T,P] 分数矩阵**:新 op 输出 scores [T,maxPool] fp32(复用
   scores_debug 的通道形态),dstOffset = mStart·P + s2Start,dstStride = P
   (现值 = actS2SizeAlign=128;uint16 上限 → P≤65535,≤256k ctx,tiling 加
   assert)。mm1Res 工作区可删(或留作 AIC 内部缓冲)。
4. **wrapper 侧选择**(全部现成,抄 triton 分支):scores[t, visible:]=-inf
   掩码 → torch.topk(NEG_INF_SENTINEL 防 TopKV2 -inf fault,已有)→ 按 T
   分片控内存(TRITON_SCORES_CHUNK_BYTES 惯例)→ _expand_pool_ids + tail。
5. 可选第二步:S2_TILE 128→256(先不做,单变量)。

**验收门(顺序固定)**:probe_block_table(60/60)→ smoke 9/9 → 精度探针
(2045/2045 跟 FP32)→ replay 414 全 pid → acceptance 9×30 →
serving_shape_bench(**必须用修好的 pool_lens 版本**)→ 引擎 profile 确认
Glm5KpoolIndexer avg < triton 同形状 → overlay 六工件部署 + aisbench
ttft-64k/192k + gsm8k-20(对照:triton 15.89 / 72.3 / 95.00)。

**止损判据**:C' 落地后引擎内仍 ≥1.2× 慢于 triton → 定性"此硬件此路径无
胜区",归档,默认永久 triton(现役已如此)。

## 2. 现场与资产

- worktree:本目录,分支 feat/glm5-kpool-ascendc-indexer @
  `fdbaadaee2653745b337900d96f93c7fbd2ef3bd`。该提交是接手基线；当前 worktree
  有未提交的 kernel 与验证脚本改动，不能按干净树处理或重置。
- 构建容器:host .39 `glmk-build`(Up;设备挂全,worktree 挂 /opt/src/vllm-ascend)。
  构建:`bash .herd-build-m6l.sh <job>`(含 rm -rf csrc/build);全套测试:
  `bash .herd-test-m6l.sh <jobdir>`;job 一律落 `.herd/out/`。
- 真形状 bench:`csrc/attention/glm5_kpool_indexer/serving_shape_bench.py`
  (pool_lens 已解耦,勿再用旧版)。语料:replay 414 份
  (run/stacks/glm53flash-029-pd/tmp/glmk-replay)。
- 基线数字(triton,T=7680,pools 2k/8k/16k/32k/48k):
  11.9 / 28.7 / 50.8 / 95.3 / 140.0 ms;ascendc 现状:22.6 / 82.8 / 159.6 /
  323.8 / 473.1;AIC 地板(F0a):17.6 / — / 110.9 / — / 325.2。
- serving 栈:029-PD = 原 bundle b74335ca2f0b + overlay 六工件(dispatcher
  py / envs.py / 模型接线 sparse_attn_indexer_kpool.py / vllm_ascend_C.so /
  libvllm_ascend_kernels.so / vendors),**dispatcher 默认=triton**(性能
  保护;env VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL=ascendc 切换)。
  glm52-pd-v2-d-39/-40 已停(docker start 可还原;029 起栈前必须停之,
  否则 HBM 不足)。

## 3. 硬约束(违反任何一条 = 白干或事故)

1. **构建**:改任何 .h 后必须 rm -rf csrc/build 全量重编,核对 build 与
   vendors 双侧 .o SHA256(mtime 会说谎)。
2. **测试**:`python3 -c` 一次性驱动在本栈**假挂死**,一律文件模式驱动。
3. **部署**:重编 bundle 再 apply = 当日快照重置挂载树(gsm8k 实测崩到
   5-10)。**只允许:原 bundle apply 一次 → overlay 六工件 → docker
   restart P;此后不许再 apply**。provision 后 chown root 残留;mooncake
   TransferEngine 与 restart 有竞态(隔时二启即愈)。
4. **c220 微架构负结果库**(lessons-ascendc.md §二,全须遵守):
   同区累加 Mmad 毗邻 fixpipe 死锁;V-ALU 写→Sort 读可见性危害(PipeBarrier
   不栅栏);mm1Res 深度须 >AIC 领先;Mmad init 清零 vs 滞后 fixpipe 需
   FIX_M;B32 repeat=64 lane;AIV GM 标量读 32B 对齐。
5. **精度红线**:TIE_TOL=2e-3 判据与 replay 414 是门;不许为性能降参考精度。
6. **事件账目**:Set/Wait 逐事件 1:1 手推;teardown 只 drain 未消费 flag。

## 4. 已知坑速查

- overlay 漏同步 = 静默跑 triton(破案法:profile op_statistic 找
  `_glm5_next_lightning_indexer_score_kernel`);herd profile 与 herd test
  同栈互斥(排他租约),profile 自带负载用 curl 驱动。
- TTFT 正式口径 = aisbench N≥4 取 clean-3 median;单发 curl 冷态会虚高 2×。
- 引擎内真实调用形状看 profile kernel_details.csv 的 Input Shapes 列。
- wrapper 的 pool_lens 陷阱:引擎 seq_lens=ctx/4,与 T 无关(64k ctx→16k
  pools,192k→48k)。

## 5. 当前续作建议 (2026-09-27)

1. C' 已撤销，不得按旧步骤恢复 `[T,P]` 全量分数矩阵方案。
2. 本地候选 H11-H13：group running strip 由 1024 pair 缩至 512 pair；
   `SortFull1024` 只复制 top-512 前缀；两个 group 512+512 MrgSort 启用
   `ifExhaustedSuspension`。2026-09-27 已从排除 `build/`/`__pycache__/` 的精确
   worktree 快照，在 `/tmp/glmk-csrc-source-20260927/build` 用 CANN 9.1 完成
   `ascend910_93` 完整单算子构建，rc=0；host tiling registration 与两个 kernel
   tiling-key 变体均通过编译，`.o`/JSON SHA256 记于主仓
   `research/active/glm5-kpool-indexer/experiments.md`。仅编译通过，NPU 正确性、
   runtime 和性能仍未验证；不安装或部署。不得 chmod/删除 worktree 的
   `csrc/build`，后续硬件验证等待重新授权的兼容 A3 隔离窗口。
   H13 的 suspension 在输入队列耗尽时停止，不是输出 512 项上限；最多可输出
   1023 对，收益依赖分数分布，不能静态宣称减半。H14 co-rank 有界归并仅为假设，
   tie/sentinel 语义与 UB 随机读取成本未闭合，见主仓 `hypotheses.md`。i.i.d.
   模型预计 H13 的 Stage C 停在约 1022 对，而旧 top-512 与新组的 late running
   merge 在 n=15 时约停在 546 对；这些是待证伪的分布估算，不是测量结果。
3. 当前用户要求停止 39/40 并释放资源；不得访问、重启或部署到这两台机器，
   也不得改用其他主机。硬件验证需等待用户重新授权。
4. 本地 CANN 隔离构建已完成，build 侧 `.o`/JSON SHA256 已记档；下一步需在
   获准且兼容的 A3 隔离窗口执行 probe、smoke、精度探针、replay414、30 轮验收。
   正确性全过后，再做同形状 Triton/AscendC bench 与 timeline；vendor 侧安装哈希
   仅在获准部署验证时采集，当前不得安装或部署。
5. Triton 保持默认；未取得可复现的同形状端到端核延迟优于 Triton 的证据前，
   性能状态仍为未达标。轮次和负结果记入主仓 `research/active/glm5-kpool-indexer/`。
6. 2026-09-27 新增 H15 opt-in 候选：`ascendc_group_topk_m64` / outputMode=4，
   仅 group-topk 使用 M_TILE=64；outputMode=3 的 M_TILE=32 和默认 Triton 不变。
   公式预测总 FLOPs 不变、每 token tile 的 key 复用加倍、Mmad 与 group
   handshake 数减半。AIV UB 估算约 162.25 KiB/184 KiB，L0A 双缓冲达到 64 KiB
   上限。已于 2026-09-27 在隔离 CANN 9.1 `ascend910_93` 完成 full-target
   编译，FP16/BF16 两组 `.o`/JSON SHA256 见主仓
   `research/active/glm5-kpool-indexer/experiments.md`。JSON `compileInfo` 为空，
   无 UB/L0 allocation 报告；因此仅证明 compile pass，不证明 resource fit。当前
   workspace 曾观察到 host IP 为 `.39` 且映射 `/dev/davinci0..15`，但未复核
   当前服务/设备使用状态；用户要求停止/释放 39/40，因此禁止初始化设备、访问
   容器或查询其运行状态。`.39` 历史 SoC guard 拒绝 group-topk，`.40` 曾繁忙。
   目前没有获准的兼容 A3 验证窗口。正确性与 paired profile/benchmark
   尚未执行。M64 保持 opt-in，Triton 保持默认，不得称性能达标。
7. 当前数学与机器状态的最终核对见主仓
   `research/active/glm5-kpool-indexer/experiments.md` 的
   `2026-09-27 handover state reconciliation`。特别注意：工作区曾观察到位于 `.39`
   且映射 NPU 设备节点，但本轮未复核服务/设备使用状态；用户要求停止/释放 39/40，
   因此不得初始化设备、访问容器或查询其运行状态。
   优化方向必须由 AIC/AIV timeline 和资源报告选择，不能只凭 FLOPs 或
   handshake 数推断提速。
8. **2026-09-27 Claude continuation**:本轮新增 arch22 `s2Num==0` 输出修复：
   AIV 对有效 query 行将 `[poolTopk=512]` raw pool IDs 写为 `-1`，不增加
   CrossCore event；CPU helper suite 6/6 通过。隔离 CANN 9.1
   `ascend910_93` full target 已重编成功，FP16/BF16 `.o`/JSON hashes 记于主仓
   `research/active/glm5-kpool-indexer/experiments.md`。JSON `compileInfo` 为空，
   只算 compile pass；设备正确性、event runtime 和性能均未验证。常规 pytest
   因缺 `vllm` 无法收集。未访问设备/远端；39/40 停止释放约束继续有效，禁止
   查询、启动或改用其他主机。Claude 接手先审查 `ProcessEmptyUnit` 对 AIV
   覆盖范围/arch22 非 group mode 的输出契约，再在获准 A3 窗口测混合 batch
   零 pool + 非空 request，确认 raw IDs 和 event 配对；之后按正确性门完成 replay
   与同形状 Triton 对照。当前 Triton 默认，AscendC 未达性能验收。工作树本身
   仍有大量既存用户改动，禁止 reset/clean；本轮没有提交。
