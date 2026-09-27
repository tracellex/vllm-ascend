# HANDOVER — Glm5KpoolIndexer 性能优化战役(方案 C')交接 codex

> 日期:2026-09-27 深夜。前史(M1-M7 精度闭环 + 两轮性能翻案)见主仓
> `research/active/glm5-kpool-indexer/{experiments,lessons-ascendc,hypotheses}.md`。
> 当前任务:**实现方案 C'(混合架构),把真实形状性能从 3.2× 慢于 triton
> 做到 1.7-2.5× 快于 triton**,精度门不许降。

## 1. 任务:方案 C'(数学推导见 experiments.md "F0 分侧计时" 节)

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

- worktree:本目录,分支 feat/glm5-kpool-ascendc-indexer @ 83ae040ba(已推
  origin)。**工作树当前干净**(F0a 临时 gate 已还原为 4)。
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

## 5. 第一步建议

1. 读 experiments.md 末三节(修正案 + F0)与 lessons-ascendc.md。
2. 在 worktree 新分支(如 feat/glmk-hybrid-topk)开工 C':先做 M_TILE=128
   + Fixp strided 直写 + AIV 清空(保留 handshake),wrapper 侧补 topk 路径。
3. 全套验收门 → serving_shape_bench 达标 → overlay 部署 → 端到端 A/B。
4. 所有负结果与轮次照旧写 experiments.md(假设→变更→结果→结论)。
