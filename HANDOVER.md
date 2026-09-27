# HANDOVER — Glm5KpoolIndexer M6/M7 收官(全部闭环)

> 状态:2026-09-27,H7/H8/H9 + 排序链归并优化 + M7 端到端全部完成。
> worktree HEAD e3b08ad8c(625315aee=H7/H8/H9,e3b08ad8c=merge-fold)。
> 端到端:gsm8k-20 正式口径 **95.00**;TTFT bs2-64k **15.19s** / 192k
> **71.08s**(基线 15.33/72.29)。研究档案见主仓
> `research/active/glm5-kpool-indexer/{experiments,hypotheses}.md`。

## 最终形态(k=256 packed Mmad)

- 契约:qbar = `[T, 2*head_dim]`,行内打包 `[q_hi | q_lo]` BF16 半部
  (FP32 头加权 query 的双 BF16 分解);tiling/infershape 强校验宽度 256。
- Cube:单条 k=256 Mmad(`[q_hi|q_lo] x [K|K]^T == q_hi@K + q_lo@K`),
  B 侧每 cache 块 Nd2Nz 双写(+0/+KEY_K_BLOCK)构成 [K|K];L0B 单 64KB 槽
  每 tile 重写(1-lag M_MTE1 守卫);L0A 恢复奇偶双槽;单 C 单 Fixp。
- mm1Res 四缓冲(%4,AIC 领先≤2 的 unit 边界竞态窗口由此关闭);
  FIX_M 握手(arch35 模式)防 Mmad 清零与滞后 fixpipe 读同 L0C 槽竞态。
- AIV 折叠回到单条 DataCopyPad——无向量加法(该 Add→Sort 可见性危害
  曾致真实语料 rank-511/512 边界互换,详见 experiments.md round 4)。

## 验收(构建 glmk-m6t-k256-185839Z,hash 460ec011…/bb1ea799…,build=install)

probe 60/60(含 hi-only/lo-only/split-half/double 判别)、smoke 9/9
(contend4k 0 非 tie)、精度探针 2045/2045 跟随 FP32(对 BF16 仅 1609)、
replay 414 calls(43 万 prefill 行)全 PASS、acceptance 9×30 PASS。
tie 判据 TIE_TOL=2e-3(见 smoke_compare.py 注释)。

## 性能(wall,perf_bench)

mid 1.63× / long 4.29× / wide 3.73× vs Triton;较 M3 单半部基线
long +25% / wide +67%(B 侧 MTE2 双写)。后续可优化项:L1 内复制替代
第二次 GM Nd2Nz 读。

## M7 待办(012 配方)

herd apply(最后一次)→ 挂载树手工同步(dispatcher python +
vllm_ascend_C.so(本轮 torch_adpt 有改动,必须用新 .so)+
_cann_ops_custom/vendors)→ docker restart P → ttft-bs2-192k A/B +
acc-gsm8k-20。坑:provision 重置挂载树;docker exec 不继承 PYTHONPATH;
aisbench warmup 前缀命中(N≥5)。

## 构建/测试硬约束(不变)

改 .h 后 `rm -rf csrc/build` 全量重编并核对 build/install 双侧 .o hash;
`python3 -c` 一次性驱动在本栈会产生假挂死——一律文件模式;长任务 Job 化
落 `.herd/out/`;测试脚本 `.herd-test-m6l.sh <jobdir>`。
