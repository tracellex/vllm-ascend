# HANDOVER — Glm5KpoolIndexer 稳定性修复（挂死/fault）

> 交接目标：**修复概率性挂死**。精度与性能已就绪，稳定性是端到端上线前的唯一阻断项。
> 日期：2026-09-26。前序上下文见本文末尾"已完成资产"。

## 1. 任务定义

**问题**：`Glm5KpoolIndexer`（arch22 MIX_AIC_1_2 混合核）在正常 launch 下概率性挂死。
**验收标准**：`smoke_compare.py` 全部 case（tiny/single96/packed2/packed3/widescreen/big）连续 30 轮零挂零 fault，且 `rows_mismatch=0` 保持。

## 2. 环境与现场

- **构建容器**：host39（=192.168.32.39，本机）`glmk-build`，镜像 `quay.io/ascend/vllm-ascend:glm53-a3-41run`，挂 `/dev/davinci0`
- **worktree**：`/data1/ascend/myascend/code/vllm-ascend-wt/glm5-kpool-indexer`（分支 `feat/glm5-kpool-ascendc-indexer`，基线 commit `d2666a16f`）
- **CANN 9.1.0 / driver 26.0.rc2 / ascend910_9391(A3)**
- 跨机（192.168.32.40）跨卡复现，挂率一致 → 非单机硬件

**构建**（必看坑）：
```bash
# 容器内 root，改 kernel 后必须三处清缓存（漏一处=打出旧二进制）：
rm -rf csrc/build/binary/ascend910_93/{src,bin}/glm5_kpool_indexer
rm -f  csrc/build/binary/ascend910_93/gen/glm5_kpool_indexer*.done \
       csrc/build/binary/ascend910_93/gen/Glm5KpoolIndexer-*glm5_kpool_indexer*.sh \
       csrc/build/binary/ascend910_93/gen/Glm5KpoolIndexer_*_param.json
# host tiling 改动还要：
find csrc/build -name 'glm5_kpool_indexer_tiling.cpp.o' -delete
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=ascend910_9391 bash csrc/build_aclnn.sh /opt/src/vllm-ascend ascend910_9391
```
kernel-only 改动**不需要**重跑 setup.py（kernel 走 OPP vendors 运行时加载）。

**跑对拍**：
```bash
source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
PYTHONPATH=/opt/src/vllm-ascend python3 csrc/attention/glm5_kpool_indexer/smoke_compare.py \
    --cases tiny single96 packed3    # 每轮 timeout 90s；rc=124=挂, rc=1=fault, rc=0=PASS
```

## 3. 现象精确描述

- **挂形态（B 型）**：host 卡在 `torch.npu.synchronize()`（faulthandler 证实），kernel 不完成，**无任何 runtime 报错**（device 级 sync 无超时）
- **fault 形态（A 型）**：~100s 后 `aicore timeout` 507014/507015；或直接 `EZ9999 ... The address for the scalar to access the internal buffer of AICore is out of bounds`（core id 2 = AIC2，`mte error info: 0x2b72514bcd`、`fixp_error0: 0x2514bcd` 恒定）
- **挂率随 lockstep 循环数连续增长**：tiny(2 units) ≈1/4、≥3 units ≈1/3、bench 连续第 5 次调用挂
- 完成的轮次**精度全部正确**（rows_mismatch=0，性能 362us@mid / 2.13ms@long）

## 4. 已排除项（勿重走！每项都有 8 连测统计）

| 排除项 | 证据 |
|---|---|
| FIA flag 信令全族 | mode2 id0 / per-pair id / mode0 / prime 记账修正，挂率不变 |
| 确定性信令 | **GM progress counter 握手**（aaafdfe84，workspace 尾部 per-pair 计数器+SyncAll 初始化），挂率不变 |
| kernel 工作量 | **空 kernel**（AIC 跳 MM1、AIV 跳 Fold/Emit）同挂率 37% |
| 核比/task 布局 | MIX_AIC_1_1（+blockDim 公式+行归属改造）3/8 |
| AIV GM 读对齐性 | gather8 布局 + 128B pad + uint32 读 + Init 预读 + 硬编码元数据（预读无辜） |
| 硬件/单机 | 卡 0/卡 1、host39/host40 同率 |
| ASCEND_LAUNCH_BLOCKING | 挂 2/8 |
| triton 参考路径 | 16/16 干净（同输入同环境） |
| **平台缺陷论** | **lightning_indexer（同构建树 vendored、同 handshake、同 MIX_AIC_1_2、AIV 上非对齐 uint32 GetValue）8/8 稳定** —— 问题是我们的 kernel 特有 |

msprof 注入会大幅降低小 shape 挂率（18/18）但大 shape 连续调用仍挂——时序敏感，非根治。

## 5. 关键资产

| 资产 | 位置 | 用途 |
|---|---|---|
| **确定性 fault 复现器** | 分支 `fault-repro`（commit `3b5d5aa88`） | 3+ units **100% fault** @AIC2，错误寄存器每轮一字不差——反汇编定位的最佳载体。**注意**：该变体把 2-unit 也弄 fault（idle 核预读也炸），仅作复现器，非候选修复 |
| 挂死复现+证据摘要 | `csrc/attention/glm5_kpool_indexer/repro_hang.sh` | 自包含，可作华为工单附件 |
| 对拍框架 | `smoke_compare.py`（13 cases） | triton vs ascendc 逐行集合比较 |
| 性能基准 | `perf_bench.py`（`--only` 单调用重试模式） | M3 数据见 commit 4604048a4 |
| 探针 | `probe_lost_task.py`（证核真挂非 task 丢失）、`triton_only_test.py`（16/16）、`lightning_hang_probe.py`（对照 8/8） | |

## 6. 未验证假说与建议路线（按性价比排序）

1. **mStart=64 的 16KB 偏移边界**（AIC0/1 的 qbar 偏移 0/8KB 不炸、AIC2 的 16KB 炸）——`QueryNd2Nz` 的 src 偏移寻址缺陷假说。**一次实验可证**：python 侧 qbar 头部 pad 32 行零（kernel mStart 整体 +32，输出行号同步 +32）；干净对照 single64（2 units 应过）vs single128（4 units 含 mStart=64 应挂）。注意 smoke 里这两个 case 在基线上不存在（回退时丢了），要重新加
2. **反汇编定位 fault pc**：fault-repro 二进制 `.o` 用 `ccec_compiler/bin/llvm-objdump -d`（符号 `_27_mix_aic` 是 bf16 运行变体）；加载基址未确定（pc start 两轮差 0x110），可对两个 fault 版本的 .o 做 diff 求地址平移来锚定
3. **逐段替换法**：把我们的 kernel.h 各函数逐个替换成 lightning 同款（lightning 源在 `csrc/attention/lightning_indexer/`，同目录结构可 diff），二分到具体函数
4. **华为工单**：材料齐（repro_hang.sh + fault-repro + 恒定错误寄存器 + lightning 对照）。若走此路，端到端等修复

## 7. 已完成资产（勿重做）

- **精度**：全对齐（gather8 修复 AIV GM 读 fault、TPipe VECOUT 修 EmitRow 竞态、python 广播 bug）
- **性能**：mid 362us vs triton ~1ms（2.6-3.3×）；**long 2.13ms vs 11.3ms（5.3×）**；wide 5.4ms
- **端到端接线**（挂死解决当天即通）：模型侧 `vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py:130` 已调 dispatcher，签名零改动；`VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL=ascendc` 切换；部署需同步 `vllm_ascend/_cann_ops_custom/vendors` 到挂载树/镜像
- 提交链：`0a5acf11d`→`330d9dae8`→`6cdd97c6f`→`aaafdfe84`→`0284ced7f`→`4604048a4`→`3b5d5aa88`(fault-repro)→`d2666a16f`(基线)

## 8. 纪律提醒

- 遵守仓内 AGENTS.md（14 步推理序、负结果留档）
- 修改用前台文件工具；长操作（构建/连测）Job 化落 rc 文件轮询
- 每轮实验记录挂率样本数（n≥8 才有区分力；1/3 vs 0 需 n≥12）
- 容器内 `pkill -f` 会误杀 herd exec 自身（命令行匹配），用 `pgrep -f "^python3 csrc"` 行首锚定
- 挂死后设备偶发 `507033 Failed to start device`——等 1-2 分钟自愈，勿急重启
