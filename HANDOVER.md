# HANDOVER — Glm5KpoolIndexer 方案2(去 mix 拆分)交接

> **终局更新 (2026-09-29 晚)**:下方断点描述已过时——fixpipe 崩溃与全部构建
> 问题已在 build20 解决,**正确性门 100% 全绿,引擎 A/B 完胜**:64k TTFT
> 9.79s vs triton 15.67(-37.5%)、192k 28.11 vs 68.61(-59%)、gsm8k 无差、
> 12 连发无劣化。现场已回滚 triton 交付态。完整总结/切换指南/踩坑全录见主仓
> `research/active/glm5-kpool-indexer/FINAL-REPORT.md`;本文件余下内容保留为
> codex 时代的过程交接存档。
>
> 日期:2026-09-29。接收方:codex。上一版 HANDOVER(M1-M7 战役史)已并入主仓
> `research/active/glm5-kpool-indexer/experiments.md`,本文件只覆盖方案 2。
> **一句话断点**:split 双算子代码全部落地、15 轮构建链打通、aclnn 公共壳
> 符号已导出、算子可被真实调度;**但 AIC kernel 单独执行即触发 fixpipe
> MTE 非法地址(aicore error 0x800000)**,正确性门一步未过。需先定位并修掉
> 该 kernel 侧崩溃,再走验证门与引擎 A/B。

## 0. 机器与现场(只允许 .39/.40)

- `.39` = P 侧(prefiller,029-PD 栈 TP8DP2);`.40` = D 侧 + **glmk-build 容器**
  (构建 + NPU 测试都在这里,worktree 挂载为容器内 `/opt/src/vllm-ascend`)。
- 两台均为 A3(板卡 IT22HMDA_2_S,die ascend910_9391);CANN `GetSocVersion()`
  对它报 ASCEND910B 枚举(已知映射坑,tiling guard 已放行 {910B, 910_93})。
- 构建 SOC = `ascend910_93`(`build_aclnn.sh` 走 910_93 分支的 CUSTOM_OPS 清单)。
- **herd serve 已停勿启**(PID 3040191/3761823 已杀):reconcile 会删漂移容器、
  洗掉 overlay。现场操作只用 `docker restart`(无 provision)。
- P 侧 029 容器现役 `VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL=triton`(serve.sh
  第 27 行,文件在挂载 bundle `1639a40ac2c2.../hosts/prefiller-0/serve.sh`);
  旧 overlay(25 文件,docker diff 核对过的最小集)仍在容器内,env=triton 不触发。
- docker 需 `sudo -n docker`(ascend 用户不在 docker 组)。
- git:本 worktree(`code/vllm-ascend-wt/glm5-kpool-indexer`,分支 main)有
  **未提交改动**(见 §5 文件清单的"本轮新改");主仓 = `/data1/ascend/myascend`。
  commit 流程:commit → `git pull --rebase origin main` → push。

## 1. 背景与目标(为什么拆)

- F1 战役结论(2026-09-27):fused AscendC 算子正确性门 100% 全绿、kernel 级
  ~50×(引擎内 0.199s vs triton 10.47s),**但引擎端到端反而慢 33%**(64k TTFT
  triton 15.20-15.78s vs ascendc 20.3-20.8s),已回滚 triton 现役。
- 根因(2026-09-28 深挖轮实锤):fused 算子是 `KERNEL_TYPE_MIX_AIC_1_2`
  (AIC+AIV 联动),执行期**整卡引擎互斥**——与 `HCCL_OP_EXPANSION_MODE=AIV`
  的 TP8 集合通信 kernel(hcom_allGather/reduceScatter)完全互斥,26 次 launch
  实测重叠仅 0.24ms,设备 idle 63%。方案 1(通信切 AIC)验证 4h 卡死不可行已关闭。
- **方案 2 = 把 mix 拆成两个 pure kernel**:AIC_ONLY 打分 + AIV_ONLY 折叠,
  wrapper 按批(4096 pools)交替 launch,恢复与通信 kernel 的块级共存调度
  (triton 路径的调度形态)。目标端到端 64k TTFT ≤ 15.2s(triton 持平或更好)。

## 2. 方案 2 设计契约(已全部落码)

per batch b(pools [b*4096, (b+1)*4096)),python wrapper 串行同 stream:
`AIC(b) → AIV(b) → AIC(b+1) → ...`;kernel 边界即全量屏障,零 cross-core flag。

- `Glm5KpoolSplitAic`(KERNEL_TYPE_AIC_ONLY,blockDim=aicNum,usedCoreNum=aicNum,
  outputMode=5):每 unit 对批窗口内每个 S2_TILE 做 mm1,fixpipe 写进
  caller 张量 scores[T_pad, 4096] fp32 的该 unit 行段(dstStride=4096,
  列偏移 = s2Start % 4096)。
- `Glm5KpoolSplitAiv`(KERNEL_TYPE_AIV_ONLY,blockDim=aivNum,aiCoreIdx=blockIdx/2,
  outputMode=6):读 scores + runningStrip[T_pad, 1024] fp32(wrapper 持久张量,
  跨批走 GM 不走 workspace),每 1024-pool 组 fold 进 per-row running top-512;
  mid 批 SaveRunningStrip / 末批 ProcessTopK emit;池数落在早期批的请求在
  "越过其池范围的第一个批"补一次 Restore+emit(batchIdx == ceil(P/4096));
  零池请求在全局末批 ProcessEmptyUnit 发 -1。
- tiling/infershape/def/公共壳/绑定详见 §5;attr 序(两算子同):
  `topk_tokens, kpool, head_dim, max_pool_seq_len, split_batch`(batch 号)。

## 3. 构建链与 ccec 约定(15 轮踩坑沉淀,codex 必读)

构建(全量,约 10min):
```bash
sudo -n docker exec glmk-build bash -c \
  "bash /opt/src/vllm-ascend/.herd-build-m6l.sh /opt/src/vllm-ascend/.herd/out/glmk-split-buildN-日期"
# 成功标志:build.log 尾部 "build_aclnn rc=0";"INST MISSING" 是脚本 glob 误报,忽略
# 产物核验(注意构建以 root 安装,先 sudo chown -R ascend:ascend vllm_ascend/_cann_ops_custom):
nm -D vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so \
  | grep aclnnGlm5KpoolSplit   # 应见 Aic/Aiv ×(本体+GetWorkspaceSize) 共 4 个 T 符号
find vllm_ascend/_cann_ops_custom -name "*Split*"  # op_impl 下应有 .o+.json
```
只有 torch adapter(csrc/torch_binding.cpp、*_torch_adpt.h)改动才需要重跑
`setup.py build_ext --inplace`;遇 `vllm_ascend_kernels_preprocess` obj merge
错,`rm -rf build/temp.*/vllm_ascend_kernels_preprocess-prefix` 后重编。

约定清单(违反即白板重来):
1. `build_aclnn.sh` CUSTOM_OPS 按 SOC 分支硬编码;目标 910_9391 走 **910_93**
   分支,`glm5_kpool_split_aic`、`glm5_kpool_split_aiv` 已插在第 158 行后。
2. kernel TU 必须 `KERNEL_TASK_TYPE_DEFAULT(...)` + `REGISTER_TILING_DEFAULT(Glm5KpoolTilingData)`
   + include **plain struct** tiling 头(op_kernel/common/,字段序与 op_host
   FIELD_DEF 版严格一致),另配 `*_template_tiling_key.h`(FP16=1/BF16=27)。
3. ccec 源副本不跨算子目录:依赖头必须 **vendor** 进各算子目录(两个 split
   目录各 vendored 了 arch22 5 头 + kernel_split.h;改 indexer 目录下原件后
   必须 cp 同步,sha256 校验三份一致)。
4. def 用 `OpAICoreConfig` 链式(910b/910_93/950 三配置);infershape 用
   `SetDimNum/SetDim` + `IMPL_OP_INFERSHAPE(OpName).InferShape(...).InferDataType(...)`;
   tiling 用 `IMPL_OP_OPTILING(OpName).Tiling(f).TilingParse<CI>(prep)`,
   平台信息 `fe::PlatFormInfos*` → `PlatformAscendC(platformInfo)`。
5. vendors 只导出 autogen `aclnnInner*`;**公共符号必须手写公共壳**
   `op_host/op_api/aclnn_glm5_kpool_split_{aic,aiv}.{h,cpp}`(extern "C"
   visibility(default) 转发 Inner 版;参数序 = def 的 Input+Attr+Output),
   且 op_host/CMakeLists.txt 要 `add_op_to_compiled_list` +
   `add_ops_compile_options OP_NAME` + `add_modules_sources(OPTYPE aclnn_inner)`,
   目录 CMakeLists glob 进来。
6. 测量坑:profile 观察本身制造假发病;压缩 prompt 假 64k;APC 互命中;
   重启热身窗(前几分钟 2-3× 慢);gsm8k-20 单题翻转 ±5 分。
7. `/tmp` 隔离树构建 = 副本陈旧假阳性,逐文件 sha256 对比才算数。

## 4. 当前故障:精确定位与已排除项

**现象**(build15,`smoke_split_bisect.py aic tiny`):AIC batch 0 单独执行、
同步点报 aicore 异常——`error code 0x800000,MTE accesses an invalid GM address`,
`core id 24`,`fixp_error0=0x30000d6, fixp_error1=0xf9`(build14 时 0x9),
mte error `0xf9030000d6`。tiny 用例:T=33 → qbar/positions_pad 已被 wrapper
pad 到 128 行,scores=[128,4096] 2MB,1 批,9 池 1 tile。**AIV 未执行**(二分
stage=aic 只跑 AIC)。设备详细 dump 在容器内 `~/ascend/log/`(plog)。

已排除:
- python 分配不足(初判 T=33 越界**不成立**:qbar 本就 128 对齐;但 defensive
  的 t_rows 对齐分配保留,无害)。
- `dstStride` 字段宽度(struct 里 uint32_t)。
- `dstStride` 单位误解:官方 `lightning_indexer_v2_service_cube.h:381` 与
  `quant_lightning_indexer_service_cube.h:561` 均按 **元素行距** 用
  (`s1gGmOffset * dstStride + s2GmOffset`),与我们的用法一致。
- 公共壳/注册/tiling/形状匹配(算子能被调度执行,错误在 kernel 运行期)。

**未验证主嫌疑**(按优先级):
1. **fixpipe dstStride 硬件字段上限**:4096(fp32 元素)恰好 2^12,若硬件
   SD_ST 字段 12bit 或有合法上限(如 ≤ nSize 的某倍数),会精确复现"地址
   非法"。实验:临时改 `glm5_kpool_indexer_service_cube.h` Fixp split 分支
   dstStride=1024(group 模式已证值)跑 bisect——不崩即坐实;随后把 scores
   条带改成 [T_pad,1024]×4 段(或 [T_pad, 4, 1024] view)按段写,规避上限。
2. **nSize×dstStride 单次 fixpipe 覆盖范围上限**(32 行 × 4096 列 = 512KB):
   官方未见超过 [m,1024] 的用法。实验:同上,缩条带宽即可一并验证。
3. **Mmad/LoadData2D 读路径在 AIC_ONLY 单独 launch 下的行为差异**(mix 下
   同 intrinsics 全绿):core 24 是末核,先在 tiling 的 OP_LOGI(容器 stdout
   或 plog)确认 aicNum 与 blockDim;若 aicNum=25,末核只分到尾巴 unit,
   崩在 fixp_error 更像写侧。
4. 兜底:ascend plog(容器 `~/root/ascend/log/device-*/*.log`)按 core 24
   过滤,看出错 PC 与源行(配合 .o 反汇编 addr2line)。

改 kernel 头后必须:cp 同步两份 vendor 副本(§3 第 3 条)→ 重跑构建 →
chown → bisect。

## 5. 文件清单(本轮方案 2 全量)

新目录(每算子 12 文件,结构对称):
```
csrc/attention/glm5_kpool_split_aic/    # aiv 目录同构
  op_kernel/glm5_kpool_split_aic.cpp            # stub entry(AIC_ONLY)
  op_kernel/common/glm5_kpool_split_tiling.h    # plain struct
  op_kernel/glm5_kpool_split_aic_template_tiling_key.h
  op_kernel/arch22/                             # vendored 5+1 头
  op_host/glm5_kpool_split_aic_{def,infershape,tiling}.cpp
  op_host/glm5_kpool_split_aic_tiling.h         # FIELD_DEF 版
  op_host/op_api/aclnn_glm5_kpool_split_aic.{h,cpp}  # 公共壳(build14 增量)
  op_host/CMakeLists.txt + 目录 CMakeLists.txt
```
改动:
- `csrc/attention/glm5_kpool_indexer/op_kernel/arch22/glm5_kpool_indexer_common.h`
  (split 常量/mode 5,6/ConstInfo.splitBatch/SPLIT_BATCH_POOLS=4096)
- `.../glm5_kpool_indexer_service_cube.h`(Fixp split 分支 + RelocateMm1Res)
- `.../glm5_kpool_indexer_service_vector.h`(split 读基址/strip save/restore/
  LoadRowMeta 挪 public)
- `.../glm5_kpool_indexer_kernel_split.h`(**新**,三份同步;含 2026-09-29 的
  补 emit 修复:batchIdx==ceil(P/4096) 时 Restore+ProcessTopK)
- `csrc/attention/glm5_kpool_indexer/glm5_kpool_indexer_torch_adpt.h`
  (两个 void EXEC_NPU_CMD 入口)
- `csrc/torch_binding.cpp`(ops.def + ops.impl,schema "-> ()")
- `vllm_ascend/ops/glm5_kpool_indexer.py`(split 分支:批循环直调 + 后处理;
  2026-09-29 加 t_rows 对齐分配)
- `csrc/attention/glm5_kpool_indexer/smoke_compare.py`(impl choices + 新用例
  earlypool / earlypool-mix)
- `csrc/attention/glm5_kpool_indexer/smoke_split_bisect.py`(**新**,二分工具)
- `csrc/build_aclnn.sh`(910_93 清单插入两算子)

## 6. 验证门(AIC 修复后依次全过才算数)

1. `smoke_split_bisect.py both tiny` → 全 17 个 smoke 用例(含 earlypool*)
   对照 triton(rows_mismatch=0;tie 按 TIE_TOL 仲裁):
```bash
sudo -n docker exec glmk-build bash -c "cd /opt/src/vllm-ascend && \
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 && \
  source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash >/dev/null 2>&1 && \
  PYTHONPATH=/opt/src/vllm-ascend python csrc/attention/glm5_kpool_indexer/smoke_compare.py \
  --cases tiny packed2 packed2x1 packed3u packed4u packed3eq single96 packed4eq packed3 \
  widescreen big contend2k contend4k contend-pack group-tail zeropool-head zeropool-mid \
  earlypool earlypool-mix --impl ascendc_group_topk_split"
```
2. qbar / replay 414(真实激活回放,哨兵目录 dump 工具在,见 experiments.md M5c)
   与 acceptance——沿用 F1 的既有脚本,impl 名换成 split。
3. 引擎 A/B(P 侧 .39):overlay 六工件(dispatcher python + `vllm_ascend_C.*.so`
   + `libvllm_ascend_kernels.so` + `_cann_ops_custom/vendors/custom_transformer/`
   等新哈希同步进容器)→ serve.sh env 切 `ascendc_group_topk_split` →
   **docker restart(无 provision,herd apply 会洗掉 overlay)** → 统一热身
   (≥5min,前几分钟 2-3× 假慢)→ aisbench 64k/192k + gsm8k-20,对照同日
   triton(15.2s / 90 分口径)。**测完回滚 env=triton**。
4. 双轨留档:主仓 experiments.md + memory;git commit → pull --rebase → push。

## 7. 关键数字速查

| 项 | 数值 |
|---|---|
| microbench T=7680@16k pools | triton 50.6ms / M32 20.9 / M64 14.2 |
| 引擎内 kernel 对照 | Glm5KpoolIndexer 0.199s vs triton 10.47s(~50×) |
| 引擎 64k TTFT | triton 15.20-15.78s / mix 20.3-20.8s(慢 33%) |
| mix×通信重叠 | 0.24ms / 26 次 launch(idle 63%) |
| SPLIT_BATCH_POOLS | 4096(POOL_GROUP=1024,S2_TILE=128,M_TILE=32) |
| 正确性参照 | F1 门:probe + smoke×3impl + qbar + replay414 + acceptance×2 全绿 |
