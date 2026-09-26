# HANDOVER — Glm5KpoolIndexer M6:topk 竞争区(满淘汰)错误攻关

> 交接目标:**定位并修复 visible_pools ≥ poolTopk(512) 竞争区的淘汰错误**。
> 稳定性(M4)与性能(M3,5.3×)已就绪;这是精度上线的唯一阻断项。
> 日期:2026-09-26 晚。前序稳定性 handover 的"已完成资产"仍有效,见 §10。
> 基线 commit:`d367bfc60`(分支 `feat/glm5-kpool-ascendc-indexer`)。

## 1. 任务定义与验收标准

**问题**:AscendC 算子在一行的可见 pool 数超过 running top-k 容量(512)需要**淘汰**
时,输出错误。非竞争区(visible < 512)bit 级正确。

**验收标准**(全部满足才算 M6 收口):
1. `smoke_compare.py` 9 case 全 PASS(含 contend2k/contend4k/contend-pack)
2. `replay_parity.py` 真实 64k serving 语料回放全 PASS(dump 语料 14G 已留)
3. `acceptance_30r.py`(9 case × 30 轮同进程)rc=0
4. M7 端到端复测:TTFT A/B(spec `ttft-bs2-192k`)+ gsm8k accuracy A/B
   (spec `run/stacks/glm53flash-029-pd/tests/acc-gsm8k-20.yaml`)——修复后重测,
   因为 012(-6.2% TTFT)/013(无精度损伤)已被降级为"bug 存在但温和"。

## 2. 现场与环境

- **worktree**:`/data1/ascend/myascend/code/vllm-ascend-wt/glm5-kpool-indexer`
  (所有命令从这里跑;勿动主 checkout)
- **构建/测试容器**:host39 本机 `glmk-build`(已停,`sudo docker start glmk-build`
  即续),镜像 `quay.io/ascend/vllm-ascend:glm53-a3-41run`,挂 worktree 于
  `/opt/src/vllm-ascend` + `/dev/davinci0` 全家桶。CANN 9.1.0 / ascend910_9391(A3)。
- **dump 语料**:`/data1/ascend/myascend/run/stacks/glm53flash-029-pd/tmp/glmk-replay/`
  (414 份 .pt,16 个 pid 目录;prefill chunk T=7680 + decode 步)
- **serving 栈**:glm53flash-029-pd(P .39 TP8DP2 / D .40 TP2DP8)目前 stop+clean,
  挂载树 python-only。M7 时按 §8 重新拉起。
- 研究档案:`research/active/glm5-kpool-indexer/{experiments,hypotheses}.md`(主仓,
  每轮实验必须追加留档,含负结果)

## 3. 构建(一票否决的坑,先读后改)

**build_aclnn 增量编译不跟踪头文件依赖**:改 `.h` 后 .o 不重编,reinstall 会刷新
mtime 造成"已更新"假象。当天多轮"修复"从未出货,直到 `rm -rf csrc/build` 全量重编
才拿到真二进制。因此:

```bash
# 容器内(/opt/src/vllm-ascend),改任何 op_kernel/op_host 头文件后:
rm -rf csrc/build                                   # 全量,别信增量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910_9391
bash csrc/build_aclnn.sh /opt/src/vllm-ascend ascend910_9391
# 验证真的出货:对比 .o 尺寸/hash(增量坑的检测器)
ls -l csrc/build/binary/ascend910_93/bin/glm5_kpool_indexer/*.o
sha256sum csrc/build/binary/ascend910_93/bin/glm5_kpool_indexer/*.o
```

host 驱动脚本 `.herd-build-m6e.sh`(容器内跑,日志落 JOB 目录)。python wrapper
(`vllm_ascend/ops/glm5_kpool_indexer.py`)改动不需要 build_aclnn,但容器验证要显式
传 PYTHONPATH(docker exec 不继承 pid1 的)。

## 4. 测试工具链(容器内)

```bash
source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 合成对拍(triton vs ascendc,集合语义比较)
ASCEND_LAUNCH_BLOCKING=1 python3 csrc/attention/glm5_kpool_indexer/smoke_compare.py \
    --cases contend4k contend2k big        # FAIL 现状:contend4k 2045/4096 行

# 真实语料回放(同一捕获张量过双链)
python3 csrc/attention/glm5_kpool_indexer/replay_parity.py \
    --dump /data1/ascend/myascend/run/stacks/glm53flash-029-pd/tmp/glmk-replay --max-calls 60

# 多轮同进程验收(python3 -u 必须,90s 塞多轮会假性 rc=124)
timeout -s ABRT 900 python3 -u -X faulthandler \
    csrc/attention/glm5_kpool_indexer/acceptance_30r.py
```

对拍比较语义:每行折成"合法 token 索引集合 + (-1) 计数"(topk 内次序合法可变)。
mode=1(outputMode=1)是 debug 通道:kernel 把每 tile 的 128 个原始 fp32 分数写
`scoresDebugGm[row*maxPoolSeqLen + s2Start]`,host 侧 smoke 已支持解析。

## 5. 当前故障精确矩阵

| case | 形状 | visible | 结果 |
|---|---|---|---|
| tiny/single96/packed2/packed3/widescreen | 各种 | <512 | PASS(非竞争区) |
| big | T=1024, 2 S2-tiles | 256 | **PASS ×282 次调用**(关键对照) |
| contend2k | T=2048 单请求, 16 tiles | ≤512 | PASS(仅最后一两个 tile 进竞争区,mask 吸收) |
| **contend4k** | T=4096 单请求, 16 tiles | 1024 | **FAIL 2045/4096 行** |
| contend-pack | [2048,1024,512] | 混合 | PASS |
| **真实 prefill chunk** | T=7680, pools≤1920 | ≤1920 | **FAIL 5629/7680 行(73%)** |
| 真实 decode 步 | T=16/8, pools=1 | 1 | PASS 22/22 |

失败行模式:恰少选 1 pool(+4 个 -1 尾);失败从**首个 visible>512 的行**开始。
受控实验:同数据 poolTopk=128/visible=512 → FAIL 1533/2048;poolTopk=512/visible=512
(零淘汰)→ PASS 0 mismatch。**失败 = 淘汰事件本身的函数**。

## 6. 已定罪 / 已洗清(勿重走!)

| 结论 | 证据 | 状态 |
|---|---|---|
| 打分/gather 路径错 | shuffled block-table 探针:mode=0 段直方图精确跟随 block table | **洗清** |
| AIV 排序链(SortFull1024 整链) | 整链置换(Sort32 32 组+定长 4 路归并树)后 deterministic 探针直方图 **bit 级不变** | **洗清**(且旧 MrgSort 递推已整体废弃替换) |
| 旧 MrgSort(512+128 / 128+128)满值合并丢 32-对 run 尾对 | 确定性探针:miss 恒为每 32-对 run 尾;inf 垫无效 | **已定罪但已移除**——换 SortFull1024 后错误**没变**,说明它不是(唯一)病因 |
| mode=1 debug 在 Sort 后拷贝=被污染 | M5c"按位置取尾段"结论已撤回;拷贝已移 Sort 前 | **已修**(d367bfc60) |
| 淘汰只在 strip 满时发生 | 受控 poolTopk 128/512 实验 | **已证实** |

## 7. 核心代码地图(arch22 = 910B/A3 路径)

```
csrc/attention/glm5_kpool_indexer/
├── op_kernel/arch22/
│   ├── glm5_kpool_indexer_kernel.h          # Process/ProcessUnit:RunInfo 构造+主循环+MODE2 握手
│   │   └─ :287-316  for s2Tile { loop/s2Start=128t/... ; AIC: ComputeMm1→SetFlag(C1V1);
│   │                   AIV: WaitFlag(C1V1)→ProcessVec→(last)ProcessTopK→SetFlag(V1C1) }
│   │   └─ :326-329  AIV 起手 pre-arm 2 × syncV1C1_(官方 lightning 模式)
│   ├── glm5_kpool_indexer_service_cube.h    # AIC 侧
│   │   └─ :148-180  KeyNd2NzForPA:logicalPool = s2Start+s2L1Offset → blockTable → keyGmOffset
│   │   └─ :250-266  Fixp(NZ2ND):写 mm1ResGm_[(loop%2)*M_BLOCK*S2_BLOCK],PIPE_FIX flag
│   │   └─ :269-      ComputeMm1:key/query L1 ping-pong + Mmad + 事件生命周期(M4 修)
│   ├── glm5_kpool_indexer_service_vector.h  # AIV 侧(当前病灶区)
│   │   └─ :187-255  FoldBlockIntoRowTopk:
│   │        rowBase=(loop%2)*M_TILE*S2_TILE + rowInTile*S2_TILE → DataCopyPad scoreUb_
│   │        laneMask 越界 pool → -inf;mode=1 debug CopyOut(**Sort 前**);
│   │        ArithProgression(scoreIdx, s2Start);SortAll(128);
│   │        strip append at poolTopk*2 → InitSortOutBuf 垫 → SortFull1024(strip 1024 对)
│   │   └─ :272-296  EmitRow:ExtractIndex 前 poolTopk 对 → indicesOutGm
│   └── glm5_kpool_indexer_vector.h          # SortAll/SortFull1024/InitSortOutBuf/ExtractIndex
├── op_host/glm5_kpool_indexer_tiling.cpp    # poolTopk==512 断言;workspace=aicNum*2*32*128*4B
├── smoke_compare.py / acceptance_30r.py / replay_parity.py / perf_bench.py
└── glm5_kpool_indexer_torch_adpt.h          # host adpt(mode 参数在此)
```

数据流:AIC 每 S2-tile 产出 [32 行 × 128 pool] fp32 分数 → per-AIC mm1Res GM strip
(深度 2 ping-pong)→ 配对 AIV 各读 16 行半区 → mask → Sort128 → append 进
per-row running strip(1024 对:512 live + 128 新 + 384 -inf 垫)→ SortFull1024
全量重排;最后 S2-tile 后 EmitRow 抽 index 半区输出(python wrapper 负责 ×4 展开、
越界过滤、causal tail)。

## 8. 三个候选假设与第一步(判别实验优先于任何修复)

**观察到的关键异常**:mode=1 debug(已修到 Sort 前拷贝)显示**从 2-tile 请求起,
每个 tile 写出的分数恒等于 tile0 的分数**。

- **(a) 数据通路真错**:AIC→AIV 每 fold 供的都是 tile0 的 mm1Res(KeyNd2Nz 的
  key L1 ping-pong 读旧 / blockTable GetValue 错 / Fixp 半区错位)。
  **张力**:big(T=1024, 2 tiles, visible 256)282 次 PASS——若 tile1 分数=tile0 的,
  strip 必然吸收错误分数,AIV 又用**正确的 s2Start** 生成 pool id(ArithProgression),
  输出不可能对。big PASS 几乎否定 (a) 的"从 2 tiles 起"形态。
- **(b) debug 写出是假象**:写出偏移/通路另有 bug,fold 数据其实正确,真因在
  AIV fold 后段(strip 跨 fold 的 UB 复用 / EmitRow / python wrapper 的竞争区分支)。
- **(c) 时序竞态(交接时新提出,建议优先)**:mm1Res ping-pong 深度=2,而 AIV 起手
  pre-arm 2 flag 允许 AIC 领先 AIV 恰好 2 个 tile——同 parity 半区存在
  "AIC 写 tile N + AIV 读 tile N-2"的理论竞态窗口。big(2 tiles)永远够不到该窗口
  (领先 ≤1),**≥3 tiles 才可能**——与"big 稳 PASS / 16-tile 的 contend4k 挂"完美
  相容;确定性探针下读到撕裂/旧数据也可能呈现"恒=tile0"。静态推演两边 flag 语义
  (PIPE_FIX / PIPE_MTE2)表面都安全,但 MODE2 计数语义与 Fixp 的 NZ2ND 异步写出
  的交互值得用实验打。

**第一步实验(≤1 小时,判 (a)/(b)/(c) 于一次运行)**:
1. 编码探针 + mode=1:cache 构造 score(p) = 单调函数(如 2048−p),T=4096 单请求。
   每 tile 的 debug 槽预期内容已知(tile t ⇒ f(128t..128t+127))。逐槽比对:
   - 全槽 = f(0..127) → (a) 系(key 恒 tile0)
   - 槽 t = f(128(t−2)..) 或撕裂 → (c)(读旧半区,滞后 2)
   - 全槽正确但 mode=0 输出仍 FAIL → (b)(真因在 AIV fold 后段/wrapper)
   注意逻辑细节:若 debug 写出位置真覆盖(全写槽 0),则读出应是"最后写入者"而非
   "恒 tile0"——观察本身已偏向数据通路错误,除非 AIC 写的内容就恒 tile0。
2. 若指向 (c):最小修复 = ping-pong 深度 2→4(Fixp dstOffset 与 AIV bufBase 的
   `%2` 同步改 `%4`;workspace 已有公式,扩到 4×≈3MB 无压力),一次重编即判。
3. 若指向 (a):在 KeyNd2NzForPA 后用 Fixp 前把 keyL1 槽位打出(或 blockTable
   GetValue 结果打点),定位 key 侧哪个变量不随 s2Tile 变。
4. 若指向 (b): suspects = strip 的跨 fold UB 生命周期(globalTopkUb_ 在
   LoadRowMeta 重置的时机)、EmitRow ExtractIndex 的 topkStride 偏移、
   python wrapper 竞争区分支(`vllm_ascend/ops/glm5_kpool_indexer.py`)。

**修改任何 .h 后必须走 §3 全量重编,并用 .o hash 变化自证。**

## 9. 修复后验收路径(顺序执行)

1. 判别实验 → 修复 → `smoke_compare.py --cases` 9 case 全 PASS
2. `replay_parity.py --max-calls 414` 真实语料全 PASS
3. `acceptance_30r.py`(9 case × 30 轮)rc=0
4. worktree commit + 主仓 `research/active/glm5-kpool-indexer/experiments.md`
   与 `docs/glm-5.3-flash/model-profile.md`(015 条目)双轨留档
5. **M7 端到端**(需 serving 栈):
   - 部署纪律(§7 坑):先 herd apply 起栈,**最后**手工同步挂载树
     (dispatcher python + `_cann_ops_custom/vendors`),之后只 `docker restart`,
     不得再带 provision 的 apply/restart;D 侧不动(python-only)
   - `herd test` 用 spec `ttft-bs2-192k`(基线:triton median 69.3s / ascendc 65.0s,
     数据只作修复后参照,须重测双臂);TTFT 解析:db_data/*.db numpy_store,
     point[1]−point[0]
   - accuracy:spec `acc-gsm8k-20`;restart 后先手工 curl 预热避免 MODEL-DATA-003
     warmup-500;容器内端口检查用 netstat(ss 缺失静默失败)
   - 判据:修复后 TTFT 收益保持且 gsm8k 与 triton 同分(每题对比,20 题样本
     ±1 题=5pt,同题同错才算模型级)
6. 全绿后:算子定型(讨论是否回源仓主线 + 编译产物进镜像,消除手工挂载脆弱性)

## 10. 资产与提交链

- **提交链**(本 worktree):`0f097122b`(replay 工具)→ `8847388da`(M4 挂死修复)
  → `2c27c5e32`(SortAll 层级归并+debug 顺序+竞争 case)→ `d367bfc60`(SortFull1024
  + poolTopk 断言 + 探针;当前 HEAD)
- **前序成果**(M1-M5,勿重做):gather8 布局修 AIV GM 读 fault;TPipe VECOUT 修
  EmitRow 竞态;PIPE_M 事件生命周期修复(282 次零挂);kernel 级 5.3×(long shape);
  E2E 接线就绪(`VLLM_ASCEND_GLM5_KPOOL_INDEXER_IMPL=ascendc`);
  挂载树基线备份 `research/active/glm5-kpool-indexer/mount-backup-20260926/`
- **工具**:`perf_bench.py`(性能)、`probe_lost_task.py`、`lightning_hang_probe.py`
  (lightning 对照)、`repro_hang.sh`(工单材料,已过时——M4 已闭环)
- vendored 参照实现:`csrc/attention/lightning_indexer/`(同构 arch22 混合核,
  生产验证;handshake/SortAll 模式均出自它)

## 11. 纪律提醒

- 遵守 worktree AGENTS.md(14 步推理序、负结果必须留档)
- 每轮实验:先写假设 → 单一变量 → 记录样本数(n≥8 才有区分力)
- 构建后**必须**用 .o hash 变化自证新二进制出货(§3 坑的检测器)
- 长操作(全量构建 ~15-20 分钟)Job 化,日志落文件轮询,勿阻塞
- 容器内 `pkill -f` 会误杀 herd exec;用 `pgrep -f "^python3 csrc"` 行首锚定
- 挂死后设备偶发 507033,等 1-2 分钟自愈
- 修改仅限本 worktree;主仓只更新 research/docs 留档(双轨纪律见主仓 CLAUDE.md)
