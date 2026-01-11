# Qwen3-Next 模型结构详解

## 1. 模型概述

Qwen3-Next 是通义实验室开发的下一代大型语言模型，采用了创新的**混合注意力架构**，结合了线性注意力和标准注意力机制，并支持混合专家（MoE）架构。

### 1.1 核心特性

- **混合注意力机制**：交替使用线性注意力（Linear Attention）和完整注意力（Full Attention）
- **Gated DeltaNet**：基于 Mamba 的线性注意力变体
- **混合专家架构**：部分层使用 MoE 提升模型容量
- **高效推理**：使用 V1 引擎优化，支持投机采样
- **多 NPU 支持**：针对 Ascend NPU 进行优化

### 1.2 技术要求

- 必须使用 `VLLM_USE_V1` 环境变量
- 暂不支持 Prefix Caching
- 需要安装 Triton Ascend（实验性功能）
- 使用 GDNAttentionBackend 作为注意力后端

---

## 2. 模型整体架构

```
CustomQwen3NextForCausalLM
├── Embedding Layer
│   └── embed_tokens (VocabParallelEmbedding)
├── CustomQwen3NextModel
│   ├── layers (N 个 CustomQwen3NextDecoderLayer)
│   └── norm (Qwen3NextRMSNorm)
└── LM Head
    └── lm_head (ParallelLMHead)
```

### 2.1 类继承关系

```
CustomQwen3NextForCausalLM
    └── Qwen3NextForCausalLM (vllm 基类)
            └── nn.Module

CustomQwen3NextModel
    └── Qwen3NextModel (vllm 基类)
            └── nn.Module

CustomQwen3NextDecoderLayer
    └── Qwen3NextDecoderLayer (vllm 基类)
            └── nn.Module

CustomQwen3NextGatedDeltaNet
    └── Qwen3NextGatedDeltaNet (vllm 基类)
            └── MambaBase
                    └── nn.Module
```

---

## 3. 核心组件详解

### 3.1 Decoder Layer 结构

每个 Decoder Layer 根据配置的 `layer_types` 决定使用线性注意力还是完整注意力：

```python
CustomQwen3NextDecoderLayer
├── linear_attn (Linear Attention) 或 self_attn (Full Attention)
├── mlp (Standard MLP 或 MoE Block)
├── input_layernorm (RMSNorm)
├── post_attention_layernorm (RMSNorm)
└── layer_scale (可选，Layer Scale 参数)
```

#### 3.1.1 层类型配置

模型通过配置文件中的 `layer_types` 数组定义每一层使用的注意力类型：

- `linear_attention`: 使用 CustomQwen3NextGatedDeltaNet
- `full_attention`: 使用 Qwen3NextAttention（标准多头注意力）

#### 3.1.2 MLP 配置

- **普通层**：使用 `Qwen3NextMLP`（基于 Qwen2MoeMLP）
- **MoE 层**：当满足以下条件时使用 `Qwen3NextSparseMoeBlock`：
  - `layer_idx` 不在 `mlp_only_layers` 中
  - `num_experts > 0`
  - `(layer_idx + 1) % decoder_sparse_step == 0`

### 3.2 线性注意力层 (Gated DeltaNet)

`CustomQwen3NextGatedDeltaNet` 是核心创新组件，实现了高效的线性注意力机制。

#### 3.2.1 继承与属性

```python
class CustomQwen3NextGatedDeltaNet(Qwen3NextGatedDeltaNet, MambaBase):
    mamba_type = "linear_attention"
    attention_backend = GDNAttentionBackend
```

#### 3.2.2 核心参数

| 参数 | 说明 | 来源 |
|------|------|------|
| `hidden_size` | 隐藏层维度 | `config.hidden_size` |
| `num_v_heads` | Value 注意力头数 | `config.linear_num_value_heads` |
| `num_k_heads` | Key 注意力头数 | `config.linear_num_key_heads` |
| `head_k_dim` | 每个 Key 头的维度 | `config.linear_key_head_dim` |
| `head_v_dim` | 每个 Value 头的维度 | `config.linear_value_head_dim` |
| `conv_kernel_size` | 卷积核大小 | `config.linear_conv_kernel_dim` |
| `activation` | 激活函数 | `config.hidden_act` |

#### 3.2.3 模块组成

```
CustomQwen3NextGatedDeltaNet
├── conv1d (ColumnParallelLinear)
│   └── 输入: [conv_kernel_size] → 输出: [key_dim * 2 + value_dim]
├── in_proj (MergedColumnParallelLinear)
│   ├── in_proj_qkvz: [hidden_size] → [key_dim * 2 + value_dim * 2]
│   └── in_proj_ba: [hidden_size] → [num_v_heads * 2]
├── dt_bias (Parameter)
│   └── 时间步偏置，shape: [num_v_heads // tp_size]
├── A_log (Parameter)
│   └── 状态转移矩阵对数，shape: [num_v_heads // tp_size]
├── norm (RMSNormGated)
│   └── 门控归一化层
└── out_proj (RowParallelLinear)
    └── 输出投影: [value_dim] → [hidden_size]
```

#### 3.2.4 前向传播流程

线性注意力的前向传播分为三个主要阶段：

**阶段 1: 输入投影**
```python
hidden_states → in_proj → projected_states
projected_states.split() → [projected_states_qkvz, projected_states_ba]
[q, k, v, z, b, a] = fix_query_key_value_ordering(...)
```

**阶段 2: 卷积序列变换**
- 对于 Prefill 阶段：使用 `causal_conv1d_fn`
- 对于 Decode 阶段：使用 `causal_conv1d_update`
- 支持投机采样（Speculative Decoding）

```python
mixed_qkv → causal_conv1d_fn/update → convolved_qkv
```

**阶段 3: 循环注意力计算**
- 使用 `fused_recurrent_gated_delta_rule` 或 `chunk_gated_delta_rule`
- 计算门控参数：
  ```python
  beta = b.sigmoid()
  g = fused_gdn_gating(A_log, a, dt_bias)
  ```
- 更新 SSM 状态并计算核心注意力输出

**阶段 4: 输出归一化与投影**
```python
core_attn_out → norm(z) → out_proj → output
```

#### 3.2.5 状态管理

Gated DeltaNet 维护两种状态：

1. **Conv State** (卷积状态)
   - Shape: `[batch, key_dim * 2 + value_dim, conv_kernel_size - 1]`
   - 用于因果卷积的历史信息

2. **SSM State** (状态空间模型状态)
   - Shape: 由 `get_state_shape()` 计算
   - 存储递归计算的中间状态

---

### 3.3 完整注意力层 (Full Attention)

对于使用 `full_attention` 类型的层，采用标准的 Qwen3NextAttention：

```python
Qwen3NextAttention
├── QKV Projection (ColumnParallelLinear)
├── Rotary Position Embedding (RoPE)
├── Attention Backend (标准注意力机制)
└── Output Projection (RowParallelLinear)
```

这种层通常用于模型的关键位置，确保对长距离依赖的建模能力。

---

### 3.4 混合专家层 (MoE Block)

`Qwen3NextSparseMoeBlock` 在特定层中激活：

#### 3.4.1 激活条件

```python
if (layer_idx not in mlp_only_layers) and \
   (config.num_experts > 0) and \
   ((layer_idx + 1) % config.decoder_sparse_step == 0):
    使用 MoE Block
else:
    使用 Standard MLP
```

#### 3.4.2 MoE 参数（从模型中提取）

```python
num_moe_layers: MoE 层数量
num_logical_experts: 逻辑专家数量
num_physical_experts: 物理专家数量
num_local_physical_experts: 每个 NPU 上的本地物理专家数
num_routed_experts: 路由专家数量
num_redundant_experts: 冗余专家数量（EPLB 功能）
```

---

### 3.5 归一化层

使用 **GemmaRMSNorm**（别名为 `Qwen3NextRMSNorm`）：

```python
Qwen3NextRMSNorm
├── weight: 可学习的缩放参数
└── eps: 数值稳定常数 (rms_norm_eps)
```

位置：
- `input_layernorm`: 在注意力/MoE 之前
- `post_attention_layernorm`: 在 MLP 之前
- `norm`: 模型输出的最终归一化

---

### 3.6 Layer Scale（可选）

当 `config.layer_scale = True` 时，每层包含两个可学习的 Layer Scale 参数：

```python
attn_layer_scale: 注意力输出的层缩放
ffn_layer_scale: FFN 输出的层缩放
```

初始化为零，允许模型从恒等映射开始训练。

---

## 4. 模型配置关键参数

### 4.1 Qwen3NextConfig 关键字段

```python
# 基础配置
hidden_size: 隐藏层维度
num_hidden_layers: 隐藏层数量
vocab_size: 词汇表大小
rms_norm_eps: RMSNorm 的 epsilon 值
hidden_act: 激活函数类型

# 线性注意力配置
linear_num_value_heads: Value 头数量
linear_num_key_heads: Key 头数量
linear_key_head_dim: Key 头维度
linear_value_head_dim: Value 头维度
linear_conv_kernel_dim: 卷积核维度

# 层类型配置
layer_types: List[str] - 每层的注意力类型 ["linear_attention", "full_attention", ...]

# MoE 配置
num_experts: 专家数量
decoder_sparse_step: MoE 层的间隔步长
mlp_only_layers: 仅使用 MLP 的层索引列表

# Layer Scale
layer_scale: bool - 是否启用 Layer Scale
```

### 4.2 并行配置

- **Tensor Parallelism**: 支持张量并行，通过 `tp_size` 和 `tp_rank` 管理
- **Expert Parallelism**: 通过 EPLB (Expert Parallel Load Balancing) 支持

---

## 5. 数据流与计算流程

### 5.1 整体前向传播

```
Input IDs
    ↓
Embedding (embed_tokens)
    ↓
For each Decoder Layer:
    ├─→ input_layernorm
    ├─→ Attention (Linear or Full)
    │   ├─→ Residual Connection
    ├─→ post_attention_layernorm
    ├─→ MLP (Standard or MoE)
    │   └─→ Residual Connection
    └─→ Layer Scale (可选)
    ↓
Final Norm
    ↓
LM Head (lm_head)
    ↓
Logits
```

### 5.2 线性注意力详细流程

```
hidden_states [batch, seq_len, hidden_size]
    ↓
in_proj 投影 → [Q, K, V, Z, B, A]
    ↓
1D Convolution (causal_conv1d) → Convolved QKV
    ↓
计算门控参数:
    - β = sigmoid(B)
    - g = gating_function(A_log, A, dt_bias)
    ↓
Gated Delta Rule (Recurrent Attention):
    - Prefill: chunk_gated_delta_rule (逐批次处理)
    - Decode: fused_recurrent_gated_delta_rule (融合递归)
    ↓
RMSNorm with Gating (z)
    ↓
out_proj 投影 → [batch, seq_len, hidden_size]
```

### 5.3 投机采样支持

模型原生支持投机采样（Speculative Decoding）：

- 区分 `spec_tokens` 和 `non_spec_tokens`
- 分别处理投机采样和常规 tokens
- 使用 `spec_state_indices_tensor` 和 `non_spec_state_indices_tensor` 管理状态
- 支持 `num_accepted_tokens` 追踪

---

## 6. 内存与状态管理

### 6.1 KV Cache 结构

对于线性注意力层，维护两种缓存：

```python
kv_cache[virtual_engine] = [
    conv_state,    # [batch, key_dim*2+value_dim, conv_kernel_size-1]
    ssm_state      # [batch, num_heads, head_dim, head_dim]
]
```

对于完整注意力层，使用标准 KV Cache。

### 6.2 状态形状计算

```python
def get_state_shape():
    conv_shape = (batch, key_dim*2+value_dim, conv_kernel_size-1)
    ssm_shape = (batch, num_v_heads, head_v_dim, head_v_dim)
    return conv_shape, ssm_shape
```

### 6.3 状态数据类型

```python
def get_state_dtype():
    conv_state_dtype = model_config.dtype
    ssm_state_dtype = cache_config.mamba_cache_dtype
    return conv_state_dtype, ssm_state_dtype
```

---

## 7. 权重加载与映射

### 7.1 堆叠参数映射

权重加载时，将多个参数合并加载：

```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("in_proj", "in_proj_qkvz", 0),
    ("in_proj", "in_proj_ba", 1),
]
```

### 7.2 专家权重映射

对于 MoE 层，使用 `expert_params_mapping` 处理专家权重的分片加载。

### 7.3 特殊处理

- 跳过 `rotary_emb.inv_freq`（不需要加载）
- 跳过 `mtp.` 前缀的权重（MTP 相关）
- 支持 GPTQ 模型的额外偏置跳过

---

## 8. 性能优化特性

### 8.1 并行计算

- **Tensor Parallelism**: Q、K、V、投影层按头维度分片
- **ColumnParallelLinear**: 输入维度不分片，输出维度分片
- **RowParallelLinear**: 输入维度分片，输出维度不分片

### 8.2 算子融合

- `fused_recurrent_gated_delta_rule`: 融合递归 Delta Rule 计算
- `causal_conv1d_fn/update`: 高效因果卷积
- `RMSNormGated`: 门控归一化融合

### 8.3 内存优化

- 状态复用：Conv State 和 SSM State 跨请求复用
- 增量更新：Decode 阶段仅更新新 token 的状态
- 精确控制：使用 `inplace_final_state=True` 原地更新状态

---

## 9. NPU 适配特性

### 9.1 Ascend NPU 优化

- `device="npu"`: RMSNormGated 层指定 NPU 设备
- `mamba_v2_sharded_weight_loader`: NPU 友好的分片权重加载
- 支持 Triton Ascend JIT 编译（实验性）

### 9.2 多 NPU 推荐配置

对于 **Qwen3-Next-80B-A3B-Instruct**：

- 64GB 显存: `tensor_parallel_size >= 4`
- 32GB 显存: `tensor_parallel_size >= 8`

---

## 10. 使用示例

### 10.1 模型加载

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    tensor_parallel_size=4,
    enforce_eager=True,
    gpu_memory_utilization=0.7,
    max_model_len=4096
)
```

### 10.2 生成

```python
from vllm import SamplingParams

prompts = ["Who are you?"]
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    max_tokens=32
)

outputs = llm.generate(prompts, sampling_params)
```

---

## 11. 限制与注意事项

### 11.1 当前限制

- ❌ 不支持 Prefix Caching
- ❌ 必须使用 V1 引擎 (`VLLM_USE_V1=1`)
- ⚠️  Triton Ascend 为实验性功能，未来可能有变更

### 11.2 推荐实践

- 使用 `--enforce-eager` 避免图编译问题
- 对于大模型（80B），确保足够的 NPU 数量和显存
- 优先使用 ModelScope 下载模型（`VLLM_USE_MODELSCOPE=True`）

---

## 12. 总结

Qwen3-Next 是一个创新的混合架构语言模型，主要特点：

1. **线性注意力 + 标准注意力混合**：平衡效率和性能
2. **Gated DeltaNet**：基于 Mamba 的高效线性注意力变体
3. **MoE 支持**：稀疏激活的混合专家架构
4. **Ascend NPU 深度优化**：专为华为 NPU 设计
5. **投机采样支持**：原生支持 speculative decoding

该架构在保持强大建模能力的同时，通过线性注意力显著降低了长序列的计算复杂度。

---

**文档版本**: v1.0
**适用版本**: vllm-ascend v0.11.0-dev
**最后更新**: 2025-01-10
