# V1 数据处理 + V0 训练 — 端到端 SFT 方案

## 架构

```
data_openai/                          V1 预处理 (本目录)               V0 训练
═══════════                          ════════════════               ════════
OpenAI 原生格式                       v1_openai_converter.py         LLaMAFactory V0
( tool_calls 数组                    v1_tokenize_dataset.py         HfTrainer
  reasoning_content )    ──────→      Renderer.apply_chat_template   + tokenized_path
                                      ↓
                                      tokenized_dataset/
                                      ( input_ids / labels / loss_weights )
```

**核心思路**: 用 V1 的 Renderer 做 token 化（自动处理 `reasoning_content` + `tool_calls`），产出 HfDataset，V0 通过 `tokenized_path` 直接加载，跳过 Template 编码。

## 相比纯 V0 的优势

| 特性 | 纯 V0 | 本方案 |
|------|-------|--------|
| `reasoning_content` 处理 | ❌ 需手动转 `<think>` 标签 | ✅ V1 Renderer 自动设 `enable_thinking=True` |
| `<system-reminder>` 处理 | ❌ 手动过滤 | ✅ 转换器内置处理（可开关） |
| `tool_calls` 格式 | ❌ 需拆为 `tool_call` 角色 | ✅ 保持 OpenAI 原生 `tool_calls` 数组 |
| 多轮 SFT 切分 | ❌ 手动 | ✅ 自动 prefix-split (每个 assistant turn 一个样本) |
| 训练后端 | ✅ DeepSpeed/FSDP | ✅ DeepSpeed/FSDP (V0 HfTrainer) |
| 源码改动 | 无 | 无 |

## 目录

```
v1_training_llamav1/
├── v1_openai_converter.py             # OpenAI 原生格式 → V1 Message 格式转换器
├── v1_tokenize_dataset.py             # 批量 token 化脚本
├── v0_train_config.yaml               # V0 训练配置（普通 HfTrainer）
├── v0_train_config_megatron_cp8.yaml  # V0 训练配置（Megatron CP=8）
└── README.md
```

## 快速开始

### 环境要求

```bash
pip install llamafactory datasets torch tqdm
USE_V1=1  # 启用 LLaMAFactory V1 模块
```

### Step 1 — Token 化数据

```bash
USE_V1=1 python v1_tokenize_dataset.py \
    --model Qwen/Qwen3.5-2B \
    --data_dir /data2/cyr/development/coding-sft/data_openai \
    --cutoff_len 32768 \
    --output_dir ./tokenized_dataset \
    --num_workers 8
```

脚本执行流程：
1. 递归加载 `data_dir` 下所有 JSON 文件
2. `v1_openai_converter` 转换每条数据为 V1 Message 格式（保留 `<system-reminder>`，映射 `reasoning_content` + `tool_calls`）
3. 前缀拆分（仅 `--train all`）：每条多轮对话按 assistant turn 拆为多个训练样本
4. V1 `Renderer.process_samples()` 调用 `tokenizer.apply_chat_template()` 做 token 化（多线程并行）
5. 保存为 HfDataset → `tokenized_dataset/`

预期输出：
```
[Load] Found 37 JSON files in /data2/cyr/development/coding-sft/data_openai
[Load] Total: 744 items
[Convert] 744 samples ready
[Split] 744 conversations → ~3000+ training samples
[Tokenize] N valid, M dropped
[Save] → ./tokenized_dataset
```

### Step 2 — 训练

```bash
llamafactory-cli train v0_train_config.yaml
```

训练配置核心参数：

```yaml
tokenized_path: ./tokenized_dataset   # ← 加载预 token 化数据（优先于 dataset）
dataset: alpaca_en_demo               # ← 占位（parser 要求非空，实际不会被加载）
template: qwen3_5                      # ← 仅用于 data_collator / special tokens
cutoff_len: 32768                      # ← 必须 >= token 化时的 cutoff_len
```

## 数据格式说明

### 输入格式 (data_openai)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful agent."},
    {"role": "user", "content": "Fix the bug."},
    {"role": "assistant",
     "reasoning_content": "I need to search.",
     "content": "Let me look it up.",
     "tool_calls": [{
       "id": "call_1",
       "function": {"name": "grep", "arguments": "{\"pattern\": \"foo\"}"}
     }]},
    {"role": "tool", "tool_call_id": "call_1", "content": "line 42: def foo(x)"},
    {"role": "assistant",
     "reasoning_content": "Found the issue.",
     "content": "The fix is on line 42."}
  ]
}
```

### 转换流程

```
OpenAI 原始 → v1_openai_converter → V1 Message (结构化 content list)
            → Renderer._to_hf_messages() → 回到 OpenAI 格式 (带 reasoning_content + tool_calls)
            → tokenizer.apply_chat_template(enable_thinking=True) → token IDs
```

### V1 Message 格式 (中间态)

```json
{
  "role": "assistant",
  "content": [
    {"type": "reasoning", "value": "I need to search."},
    {"type": "text", "value": "Let me look it up."},
    {"type": "tool_call", "value": "{\"name\": \"grep\", \"arguments\": {...}}"}
  ],
  "loss_weight": 1.0
}
```

## 配置调整

### 修改 cutoff_len

```bash
# Token 化时
python v1_tokenize_dataset.py --cutoff_len 65536 ...

# 训练 YAML 中
cutoff_len: 65536
```

### 修改 output 目录

```bash
python v1_tokenize_dataset.py --output_dir ./tokenized_64k ...
```

YAML 中同步修改:
```yaml
tokenized_path: ./tokenized_64k
output_dir: ./saves/qwen3.5-2b-64k
```

### 多机训练 (DeepSpeed)

添加 deepspeed 配置:
```yaml
deepspeed: ds_z3_config.json
```

### Megatron 后端（长序列 / CP）

V1 token 化 + V0 训练同样支持 Megatron 后端（MCA），用于 TP/PP/CP 并行:

```bash
USE_MCA=1 llamafactory-cli train v0_train_config_megatron_cp8.yaml
```

并行策略:
```yaml
# v0_train_config_megatron_cp8.yaml
finetuning_type: full            # MCA 目前仅支持 full 微调（不支持 LoRA）
use_mca: true                    # 路由到 Megatron 后端
context_parallel_size: 8         # CP=8，序列拆分到 8 卡
tensor_model_parallel_size: 1    # TP=1
pipeline_model_parallel_size: 1  # PP=1
# 总计需 8 GPU
```

> **注意：** MCA 目前仅支持 `finetuning_type: full`，不支持 LoRA。需要 8 卡 memory 足以容纳 2B 模型的完整 optimizer states。

## 自定义转换器

如果数据格式不同，继承 `DataConverterPlugin` 注册新转换器：

```python
from llamafactory.v1.plugins.data_plugins.converter import DataConverterPlugin

@DataConverterPlugin("my_format").register()
def my_converter(raw_sample):
    messages = []
    # ... your conversion logic ...
    return {"messages": messages}
```

在 tokenize 脚本中 `import my_converter` 即可生效。
