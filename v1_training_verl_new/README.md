# VeRL SFT 多轮对话 + loss-norm 数据管线

为 VeRL SFT 准备**多轮对话形式**的训练数据，支持三种 loss 归一化模式。
多轮形式（一条对话一个样本，VeRL 单次 forward 监督所有 assistant turn）比
前缀拆分形式 token 量少约 **6.8 倍**（实测 100 条对话），训练显著更快。

## 数据流

```
data_openai/                     Step 1                    Step 2
(OpenAI 格式 JSON)   ──→   multiturn_merged.jsonl   ──→   sft_train_<mode>.parquet
                            (完整多轮对话, 可检视)          (预 token 化 + 连续 loss_mask)
                                    │                             │
                                    │                             └─→ 训练时 qwen_multiturn_sft_dataset.py
                                    │                                 直通读取, 不再 tokenize
                                    └── 中间产物, 人工检视
```

## Step 1: 前缀合并

```bash
python merge_split_to_multiturn.py \
    --input /path/to/data_openai \
    --output ./data/multiturn_merged.jsonl
```

- 检测 messages 前缀链：样本 A 是样本 B 的严格前缀 → A 被 B 吸收
- 完全重复的对话去重
- 无前缀关系的数据原样通过（幂等）
- 输出一行一条完整对话，含 `conversation_id` 和 `source_file`

## Step 2: 转换 + tokenize + loss-norm

```bash
python multiturn_to_verl_norm.py \
    --input ./data/multiturn_merged.jsonl \
    --output ./data/sft_train_turn.parquet \
    --model Qwen/Qwen3.5-2B \
    --loss-norm turn        # token / turn / sqrt
    # 默认过滤: 多模态(content为list) + 无assistant结尾的对话
    # 加 --no-filter 关闭过滤, 保持原行为
```

过滤行为（与 LLaMAFactory 管线对齐）：
- **多模态**：任何消息 `content` 是 list（OpenAI image_url 块）→ 丢弃
- **无 assistant 结尾**：最后一条消息不是 assistant → 丢弃（无监督信号）
- `--no-filter`：关闭过滤，全部保留（732 条）
- 默认过滤后：**704 条**（multimodal=1, no_assistant_end=27）

转换规则（与 v1_training_verl 旧管线一致）：
- `tool_calls` 数组 → Qwen XML 内联到 assistant content
- `tool` 角色 → `user` + `<tool_response>` 包裹
- `reasoning_content` 字段（或 content 内嵌 `<think>`）→ 保留为 reasoning_content
- system 消息**保留**（agent 数据依赖 system prompt 中的工具说明）

tokenize 方式：逐消息独立 `apply_chat_template` + 拼接（与
`qwen_multiturn_sft_dataset.py` 运行时路径同一套逻辑，含 dummy-user
前缀剥离 fallback）。

### 三种 loss-norm 模式

| 模式 | 每 token 权重 | 每轮总权重 | 说明 |
|------|--------------|-----------|------|
| `token` | 1 | len | per-token 平均（默认，长轮主导） |
| `sqrt` | 1/√len | √len | 折中：长轮权重更大但被阻尼 |
| `turn` | 1/len | 1 | per-turn 等权 |

数学：VeRL 的 `sft_loss` = `-masked_sum(log_prob, loss_mask) / mask.sum()`，
连续 mask 自动适配加权平均。每轮总权重 = len × w_token。

### 输出 Parquet 列

| 列 | 用途 |
|---|---|
| `messages` | 转换后的多轮对话（检视） |
| `input_ids` / `attention_mask` / `position_ids` | 训练 |
| `loss_mask` | 连续值 mask（已含归一化，检视） |
| `turn_lengths` | 每轮 assistant 响应 token 长度（检视归一化依据） |

## 训练

`qwen_multiturn_sft_dataset.py` 支持两种模式自动切换：
- Parquet 有 `input_ids`/`loss_mask` 列 → **pre-tokenized 直通**（跳过 tokenize）
- 只有 `messages` 列 → 运行时 tokenize（0/1 mask，兼容旧数据）

VeRL 配置示例（`custom_cls` 指向本目录的 dataset 类）：

```yaml
data:
  train_files: ./data/sft_train_turn.parquet
  custom_cls:
    path: qwen_multiturn_sft_dataset.py
    name: QwenMultiTurnSFTDataset
  pad_mode: no_padding
  max_length: 65536
  truncation: left      # 超长保留尾部（最新轮次）
  loss_mask_key: loss_mask
```

## 验证结果（732 条对话, 13,629 个 assistant turn）

- 三种模式每轮总权重校验：0 errors（= len / √len / 1）
- pre-tokenized 与运行时 tokenize 前 10 条逐 token 一致
- 合并 vs 拆分 token 比值：**6.81x**（拆分形式的训练成本）
