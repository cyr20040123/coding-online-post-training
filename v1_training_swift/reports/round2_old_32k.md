# Round 2: Old Data (Parquet-converted) 32K SFT

## 基本信息
- **数据文件**: `data/jiuwen_sft_train.jsonl` (731 samples, from Parquet)
- **开始时间**: 2026-07-29 20:56:26 CST
- **完成时间**: 2026-07-29 22:29 CST
- **耗时**: ~1.5h
- **退出码**: 0 (成功)

## 配置
| 参数 | 值 |
|---|---|
| 命令 | `megatron pt` |
| 模型 | Qwen/Qwen3.5-2B |
| 上下文 | 32K (`max_length=32768`) |
| 并行 | TP=1 PP=1 CP=8 DP=1 (4K tokens/GPU) |
| 训练类型 | LoRA r=64, alpha=64, all-linear |
| Batch | micro=1, global=16 |
| Epochs | 4 |
| LR | 1e-4, cosine schedule, warmup=10% |
| Loss scale | `last_round+ignore_empty_think` |
| 模板 | `agent_template=qwen3_5`, `template_backend=jinja` |
| 精度 | bf16 |
| 显存优化 | `recompute=selective`, `no_cross_entropy_loss_fusion`, `expandable_segments:True` |
| 设备 | 8× RTX 3090 (24GB) |

## Loss 变化

| Checkpoint | Step | Loss | 说明 |
|---|---|---|---|
| — | 1 | 4.16 | 初始 |
| checkpoint-45 | 45 | 0.34 | Epoch 1 完成 |
| checkpoint-90 | 90 | 0.12 | Epoch 2 完成 |
| — | 91 | 0.11 | Epoch 2 → 3 过渡 |
| checkpoint-135 | 135 | 0.04 | Epoch 3 完成 |
| — | 137 | 0.04 | Epoch 3 → 4 过渡 |
| checkpoint-180 | 180 | 0.03 | Epoch 4 末 |
| checkpoint-182 | 182 | **0.03** | 最终 |

## 输出
- **Checkpoint**: `output/qwen3.5-2b-lora64-32k-megatron/v8-20260729-205701/checkpoint-182`
- **Merged safetensors**: `checkpoint-182-merged/model.safetensors`
- **LoRA adapter**: `checkpoint-182/adapter_model.safetensors`
- **日志**: `logs/train_qwen3.5_2b_megatron_32k_20260729_205626.log`

## 与 Round 1 对比
| | Round 1 (OpenAI) | Round 2 (Old) |
|---|---|---|
| 数据源 | 原生 JSON (37 files) | Parquet 转换 |
| 样本数 | 744 | 731 |
| 初始 loss | 4.20 | 4.16 |
| 最终 loss | 0.03 | 0.03 |
| 总步数 | 186 | 182 |
| 总时间 | ~1.5h | ~1.5h |
