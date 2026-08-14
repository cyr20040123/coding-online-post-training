# Round 3: OpenAI Data 64K SFT

## 基本信息
- **数据文件**: `data/jiuwen_openai_train.jsonl` (744 samples)
- **开始时间**: 2026-07-29 22:34:50 CST
- **完成时间**: 2026-07-30 00:15:51 CST
- **耗时**: ~1h 41min
- **退出码**: 0 (成功)

## 配置
| 参数 | 值 |
|---|---|
| 命令 | `megatron pt` |
| 模型 | Qwen/Qwen3.5-2B |
| 上下文 | **64K** (`max_length=65536`) |
| 并行 | TP=1 PP=1 CP=8 DP=1 (8K tokens/GPU) |
| 训练类型 | LoRA r=64, alpha=64, all-linear |
| Batch | micro=1, global=16 |
| Epochs | 4 |
| LR | 1e-4, cosine schedule, warmup=10% |
| Loss scale | `last_round+ignore_empty_think` |
| 模板 | `agent_template=qwen3_5`, `template_backend=jinja` |
| 精度 | bf16 |
| **CE Fusion** | **TE fused** (`--cross_entropy_loss_fusion --cross_entropy_fusion_impl te`) |
| 显存优化 | `recompute=selective`, `expandable_segments:True` |
| 设备 | 8× RTX 3090 (24GB) |

## Loss 变化

| Checkpoint | Step | Loss | 说明 |
|---|---|---|---|
| — | 1 | 3.66 | 初始 |
| checkpoint-46 | 46 | 0.33 | Epoch 1 完成 |
| checkpoint-92 | 92 | 0.12 | Epoch 2 完成 |
| — | 93 | 0.10 | Epoch 2 → 3 过渡 |
| checkpoint-138 | 138 | 0.05 | Epoch 3 完成 |
| — | 139 | 0.06 | Epoch 3 → 4 过渡 |
| checkpoint-184 | 184 | 0.06 | Epoch 4 末 |
| checkpoint-186 | 186 | **0.03** | 最终 |

## 输出
- **Checkpoint**: `output/qwen3.5-2b-lora64-64k-megatron/v1-20260729-223526/checkpoint-186`
- **日志**: `logs/train_qwen3.5_2b_megatron_32k_20260729_223450.log`

## 关键发现
- 64K 上下文在 8x24GB 上**可以训练**，但需要：
  1. `CP=8` (每 GPU 8K tokens)
  2. `cross_entropy_loss_fusion=true` + `cross_entropy_fusion_impl=te` (TE fused CE)
  3. `expandable_segments:True`
- 不使用 TE fused CE 时，`vocab_parallel_logits.float()` 分配 ~5.5GB 导致 OOM
- TE fused CE 避免了显式 logits 物化，节省了关键显存
