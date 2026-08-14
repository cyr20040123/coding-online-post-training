# Round 1: OpenAI Data 32K SFT

## 基本信息
- **数据文件**: `data/jiuwen_openai_train.jsonl` (744 samples, 37 source files)
- **开始时间**: 2026-07-29 19:26:26 CST
- **完成时间**: 2026-07-29 ~21:00 CST
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
| 显存优化 | `recompute=selective`, `no_cross_entropy_loss_fusion`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 设备 | 8× RTX 3090 (24GB) |

## Loss 变化

| Checkpoint | Step | Loss | 说明 |
|---|---|---|---|
| — | 1 | 4.20 | 初始 |
| checkpoint-46 | 46 | 0.34 | Epoch 1 完成 |
| checkpoint-92 | 92 | 0.12 | Epoch 2 完成 |
| — | 93 | 0.10 | Epoch 2 → 3 过渡 |
| checkpoint-138 | 138 | 0.05 | Epoch 3 完成 |
| — | 139 | 0.05 | Epoch 3 → 4 过渡 |
| checkpoint-184 | 184 | 0.06 | Epoch 4 末 |
| checkpoint-186 | 186 | **0.03** | 最终 |

## 输出
- **Checkpoint**: `output/qwen3.5-2b-lora64-32k-megatron/v7-20260729-192659/checkpoint-186`
- **Merged safetensors**: `checkpoint-186-merged/model.safetensors` (3.6 GB)
- **LoRA adapter**: `checkpoint-186/adapter_model.safetensors` (129 MB)
- **日志**: `logs/train_qwen3.5_2b_megatron_32k_20260729_192626.log`

## 备注
- 初始 OOM 在 step 47 (vocab_parallel_logits.float())，添加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后解决
- 数据加载需要 `agent_template=qwen3_5` 处理工具调用消息
- `template_backend=jinja` 绕过 Swift 内置模板对连续 user 消息的限制
