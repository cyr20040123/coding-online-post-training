# Round 7: Swift Data (LSP tools) 48K SFT

## 基本信息
- **数据文件**: `data/jiuwen_swift_train_swift.jsonl` (704 samples, with tools)
- **开始时间**: 2026-07-30 17:08 CST
- **完成时间**: 2026-07-30 19:10 CST
- **耗时**: ~2h
- **退出码**: 0

## 配置
| 参数 | 值 |
|---|---|
| 命令 | `megatron sft` |
| 模型 | Qwen/Qwen3.5-2B |
| 上下文 | 48K (`max_length=49152`) |
| 并行 | TP=1 PP=1 CP=8 DP=1 (6K tokens/GPU) |
| 训练类型 | LoRA r=64, alpha=64, all-linear |
| Batch | micro=1, global=16 |
| Epochs | 3 |
| LR | 1e-4, cosine schedule, warmup=10% |
| Loss scale | `last_round+ignore_empty_think` |
| 后端/模板 | `template_backend=swift`, `agent_template=qwen3_5` |
| CE Fusion | TE fused |
| 设备 | 8× RTX 3090 (24GB) |
| 数据处理 | tool→user 插入空 assistant（681处） |

## Loss 变化
| Epoch | Start | End |
|---|---|---|
| 1 | 0.619 | 0.605 |
| 2 | 0.605 | 0.445 |
| 3 | 0.445 | 0.191 |

## 输出
- **Checkpoint**: `v8-20260730-170839/checkpoint-139`
- **日志**: `logs/train_qwen3.5_2b_megatron_32k_20260730_170806.log`

## 备注
- 每条数据带 27 个 LSP 工具定义，序列约 20K tokens，训练速度较慢（~50s/it）
- 数据需要 tool→user 修复才能通过 Swift 后端
