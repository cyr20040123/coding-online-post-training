# Round 8: Swift Data v2 (no fix) 48K SFT

## 基本信息
- **数据文件**: `data/jiuwen_swift_train_v2.jsonl` (704 samples, with tools, 无需 tool→user 修复)
- **开始时间**: 2026-07-30 19:11 CST
- **完成时间**: 2026-07-30 21:03 CST
- **耗时**: ~1h52min
- **退出码**: 0

## 配置
| 参数 | 值 |
|---|---|
| 命令 | `megatron sft` |
| 模型 | Qwen/Qwen3.5-2B |
| 上下文 | 48K (`max_length=49152`) |
| 并行 | TP=1 PP=1 CP=8 DP=1 |
| 训练类型 | LoRA r=64, alpha=64, all-linear |
| Batch | micro=1, global=16 |
| Epochs | 3 |
| LR | 1e-4, cosine schedule, warmup=10% |
| Loss scale | `last_round+ignore_empty_think` |
| 后端/模板 | `template_backend=swift`, `agent_template=qwen3_5` |
| CE Fusion | TE fused |
| 设备 | 8× RTX 3090 (24GB) |

## Loss 变化
| Epoch | Start | End |
|---|---|---|
| 1 | 0.646 | 0.137 |
| 2 | 0.137 | 0.035 |
| 3 | 0.035 | 0.013 |

## 输出
- **Checkpoint**: `v9-20260730-191140/checkpoint-132`
- **日志**: `logs/train_qwen3.5_2b_megatron_32k_20260730_191106.log`

## 与 Round 7 对比
| | Round 7 (v1+fix) | Round 8 (v2) |
|---|---|---|
| 数据 | 需要 tool→user 修复 | 原生支持 |
| 最终 loss | 0.19 | 0.01 |
| 原因 | v1用tool→assistant(空)→user，空assistant引入噪声 | v2天然符合Swift格式 |
