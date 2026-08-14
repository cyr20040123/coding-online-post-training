# Round 4: Old Data 64K SFT (FAILED - OOM)

## 基本信息
- **数据文件**: `data/jiuwen_sft_train.jsonl` (731 samples)
- **启动时间**: 2026-07-30 00:16:31 CST
- **失败时间**: 2026-07-30 00:19:14 CST
- **存活时间**: ~2分43秒
- **状态**: ❌ OOM at step 0 (首次 forward)
- **训练步数**: 0
- **Checkpoint**: 无 (`last_model_checkpoint: None`)
- **原因**: 64K + 8K/GPU (CP=8) 在 24GB 显存边界，旧数据序列略长触发 OOM

## 配置（与 Round 3 完全相同）
| 参数 | 值 |
|---|---|
| 上下文 | 64K (`max_length=65536`) |
| 并行 | TP=1 PP=1 CP=8 DP=1 |
| CE Fusion | TE fused (`--cross_entropy_loss_fusion --cross_entropy_fusion_impl te`) |
| Batch | micro=1, global=16 |
| 输出目录 | `output/qwen3.5-2b-lora64-64k-megatron/v2-20260730-001706` |
| 日志 | `logs/train_qwen3.5_2b_megatron_32k_20260730_001631.log` |

## 时间线
| 时间 | 事件 |
|---|---|
| 00:16:31 | 训练脚本启动 (`bash train_swift_sft_64k_megatron.sh`) |
| 00:16:31~00:19:14 | 模型加载、权重下载、tokenization、分布式初始化 |
| ~00:19:00 | 首次 forward pass 开始 |
| 00:19:14 | OOM: `vocab_parallel_logits.float()` 尝试分配 3.24 GiB 失败 |
| 00:19:14 | torchrun 发送 SIGTERM，所有进程终止 |

## OOM 细节
```
torch.OutOfMemoryError: CUDA out of memory.
  Tried to allocate 3.24 GiB
  GPU total capacity: 23.56 GiB
  Already in use: ~20.4 GiB (allocated 19.0 GiB + reserved 0.15 GiB)
  Free at time of error: ~3.2 GiB
```
OOM 发生在 `vocab_parallel_cross_entropy` → `calculate_logits_max` → `vocab_parallel_logits.float()`。

## 分析
- Round 3 (OpenAI data, 744 samples) 64K 成功完成 186/186 steps，loss 3.66→0.03
- Round 4 (Old data, 731 samples) 相同配置在 step 0 即 OOM
- 旧数据可能因为 Parquet 转换保留了更长序列（或更多 tool call 消息），导致单条样本的 token 数接近 64K 上限
- 8×24GB 训练 64K + Qwen3.5-2B (248K vocab) 已到硬件极限，能否成功取决于数据序列长度分布的微小差异
- 如需稳定 64K 训练，需要 >24GB 显存的 GPU 或更多 GPU
