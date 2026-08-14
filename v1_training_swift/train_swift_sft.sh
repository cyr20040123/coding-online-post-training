#!/bin/bash
# =============================================================================
# ms-swift SFT Training: Qwen3.5-2B + LoRA (r=64) — JiuwenSwarm Resolved
# =============================================================================
# Matches ../v1_training_verl/train_qwen3.5_2b_sft_verl.yaml hyperparameters
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_swift_sft.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

# ── GPU Selection ────────────────────────────────────────────────────────────
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_FILE="${SCRIPT_DIR}/data/jiuwen_sft_train.jsonl"

# ── Logging ──────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_qwen3.5_2b_swift_${TIMESTAMP}.log"

echo "==============================================" | tee -a "${LOG_FILE}"
echo "ms-swift SFT Training: Qwen3.5-2B + LoRA (r=64)" | tee -a "${LOG_FILE}"
echo "Start time: $(date)" | tee -a "${LOG_FILE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Hyperparameters:" | tee -a "${LOG_FILE}"
echo "  model: Qwen/Qwen3.5-2B" | tee -a "${LOG_FILE}"
echo "  lora_r=64, lora_alpha=64" | tee -a "${LOG_FILE}"
echo "  max_length=8192, lr=1e-4, epochs=4" | tee -a "${LOG_FILE}"
echo "  per_device_batch=1, gradient_accum=16, global_batch=16" | tee -a "${LOG_FILE}"
echo "  bf16=true, gradient_checkpointing=true" | tee -a "${LOG_FILE}"
echo "==============================================" | tee -a "${LOG_FILE}"

# ── Verify data ──────────────────────────────────────────────────────────────
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data not found: ${DATA_FILE}" | tee -a "${LOG_FILE}"
    echo "Run: python3 convert_data.py --input ../v1_training_verl/data/jiuwen_sft_train.parquet --output ${DATA_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi
SAMPLE_COUNT=$(wc -l < "${DATA_FILE}")
echo "Training data: ${DATA_FILE} (${SAMPLE_COUNT} samples)" | tee -a "${LOG_FILE}"

# ── NCCL ─────────────────────────────────────────────────────────────────────
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=true

# ms-swift 使用 PEFT 后端 (默认 peft) 进行 LoRA 训练
# 也支持 --tuner_backend unsloth 加速

echo "" | tee -a "${LOG_FILE}"
echo ">>> Starting ms-swift SFT training..." | tee -a "${LOG_FILE}"

PYTHONUNBUFFERED=1 swift sft \
    --tuner_backend peft \
    --model Qwen/Qwen3.5-2B \
    --dataset "${DATA_FILE}" \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --max_length 8192 \
    --truncation_strategy right \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --output_dir "${SCRIPT_DIR}/output/qwen3.5-2b-lora64" \
    --dataloader_num_workers 4 \
    --seed 42 \
    --model_author Qwen \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "==============================================" | tee -a "${LOG_FILE}"
echo "Training finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Output: ${SCRIPT_DIR}/output/qwen3.5-2b-lora64" | tee -a "${LOG_FILE}"
echo "==============================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
