#!/bin/bash
# =============================================================================
# Megatron SFT: Qwen3.5-2B + LoRA (r=64) — 48K Context + Context Parallel
# =============================================================================
# Uses Megatron backend (megatron pt) with context_parallel_size=8 to handle
# 48K sequences on 8×24GB GPUs. Each GPU processes 48K/8 = 8K tokens.
#
# Parallel layout (8 GPUs):
#   TP=1  PP=1  CP=8  DP=1
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_swift_sft_64k_megatron.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

# ── GPU Selection ────────────────────────────────────────────────────────────
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

if [ "${NUM_GPUS}" -ne 8 ]; then
    echo "ERROR: Megatron requires exactly 8 GPUs. Got ${NUM_GPUS}."
    exit 1
fi

# ── Parallel config ──────────────────────────────────────────────────────────
TP=1   # tensor_model_parallel_size
PP=1   # pipeline_model_parallel_size
CP=8   # context_parallel_size  (48K / 8 = 8K per GPU)
DP=$(( NUM_GPUS / (TP * PP * CP) ))  # data_parallel_size = 1

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_FILE="${SCRIPT_DIR}/data/jiuwen_swift_train_v2.jsonl"

# ── Logging ──────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_qwen3.5_2b_megatron_32k_${TIMESTAMP}.log"

echo "==============================================" | tee -a "${LOG_FILE}"
echo "Megatron SFT: Qwen3.5-2B + LoRA (r=64) — 48K Context" | tee -a "${LOG_FILE}"
echo "Start time: $(date)" | tee -a "${LOG_FILE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Parallel Layout:" | tee -a "${LOG_FILE}"
echo "  TP=${TP}  PP=${PP}  CP=${CP}  DP=${DP}" | tee -a "${LOG_FILE}"
echo "  Each GPU handles 48K/${CP} = $((49152/CP))K tokens (${CP} GPUs per sequence)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Hyperparameters:" | tee -a "${LOG_FILE}"
echo "  model: Qwen/Qwen3.5-2B" | tee -a "${LOG_FILE}"
echo "  tuner_type=lora, lora_rank=64, lora_alpha=64" | tee -a "${LOG_FILE}"
echo "  max_length=49152, lr=1e-4, epochs=3" | tee -a "${LOG_FILE}"
echo "  micro_batch=1, global_batch=16" | tee -a "${LOG_FILE}"
echo "  bf16, recompute=selective, attention=flash" | tee -a "${LOG_FILE}"
echo "==============================================" | tee -a "${LOG_FILE}"

# ── Verify data ──────────────────────────────────────────────────────────────
if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: Data not found: ${DATA_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi
SAMPLE_COUNT=$(wc -l < "${DATA_FILE}")
echo "Training data: ${DATA_FILE} (${SAMPLE_COUNT} samples)" | tee -a "${LOG_FILE}"

# ── Verify mcore-bridge ──────────────────────────────────────────────────────
python3 -c "from mcore_bridge.model import get_mcore_model_type" 2>/dev/null || {
    echo "ERROR: mcore-bridge not installed. Run: python3 -m pip install mcore-bridge"
    exit 1
}

# ── Torch distributed ────────────────────────────────────────────────────────
# Required by megatron CLI to dispatch via torch.distributed.run
export NPROC_PER_NODE=${NUM_GPUS}
export MASTER_PORT="${MASTER_PORT:-29500}"

# ── NCCL ─────────────────────────────────────────────────────────────────────
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "" | tee -a "${LOG_FILE}"
echo ">>> Starting Megatron SFT training (48K context, CP=${CP})..." | tee -a "${LOG_FILE}"

PYTHONUNBUFFERED=1 megatron sft \
    --model Qwen/Qwen3.5-2B \
    --dataset "${DATA_FILE}" \
    --max_length 49152 \
    --truncation_strategy right \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --agent_template qwen3_5 \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --num_train_epochs 3 \
    --lr=1e-4 \
    --lr_decay_style cosine \
    --lr_warmup_fraction 0.1 \
    --min_lr 0.0 \
    --weight_decay 0.01 \
    --clip_grad 1.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --bf16 True \
    --recompute_granularity selective \
    --attention_backend flash \
    --finetune \
    --loss_scale last_round+ignore_empty_think \
    --no_save_optim \
    --merge_lora False \
    --cross_entropy_loss_fusion \
    --cross_entropy_fusion_impl te \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --output_dir "${SCRIPT_DIR}/output/qwen3.5-2b-lora64-64k-megatron" \
    --dataloader_num_workers 4 \
    --seed 42 \
    --report_to tensorboard \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "${LOG_FILE}"
echo "==============================================" | tee -a "${LOG_FILE}"
echo "Training finished at: $(date)" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Output: ${SCRIPT_DIR}/output/qwen3.5-2b-lora64-64k-megatron" | tee -a "${LOG_FILE}"
echo "==============================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
