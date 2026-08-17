#!/bin/bash
# =============================================================================
# verl SFT Training: Qwen3.5-2B + LoRA + 48K + sliding window
# 基于 v1_training_verl 已验证的 FSDP2 + Ulysses SP=4 路径
#
# Usage:
#   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_verl_sft_48k.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_FILE="${SCRIPT_DIR}/train_qwen3.5_27b_sft_npu.yaml"
DATA_FILE="${SCRIPT_DIR}/data/sft_train_sqrt.parquet"

cd "${SCRIPT_DIR}"

VERL_PYTHON="${VERL_PYTHON:-/root/data2/cyr/venvs/verl/.venv/bin/python3}"

VERL_SFT_ENTRY="${VERL_SFT_ENTRY:-verl.trainer.sft_trainer}"

if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
NUM_GPUS=$(echo "${ASCEND_RT_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "Start: $(date)  GPUs: ${NUM_GPUS}" | tee -a "${LOG_FILE}"

if [ -f "${DATA_FILE}" ]; then
    N=$("${VERL_PYTHON}" -c "import pandas as pd; print(len(pd.read_parquet('${DATA_FILE}')))" 2>/dev/null || echo "?")
    echo "Data: ${DATA_FILE} (${N} conversations)" | tee -a "${LOG_FILE}"
else
    echo "ERROR: data not found: ${DATA_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

# custom_cls 加载路径
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=true

"${VERL_PYTHON}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    -m "${VERL_SFT_ENTRY}" \
    --config-path="${SCRIPT_DIR}" \
    --config-name="train_qwen3.5_27b_sft_npu" \
    2>&1 | tee -a "${LOG_FILE}"

echo "Done: $(date)" | tee -a "${LOG_FILE}"
