#!/bin/bash
# Install mcore-bridge required for swift Megatron backend
set -euo pipefail
/data2/cyr/venvs/verl/.venv/bin/pip install mcore-bridge
echo "=== Checking installation ==="
python3 -c "from mcore_bridge.model import get_mcore_model_type; print('mcore-bridge OK')"
