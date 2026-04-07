#!/bin/bash
# Ensure PYTHONPATH includes ms_pred source
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Ensure conda env's python3 is used by subprocess
export PATH="$(dirname $(which python)):$PATH"

python launcher_scripts/run_from_config.py configs/clip/train_msg_unimol.yaml
