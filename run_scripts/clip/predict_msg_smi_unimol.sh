#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(cd /home/weiwentao/workspace/ms-pred && pwd)/src"
export PATH="$(dirname $(which python)):$PATH"
cd /home/weiwentao/workspace/ms-pred
python launcher_scripts/run_from_config.py configs/clip/predict_msg_smi_unimol.yaml
