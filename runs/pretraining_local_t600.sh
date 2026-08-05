#!/bin/bash

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

source .venv/bin/activate
python -m scripts.base_train \
  --depth=2 \
  --max-seq-len=256 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --num-iterations=100 \
  --eval-tokens=256 \
  --core-metric-every=-1 \
  --sample-every=50 \
  --window-pattern=L

