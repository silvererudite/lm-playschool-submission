#!/usr/bin/env bash
# Source this file before running playpen: `source env.sh`

ROOT=/home/sagemaker-user/lm-playschool-submission
NVME=/mnt/sagemaker-nvme/lm-playschool

export HF_HOME="$NVME/hf-cache"
export HF_DATASETS_CACHE="$NVME/hf-cache/datasets"
export TRANSFORMERS_CACHE="$NVME/hf-cache/transformers"
export TORCH_HOME="$NVME/torch"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

# huggingface api key from playpen's key.json (so `huggingface-cli` and gated repos work)
if [ -f "$ROOT/playpen/key.json" ]; then
  HF_KEY=$(python -c "import json; print(json.load(open('$ROOT/playpen/key.json'))['huggingface']['api_key'])" 2>/dev/null)
  [ -n "$HF_KEY" ] && export HF_TOKEN="$HF_KEY" && export HUGGING_FACE_HUB_TOKEN="$HF_KEY"
fi

# checkpoints land on NVMe via symlink in playpen/models -> /mnt/sagemaker-nvme/lm-playschool/models
[ ! -e "$ROOT/playpen/models" ] && ln -s "$NVME/models" "$ROOT/playpen/models"
[ ! -e "$ROOT/playpen/playpen-eval" ] && ln -s "$NVME/playpen-eval" "$ROOT/playpen/playpen-eval"

source "$ROOT/playpen/venv/bin/activate"
cd "$ROOT/playpen"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] cwd=$(pwd) ; venv=$(which python)"
