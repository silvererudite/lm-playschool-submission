#!/usr/bin/env bash
# Post-training pipeline:
#   1. wait for the SFT process to exit
#   2. find the latest checkpoint
#   3. update model_registry's peft_model path
#   4. merge + push to HF Hub
#   5. run baseline eval (Qwen3.5-2B)
#   6. run SFT eval (Qwen3.5-2B-sft)
#   7. compute deltas via scripts/eval.py
#
# Usage:
#   bash scripts/post_train_pipeline.sh <sft_pid> <hf_repo>
#
# Example:
#   bash scripts/post_train_pipeline.sh 8441 silvererudite/lm-playschool-qwen3.5-2b-sft

set -euo pipefail

ROOT=/home/sagemaker-user/lm-playschool-submission
SFT_PID="${1:?need sft pid as $1}"
HF_REPO="${2:?need hf repo as $2}"
LOG="$ROOT/logs/post-train.log"
mkdir -p "$ROOT/logs" "$ROOT/results"

exec >"$LOG" 2>&1

echo "===== START $(date -Iseconds) ====="
echo "[post-train] watching pid=$SFT_PID, repo=$HF_REPO"

# 1. wait
while kill -0 "$SFT_PID" 2>/dev/null; do
    sleep 60
done
echo "[post-train] SFT process $SFT_PID exited at $(date -Iseconds)"

# Verify training actually finished successfully (look for failure in log).
if grep -qE "exit=[1-9]|ChildFailedError|RuntimeError" "$ROOT/logs/sft-lora.log"; then
    echo "[post-train] SFT log contains failure markers — aborting"
    tail -30 "$ROOT/logs/sft-lora.log"
    exit 1
fi

# shellcheck disable=SC1091
source "$ROOT/env.sh" >/dev/null

# 2. find latest checkpoint
CKPT_PARENT="$ROOT/playpen/models/sft+lora/Qwen3.5-2B"
CKPT=$(ls -d "$CKPT_PARENT"/checkpoint-* 2>/dev/null \
       | awk -F'checkpoint-' '{print $2"\t"$0}' \
       | sort -n | tail -1 | cut -f2)
if [ -z "$CKPT" ]; then
    echo "[post-train] no checkpoints under $CKPT_PARENT — abort"
    exit 1
fi
echo "[post-train] using checkpoint: $CKPT"
REL_CKPT="models/sft+lora/Qwen3.5-2B/$(basename "$CKPT")"

# 3. update registry's peft_model path
python - <<PY
import json, pathlib
p = pathlib.Path("$ROOT/playpen/model_registry.json")
data = json.loads(p.read_text())
for entry in data:
    if entry["model_name"] == "Qwen3.5-2B-sft":
        entry["model_config"]["peft_model"] = "$REL_CKPT"
        break
p.write_text(json.dumps(data, indent=2) + "\n")
print(f"[post-train] registry updated: peft_model -> $REL_CKPT")
PY

# 4. merge + push
python "$ROOT/scripts/merge_and_push.py" \
    --base Qwen/Qwen3.5-2B \
    --adapter "$CKPT" \
    --repo "$HF_REPO"

# 5+6+7. eval baseline and SFT, write comparison.
# Output filename derives from the HF_REPO basename so successive iters don't
# overwrite each other. e.g. Shamima/lm-playschool-qwen3.5-2b-sft-iter2 ->
# results/lm-playschool-qwen3.5-2b-sft-iter2.md.
OUT_FILE="$ROOT/results/$(basename "$HF_REPO").md"
python "$ROOT/scripts/eval.py" \
    --suite all \
    --model Qwen3.5-2B \
    --model Qwen3.5-2B-sft \
    --baseline Qwen3.5-2B \
    --out "$OUT_FILE"

echo "===== END $(date -Iseconds) ====="
