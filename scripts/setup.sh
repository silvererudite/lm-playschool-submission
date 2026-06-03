#!/usr/bin/env bash
# One-shot setup for a fresh checkout of this repo.
# Clones playpen + clembench (which are gitignored), creates a venv,
# installs deps, applies our config overrides, and points caches at NVMe.
#
# Usage:  bash scripts/setup.sh
# Prereq: python>=3.10, git, ~10GB free under /mnt/sagemaker-nvme (or edit NVME below).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NVME="${NVME:-/mnt/sagemaker-nvme/lm-playschool}"

echo "[setup] ROOT=$ROOT"
echo "[setup] NVME=$NVME"

mkdir -p "$NVME"/{hf-cache,models,playpen-eval,datasets,pip-cache,tmp}

# --- clone upstream repos (gitignored in our repo) ---
if [ ! -d "$ROOT/playpen" ]; then
  git clone https://github.com/lm-playpen/playpen.git "$ROOT/playpen"
fi
if [ ! -d "$ROOT/clembench" ]; then
  git clone https://github.com/clp-research/clembench.git "$ROOT/clembench"
fi

# --- venv + deps ---
cd "$ROOT/playpen"
if [ ! -d venv ]; then
  python -m venv venv --system-site-packages
fi
# shellcheck disable=SC1091
source venv/bin/activate

export PIP_CACHE_DIR="$NVME/pip-cache"
export TMPDIR="$NVME/tmp"
pip install --no-input -e .
pip install --no-input -r "$ROOT/clembench/requirements.txt"
pip install --no-input '.[trl]'
pip install --no-input 'clemcore[huggingface]'

# --- apply our config overrides ---
# game_registry: replace the placeholder
cp "$ROOT/config/game_registry.json" "$ROOT/playpen/game_registry.json"

# model_registry: prepend our Qwen3.5-2B entries to the upstream registry, idempotent
python "$ROOT/scripts/merge_model_registry.py" \
  --upstream "$ROOT/playpen/model_registry.json" \
  --add "$ROOT/config/qwen3.5-2b.entries.json"

# key.json: copy from template if absent (user must fill in HF token)
if [ ! -f "$ROOT/playpen/key.json" ]; then
  cp "$ROOT/playpen/key.json.template" "$ROOT/playpen/key.json"
  echo "[setup] WARN: created playpen/key.json from template — paste your HF token in 'huggingface.api_key'"
fi

# symlink models/ + playpen-eval/ to NVMe so /home doesn't fill
[ -e "$ROOT/playpen/models" ] || ln -s "$NVME/models" "$ROOT/playpen/models"
[ -e "$ROOT/playpen/playpen-eval" ] || ln -s "$NVME/playpen-eval" "$ROOT/playpen/playpen-eval"

echo "[setup] done. Next: source $ROOT/env.sh && playpen list models"
