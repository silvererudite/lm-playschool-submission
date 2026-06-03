# lm-playschool-submission

Submission for the [LM Playschool Challenge](https://lm-playschool.github.io/challenge/) —
training small LLMs to be better agents via interactive dialogue games.

## Layout

```
.
├── env.sh                      # source before any playpen command
├── scripts/
│   ├── setup.sh                # one-shot: clones upstream, installs deps, applies config
│   └── merge_model_registry.py # idempotent registry patcher
└── config/
    ├── qwen3.5-2b.entries.json # our model_registry additions
    └── game_registry.json      # paths the game registry resolves to
```

`playpen/` and `clembench/` are upstream repos and are **not** tracked here —
`scripts/setup.sh` clones them into the working tree at install time.
Model checkpoints, eval results, and HF caches live on a separate disk
(`/mnt/sagemaker-nvme/lm-playschool/` by default; override with `NVME=...`).

## Quickstart

```bash
bash scripts/setup.sh
# paste your HuggingFace token into playpen/key.json -> huggingface.api_key
source env.sh
playpen list models           # sanity check
playpen eval Qwen3.5-2B -g taboo  # smoke test on one game
```

## Status

See the issue tracker for the live task list. First-attempt strategy:
- **Base model:** Qwen3.5-2B (leaderboard baseline: clemscore 13.05, statscore 44.02).
- **Training:** LoRA SFT on success-filtered `colab-potsdam/playpen-data`.
- **Submission window:** 2026-06-22 → 2026-07-05.

## Reproducing an eval

```bash
source env.sh
playpen eval Qwen3.5-2B --suite all
# results land in playpen-eval/<timestamp>/Qwen3.5-2B.val.json (symlinked to NVMe)
```
