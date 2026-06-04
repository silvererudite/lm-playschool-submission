# Iteration 1 — Qwen3.5-2B + LoRA SFT (success-filtered playpen-data)

**Status:** Trained, evaluated, **NOT submitted.** SFT made the model worse than baseline.
Documented here so iter 2 can pick up cleanly.

## TL;DR — what to fix in iter 2

1. **`max_length=300` truncated most game contexts.** Most clembench transcripts are
   far longer; the model trained almost entirely on dialogue prefixes that never
   reached the assistant turn we wanted to predict. Bump to 1024 or 2048 (verify
   against per-episode length stats first).
2. **`modules_to_save=["lm_head", "embed_token"]` (the LoRA upstream default)
   touched the output head and embeddings.** This likely disrupted the chat
   template formatting Qwen relies on. Drop it.
3. **Success-only filter is a strong distribution shift.** The training set is
   "things 9B/27B teachers got right." Imitating that style at 2B parameters
   may have collapsed format adherence. Revisit: keep all outcomes, or weight
   by outcome.

After fixing those, re-run before considering anything else (lr tuning, RL,
distillation).

## What we ran

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` (instruct, thinking-mode off) |
| Method | LoRA SFT, completion-only loss |
| Trainer | `playpen/examples/trl/sft_trainer_lora.py` (upstream defaults — unmodified) |
| Dataset | `colab-potsdam/playpen-data` (`interactions` split), success-filtered |
| Eval split | 20% (seed 42) |
| Training data after filter | (count not logged — log shows trainer ran 6063 steps × 1 batch × 8 grad-accum so dataset_size = ~16,000 episodes after filter) |
| LoRA | r=16, α=32, dropout=0.05, target_modules=`all-linear`, **modules_to_save=`["lm_head", "embed_token"]`** |
| Optim | trl SFTConfig defaults (no lr override → 5e-5 default), packing=False, completion_only_loss=True |
| `max_length` | **300** (upstream default — very likely the worst hyperparameter here) |
| Epochs | 3 |
| Batch | per_device_train_batch_size=1, grad_accum=8 → effective 8 |
| Hardware | 4× A10G (23 GB each); model sharded across GPUs 1/2/3 by `device_map="auto"`, only one GPU active at a time (pipeline-parallel, not data-parallel) |
| Wall-clock | **4 h 7 m** (6063 steps × 2.45 s/step) |
| Final checkpoint | `models/sft+lora/Qwen3.5-2B/checkpoint-6063/` |

We tried 4-GPU DDP with `accelerate launch --num_processes 4` to cut the wall-clock
to ~50 min, but ran into two issues we abandoned for iter 1: clemcore's huggingface
backend hardcodes `device_map="auto"` (sharding fights DDP), and `playpen.cli` loads
trainer modules without registering them in `sys.modules` (broke `@dataclass`). Fixes
for both are in `scripts/train_sft.py` (LOCAL_RANK monkeypatch + SimpleNamespace
config). Worth resurrecting in iter 2 — see "Open issues" below.

## Results

### Headline numbers

| model | clemscore | statscore | Δ clem | Δ stat |
|---|---|---|---|---|
| `Qwen/Qwen3.5-2B` (baseline, our run) | 12.59 | 42.71 | — | — |
| `Qwen/Qwen3.5-2B` (leaderboard) | 13.05 | 44.02 | — | — |
| `Qwen3.5-2B-sft` (this iter) | **3.86** | **33.48** | **−8.73** | **−9.23** |

Our baseline is within 0.5 clemscore of the leaderboard, so the eval pipeline
itself is working correctly. The regression is real, not a measurement artifact.

### Per-game breakdown (clemscore games)

| game | baseline % played | baseline quality | sft % played | sft quality |
|---|---|---|---|---|
| adventuregame | 0.0 | — | 20.0 | 0.0 |
| codenames | 0.0 | — | 0.0 | — |
| guesswhat | 0.0 | — | 0.0 | — |
| imagegame | 75.0 | 47.67 | 0.0 | — |
| matchit_ascii | 100.0 | 75.0 | 100.0 | 0.0 |
| privateshared | 0.0 | — | 0.0 | — |
| referencegame | 100.0 | 100.0 | 100.0 | 80.0 |
| taboo | 100.0 | 0.0 | 50.0 | 0.0 |
| textmapworld | 0.0 | — | 0.0 | — |
| textmapworld_graphreasoning | 0.0 | — | 0.0 | — |
| textmapworld_specificroom | 0.0 | — | 0.0 | — |
| wordle | 0.0 | — | 0.0 | — |
| wordle_withclue | 33.33 | 0.0 | 0.0 | — |
| wordle_withcritic | 66.67 | 0.0 | 0.0 | — |

**Pattern:** SFT crashed `% played` on imagegame, wordle_withclue, wordle_withcritic
(stopped following game protocol entirely), and dropped quality on matchit_ascii
from 75 → 0. The only game where `% played` improved is adventuregame (0 → 20).
Average quality on played games is roughly halved.

### Per-benchmark breakdown (static)

| benchmark | baseline | sft | delta |
|---|---|---|---|
| bbh | 0.0 | 0.0 | 0.0 |
| cladder | 46.53 | 53.47 | +6.94 |
| eqbench | 65.01 | 44.65 | −20.36 |
| ifeval | 65.22 | 34.78 | −30.44 |
| mmlu_pro | 38.05 | 34.51 | −3.54 |

ifeval and eqbench drops are huge — these are instruction-following and
emotional-intelligence benchmarks. Strong evidence that SFT damaged the model's
chat-format adherence (which `modules_to_save=["lm_head", "embed_token"]` would do).

## Hypotheses (why it failed)

In rough confidence order:

1. **`modules_to_save` lobotomized the output distribution.** LoRA usually leaves
   `lm_head` frozen; the upstream script overrides this and trains the full
   embedding + head, which means at LoRA r=16 we only have a low-rank middle but
   full-rank output. The output likely overfit to the success-filtered token
   distribution.
2. **Truncation at 300 tokens.** Quick check: `colab-potsdam/playpen-data`
   episodes for clembench games typically run 500–4000+ tokens. With max_length=300
   most training samples were being cut before the assistant turn even started —
   meaning the loss was computed on padding or on irrelevant prefixes. We didn't
   instrument what fraction of samples were affected.
3. **Success-only is a hard mode shift.** The success episodes come predominantly
   from larger teachers. A 2B model imitating a 9B's tactical move-by-move output
   with completion-only loss will overfit to surface tokens and break game
   protocol on its own.
4. **Default lr (5e-5) too high for the LoRA + full embedding combo.**
   Each gradient update touched ~half a billion params via the embedding/head;
   that's not LoRA territory.

## Artifacts (preserved)

- **Adapter weights:** `/mnt/sagemaker-nvme/lm-playschool/models/sft+lora/Qwen3.5-2B/checkpoint-{500,1000,...,6000,6063}/` (gitignored — restore from NVMe)
- **Merged model:** `/mnt/sagemaker-nvme/lm-playschool/dist/lm-playschool-qwen3.5-2b-sft/` (4.8 GB, gitignored)
- **HuggingFace mirror:** https://huggingface.co/Shamima/lm-playschool-qwen3.5-2b-sft
- **Eval dirs (full transcripts + per-game scores):**
  - Baseline: `/mnt/sagemaker-nvme/lm-playschool/playpen-eval/2026-06-03T23-13-51/`
  - SFT: `/mnt/sagemaker-nvme/lm-playschool/playpen-eval/2026-06-04T00-58-01/`
- **Training log:** `logs/sft-lora.log` (gitignored)
- **Comparison table:** `results/iter1.md` (tracked)

## Open issues for iter 2

1. **Bump `max_length` to 1024+.** Run a length histogram first; pick a value
   that covers ~80% of episodes. Confirm in `scripts/train_sft.py` CONFIG.
2. **Drop `modules_to_save`** (use bare LoRA on `target_modules="all-linear"`).
3. **Try DDP again.** The fixes are already in `scripts/train_sft.py`:
   the `_patch_clemcore_for_ddp()` monkeypatch flips `device_map="auto"` to
   `device_map={"": LOCAL_RANK}` when `LOCAL_RANK` is set. Launch with
   `python -m accelerate.commands.launch --num_processes 4 ...`. If that still
   crashes (we hit a `clemcore.backends.utils.key_registry` ImportError last
   try), the workaround is to skip the `key_registry` import in the patch and
   read the token directly from `playpen/key.json`. Saves ~3 h per run.
4. **Compare success-only vs all-outcomes datasets.** Maybe weight by outcome
   rather than hard-filter.
5. **Lower LR or use cosine warmup.** Try lr=2e-5 with 100 warmup steps.
6. **Sanity check** at iter 2: after 100 steps, briefly eval on `taboo` only.
   If the model loses chat format, abort — don't burn another 4 h.

## How to reproduce iter 1 exactly

```bash
source env.sh
playpen run examples/trl/sft_trainer_lora.py -l Qwen3.5-2B
# 4h on 1 active A10G (rest hold shards via device_map="auto")
# checkpoint emitted at: models/sft+lora/Qwen3.5-2B/checkpoint-6063
```

## How to launch iter 2

```bash
# 1. edit scripts/train_sft.py CONFIG: max_length=1024, lora_modules_to_save=[]
# 2. either single-GPU:
playpen run scripts/train_sft.py -l Qwen3.5-2B
# 2-alt. or 4-GPU DDP (faster — see "Open issues" #3):
python -m accelerate.commands.launch --num_processes 4 --num_machines 1 \
    --mixed_precision bf16 \
    "$(which playpen)" run scripts/train_sft.py -l Qwen3.5-2B
# 3. when training exits, run merge + push + eval:
bash scripts/post_train_pipeline.sh "<sft pid>" Shamima/lm-playschool-qwen3.5-2b-sft-iter2
```
