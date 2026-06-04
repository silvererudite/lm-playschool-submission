# Iteration 1 — Qwen3.5-2B + LoRA SFT (success-filtered playpen-data)

**Status:** Trained, evaluated, **NOT submitted.** SFT made the model worse than baseline.
Documented here so iter 2 can pick up cleanly.

## TL;DR — root causes (verified, not guesses)

1. **Upstream typo + tied-weight breakage (the destructive bug).**
   `playpen/examples/trl/sft_trainer_lora.py` line 51 sets
   `modules_to_save=["lm_head", "embed_token"]`. Qwen's module is named
   `embed_tokens` (plural) — peft silently ignored the typo'd name and *only*
   made `lm_head` trainable. But Qwen3.5-2B has `tie_word_embeddings: True`
   (input embedding == output head, same tensor). Adding `lm_head` to
   `modules_to_save` made peft set `tie_word_embeddings=False` and create a
   separate trainable copy of `lm_head`. After 6063 steps, the output head
   drifted from the still-original-tied embedding; at inference, the model is
   using a *different* lm_head than the embedding it was pretrained against.
   This breaks an architectural invariant Qwen depends on, which explains why
   the model loses chat-format adherence.

   *Confirmed by inspecting the merged model on HF: `tie_word_embeddings: False`
   in our config, `True` in the base. peft's merge step also emitted the
   warning "Setting `tie_word_embeddings=False` in the model config" — visible
   in `logs/post-train.log`.*

2. **`max_length=300` truncated 64% of training episodes.**
   Measured episode lengths (after applying Qwen's chat template, n=500
   sample of the success-filtered split):

   ```
   median: 462    p75: 814    p90: 1373    p95: 2020    max: 8962
   over 300 tokens: 318/500 = 63.6%
   over 1024 tokens: 88/500 = 17.6%
   ```

   With `completion_only_loss=True`, two-thirds of samples were truncated
   *before reaching the assistant turn we wanted to predict*. Most training
   steps got near-zero useful gradient. This made training inefficient but
   isn't destructive on its own — problem (1) is.

3. **Success-only filter is a 1.7× shrink (34,909 → 20,202 episodes), unverified.**
   Possibly fine on its own. Not the immediate culprit. Revisit only after
   fixing (1) and (2).

## Iter 2 fixes (in priority order)

1. **Stop training the tied weight.** Drop `modules_to_save` entirely:
   ```python
   peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM")
   ```
   Or, if we want to train embeddings, use the *correct* name `embed_tokens`
   AND keep `tie_word_embeddings=True` after merging.
2. **Bump `max_length` to 1024.** Covers ~82% of episodes uncut. Going higher
   (2048 = ~95% covered) costs ~4× more memory at attention; may force
   batch_size=1 with grad accum.
3. (Defer) success-filter and lr tuning until 1 + 2 are validated.

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

## Root cause (verified post-debug)

See "TL;DR — root causes" at the top of this file. Two confirmed problems:

1. **The destructive bug** is the upstream `embed_token` typo in
   `playpen/examples/trl/sft_trainer_lora.py` line 51 combined with Qwen's
   tied embeddings. peft set `tie_word_embeddings: False` and trained a
   separate `lm_head`, leaving the embedding side pinned to base. After 6063
   steps the head drifted away from the embedding it's meant to be tied to.
   This explains the format collapse on imagegame, wordle_*, ifeval, and
   eqbench (all of which require strict output-token discipline).
2. **The inefficiency multiplier** is `max_length=300`. Empirically, 64% of
   success-filtered episodes are longer than 300 tokens — for those, with
   `completion_only_loss=True`, the loss target was truncated and we got no
   useful gradient. Two-thirds of compute wasted.

The combination — destructive (1) + 64% wasted compute (2) — is what got us
from "small SFT improvement" to "−8.7 clemscore."

## Artifacts

⚠ **`/mnt/sagemaker-nvme/` is ephemeral on this SageMaker instance.** It got
wiped between sessions (instance restart). All checkpoints, eval transcripts,
training logs that lived on NVMe are gone. Anything not in git or on
HuggingFace was lost.

Surviving:
- **HuggingFace mirror:** https://huggingface.co/Shamima/lm-playschool-qwen3.5-2b-sft
  (4.8 GB merged model + tokenizer + provenance JSON)
- **In git:** `RUNS/iter1.md`, `results/iter1.md`, `results/iter1/*.csv`,
  `results/iter1/*.val.json`
- **In `/home/sagemaker-user/`:** `logs/sft-lora.log` (training output),
  the project repo, the venv, `key.json`

For iter 2: if we want diagnostic transcripts to compare against, write a
small recurring sync to push them to S3 or to git-lfs. The eval dir is
~hundreds of MB so git-lfs is reasonable for a sample.

## Open issues for iter 2

1. **Drop `modules_to_save` entirely** (or fix the `embed_token` →
   `embed_tokens` typo *and* keep `tie_word_embeddings=True` after merge).
   This is the single change most likely to flip the regression to a gain.
2. **Bump `max_length` to 1024.** Already verified: this covers ~82% of
   episodes (vs 36% at 300). Going to 2048 covers ~95% but ~4× attention
   memory; check it fits before committing.
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
