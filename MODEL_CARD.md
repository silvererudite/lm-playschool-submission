---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
tags:
  - lm-playschool
  - clembench
  - playpen
  - sft
  - lora
  - qwen
language:
  - en
  - zh
library_name: peft
---

# Qwen3.5-2B SFT for LM Playschool Challenge (iter 2)

A LoRA SFT fine-tune of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) on the success-filtered split of [colab-potsdam/playpen-data](https://huggingface.co/datasets/colab-potsdam/playpen-data), submitted to the [LM Playschool Challenge](https://lm-playschool.github.io/challenge/).

**Model:** [Shamima/lm-playschool-qwen3.5-2b-sft-iter2](https://huggingface.co/Shamima/lm-playschool-qwen3.5-2b-sft-iter2)
**Code + run history:** [github.com/silvererudite/lm-playschool-submission](https://github.com/silvererudite/lm-playschool-submission)

## Headline results

| | clemscore | statscore |
|---|---|---|
| `Qwen/Qwen3.5-2B` (leaderboard baseline) | 13.05 | 44.02 |
| **This model** | **46.66** | **40.39** |
| **Δ** | **+33.61** | **−3.63** |

Evaluated with `playpen eval --suite all`. Per-game and per-benchmark breakdowns in [`results/iter2.md`](https://github.com/silvererudite/lm-playschool-submission/blob/main/results/iter2.md).

For context, the published leaderboard baselines for the same eval suite are:

| Model | clemscore | statscore |
|---|---|---|
| Qwen3.5-27B | 65.72 | 65.99 |
| Qwen3.5-9B | 41.92 | 54.16 |
| **This model (2B SFT)** | **46.66** | 40.39 |
| Qwen3.5-4B | 39.70 | 50.63 |
| Llama-3.1-8B | 25.67 | 45.81 |
| Qwen3.5-2B (base) | 13.05 | 44.02 |

## Training methodology

**SFT with LoRA**, completion-only loss disabled (effectively standard language-modeling loss across the full chat-templated sequence), single epoch over success-filtered episodes. Standard hyperparameters with two non-default choices that were critical to the result (see "Design decisions" below).

### Data usage

- **Source:** `colab-potsdam/playpen-data`, `interactions` split, train portion (34,909 episodes total).
- **Filter:** kept only episodes with `meta.outcome == "success"` → **20,202 episodes**.
- **Eval split:** none. Iter 1 used a 20% holdout for in-training eval; iter 2 disabled in-training eval (it OOMed on A10G; eval-mode batches don't use gradient checkpointing). All 20,202 episodes used for training.
- **Tokenization:** Qwen's chat template applied. Sequences truncated to `max_length=1024`, which covers ~82% of the dataset uncut (verified empirically — see [`scripts/dataset_length_stats.py`](https://github.com/silvererudite/lm-playschool-submission/blob/main/scripts/dataset_length_stats.py); iter 1 used 300, which truncated 64% of episodes — see iter-1 writeup).
- **No external data, no test contamination.** Only the public `playpen-data` train split was used.

### Hyperparameters

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` (instruct, thinking-mode off) |
| Method | LoRA SFT, full chat-template language-modeling loss |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Target modules | `all-linear` |
| **`modules_to_save`** | `[]` (empty — see "Design decisions" #1) |
| Optimizer | TRL `SFTTrainer` defaults (AdamW) |
| Learning rate | 2e-4 |
| LR schedule | cosine, warmup_ratio=0.03 |
| Weight decay | 0.0 |
| Batch (per device × grad-accum × ranks) | 1 × 2 × 4 = **effective 8** |
| Sequence length | 1024 |
| Precision | bf16 mixed |
| Gradient checkpointing | enabled (required to fit `seq_len=1024` on A10G) |
| Epochs | 1 |
| Total steps | 2526 |
| Save | every 1000 steps, `save_total_limit=2` |
| Eval during training | disabled (`eval_strategy="no"`) |

### Compute budget

| | |
|---|---|
| Hardware | 4× NVIDIA A10G (23 GB each) |
| Parallelism | DDP via `accelerate launch --num_processes 4` |
| Wall-clock training | **2 h 36 m** |
| Final train_loss | 0.2207 |
| Final mean_token_accuracy | 0.9161 |
| Tokens trained on | 10.86M (`num_tokens` reported by trainer) |
| Eval wall-clock (full suite, baseline + SFT) | ~9 h cumulative |
| Total compute | ~12 GPU-hours train + ~9 GPU-hours eval ≈ **21 A10G-hours** |

### Reproducibility

The repo at [github.com/silvererudite/lm-playschool-submission](https://github.com/silvererudite/lm-playschool-submission) contains:

- `scripts/setup.sh` — clones playpen + clembench, builds venv, installs deps, applies our config patches. Idempotent.
- `scripts/train_sft.py` — the full trainer (CONFIG block at top exposes every hyperparameter).
- `scripts/merge_and_push.py` — LoRA merge + HF Hub upload.
- `scripts/dataset_length_stats.py` — episode-length analysis used to pick `max_length`.
- `scripts/post_train_pipeline.sh` — end-to-end orchestrator (waits on training pid → merge → push → eval).
- `RUNS/iter1.md` — iter 1 writeup with the failure analysis that motivated iter 2's hyperparameters.
- `results/iter2.md` — full per-game / per-benchmark breakdown of this model.

To reproduce iter 2 exactly:

```bash
git clone https://github.com/silvererudite/lm-playschool-submission && cd lm-playschool-submission
bash scripts/setup.sh
# paste your HF token into playpen/key.json
source env.sh
python -m accelerate.commands.launch --num_processes 4 --num_machines 1 \
    --mixed_precision bf16 \
    "$(which playpen)" run scripts/train_sft.py -l Qwen3.5-2B
```

## Design decisions

These are the choices that mattered most (deltas vs the upstream `examples/trl/sft_trainer_lora.py` defaults):

1. **`modules_to_save=[]` instead of upstream's `["lm_head", "embed_token"]`.** This was the load-bearing fix between iter 1 (regression to clemscore 3.86) and iter 2 (gain to 46.66). Reasons:
   - **Typo bug**: the upstream string is `embed_token` (singular). Qwen's actual module is `embed_tokens` (plural), so peft silently ignored that entry — only `lm_head` was made trainable.
   - **Tied-weight breakage**: Qwen3.5 ships with `tie_word_embeddings=True` (the input embedding *is* the output head). Adding `lm_head` to `modules_to_save` causes peft to set `tie_word_embeddings=False` and create a separate trainable copy. After thousands of steps the head drifts away from the still-tied embedding, breaking an architectural invariant the model was pretrained against.
   - With `modules_to_save=[]`, training is pure low-rank LoRA — no full-rank parameters touched, tied weights preserved.

2. **`max_length=1024` instead of upstream's `300`.** Empirical: median episode length in the success-filtered dataset is 462 tokens, p75 is 814, p90 is 1373. At 300, ~64% of training sequences had their assistant turn truncated; with TRL's effective full-sequence loss, that's ~64% of training compute spent on prefixes that never reached the target. Bumping to 1024 covers ~82% of episodes uncut. 2048 would cover ~95% but costs ~4× attention memory; we left the headroom for iter 3.

3. **`eval_strategy="no"` (no in-training eval).** TRL's eval loop disables gradient checkpointing for eval batches. On 4× A10G this OOMed at the first epoch boundary (iter 2's first attempt crashed at step 2021/6063 trying to allocate +7.58 GB on top of 15.85 GB already in use). We measure model quality via the full clembench eval after training, so eval_loss during training adds nothing.

4. **`save_strategy="steps"`, `save_steps=1000`, `save_total_limit=2`.** Crash-recovery insurance. With `save_strategy="epoch"` the first checkpoint wouldn't appear until ~1.5 h in; on shared infrastructure that's a long uninsured stretch.

5. **DDP via accelerate** with a small monkeypatch in `train_sft.py`: `clemcore`'s huggingface backend hardcodes `device_map="auto"`, which pipeline-shards the model across all visible GPUs and runs only one GPU at a time. Under accelerate (`LOCAL_RANK` set), we override `AutoModelForCausalLM.from_pretrained` to use `device_map={"": LOCAL_RANK}` so each rank holds a full model copy on its own GPU. Single-process pipeline-parallel iter 1: 4 h. 4-GPU DDP iter 2: 2 h 36 m.

6. **Generation-config fix on the merged model.** Qwen3.5-2B's `text_config.eos_token_id` is `248044` (`<|endoftext|>`), but its chat template terminates with `<|im_end|>` (`248046`). After SFT, the model emitted `<|im_end|>` correctly but `model.generate()` didn't stop there (wrong eos_token_id), causing the model to "continue" past its turn and hallucinate fake user/assistant exchanges. Fixed by setting `eos_token_id: [248046, 248044]` and `pad_token_id: 248044` in `generation_config.json` of the uploaded model. *This is a property of the model, not an eval-pipeline trick.*

## Limitations and known regressions

- **referencegame quality 100 → 8.33**: largest per-game regression. SFT learned a verbose explanation style from playpen-data that hurts on the single-turn exact-match protocol of referencegame.
- **ifeval 65.22 → 36.96, eqbench 65.01 → 46.77**: SFT softened the model's instruction-following discipline. The base model's strict format adherence on these benches partially eroded.
- **wordle unchanged at 33.33 % played, 0 quality**: the model now follows wordle's protocol but doesn't have the world-model to actually win wordle.

## License

Apache 2.0 (inherited from Qwen3.5-2B and from the LoRA-only adapter on top).
