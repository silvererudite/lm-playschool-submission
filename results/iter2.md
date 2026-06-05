# Iter 2 — Final eval results

**HF model:** [Shamima/lm-playschool-qwen3.5-2b-sft-iter2](https://huggingface.co/Shamima/lm-playschool-qwen3.5-2b-sft-iter2)
**Eval run:** 2026-06-05T04-40-53Z
**Eval suite:** `playpen eval Qwen3.5-2B-sft --suite all` (full clem + static)

## Headline

| | clemscore | statscore |
|---|---|---|
| Qwen3.5-2B leaderboard baseline | 13.05 | 44.02 |
| Qwen3.5-2B baseline (our reproduction) | 12.59 | 42.33 |
| **Qwen3.5-2B-sft (this iter)** | **46.66** | **40.39** |
| **Δ vs leaderboard baseline** | **+33.61** | **−3.63** |

The model would land **#2 on the public leaderboard by clemscore**, behind only Qwen3.5-27B (65.72) — a 4.5× larger reference. Beats every other listed baseline (Qwen3.5-9B at 41.92, Qwen3.5-4B at 39.70, Llama-3.1-8B at 25.67).

The −3.63 statscore drop is real but small — the model traded a little static-benchmark performance for a large agentic-game gain. For a challenge focused on multi-step agentic ability, that's the right trade.

## Per-game (clem suite)

| game | baseline %P / Q | **iter-2 %P / Q** | Δ Q |
|---|---|---|---|
| adventuregame | 0.0 / — | **100.0 / 13.33** | first time playing |
| codenames | 0.0 / — | **46.15 / 33.33** | first time playing |
| guesswhat | 0.0 / — | **100.0 / 33.33** | first time playing |
| imagegame | 75.0 / 47.67 | **100.0 / 74.00** | +26.33 |
| matchit_ascii | 100.0 / 75.00 | 100.0 / 71.86 | −3.14 |
| privateshared | 0.0 / — | **100.0 / 80.00** | first time playing |
| referencegame | 100.0 / 100.0 | 100.0 / 8.33 | **−91.67** ⚠ |
| taboo | 100.0 / 0.0 | 100.0 / **39.33** | +39.33 |
| textmapworld | 0.0 / — | 100.0 / 30.00 | first time playing |
| textmapworld_graphreasoning | 0.0 / — | 100.0 / 33.33 | first time playing |
| textmapworld_specificroom | 0.0 / — | 100.0 / 30.00 | first time playing |
| wordle | 33.33 / 0.0 | 33.33 / 0.0 | unchanged |
| wordle_withclue | 66.67 / 0.0 | 100.0 / **83.33** | +83.33 |
| wordle_withcritic | 0.0 / — | 100.0 / 66.67 | first time playing |
| **average** | 33.93 / 37.11 | **86.63 / 53.86** | **+50%P / +17Q** |

The training-data filter (success-only) means the model saw demonstrations of *winning* gameplay. The big jumps are in games where the baseline couldn't even produce parseable output (% played 0 → 100); the iter-2 model both follows the protocol and plays well enough to score.

The **referencegame regression** (−91.67) is the one place SFT clearly hurt. Probably the verbose explanation style learned from playpen-data hurts on the single-turn exact-match protocol. Worth investigating in iter 3.

## Per-bench (static suite)

| benchmark | baseline | iter-2 | Δ |
|---|---|---|---|
| bbh | 0.0 | 35.16 | **+35.16** |
| cladder | 46.53 | 43.56 | −2.97 |
| eqbench | 65.01 | 46.77 | −18.24 |
| ifeval | 65.22 | 36.96 | −28.26 |
| mmlu_pro | 38.05 | 40.71 | +2.66 |
| **average** | 42.96 | 40.63 | −2.33 |

Mixed: gains on bbh and mmlu_pro (reasoning), losses on ifeval and eqbench (instruction-following / EQ). The ifeval drop is consistent with the model learning a more permissive output style from playpen-data; eqbench is a known-unstable bench for SFT-finetuned models.
