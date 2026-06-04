"""Measure token-length distribution of the SFT dataset under a given tokenizer.

Use this BEFORE choosing `max_length` for SFT, to understand how much will be
truncated. Iter 1 picked max_length=300 blind and lost 64% of episodes.

Usage:
    source env.sh
    python scripts/dataset_length_stats.py \\
        [--dataset colab-potsdam/playpen-data] \\
        [--subset interactions] \\
        [--tokenizer Qwen/Qwen3.5-2B] \\
        [--success-only] \\
        [--n 500]

Output: percentile table + truncation rates at common max_length values.
"""

from __future__ import annotations

import argparse
import random
import statistics

from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="colab-potsdam/playpen-data")
    ap.add_argument("--subset", default="interactions")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--success-only", action="store_true",
                    help="filter to outcome==success (matches iter 1 trainer)")
    ap.add_argument("--n", type=int, default=500,
                    help="sample N episodes (None = all, slow)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[load] {args.dataset} / {args.subset} / {args.split}")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    print(f"[load] total episodes: {len(ds)}")

    if args.success_only:
        ds = ds.filter(lambda ep: ep["meta"]["outcome"] == "success")
        print(f"[filter] success episodes: {len(ds)}")

    sample_n = min(args.n, len(ds)) if args.n else len(ds)
    random.seed(args.seed)
    sample = ds.select(random.sample(range(len(ds)), sample_n))

    print(f"[load] tokenizer {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"\n[measure] tokenizing {sample_n} episodes (chat template applied)")
    lengths = []
    for ep in sample:
        rendered = tok.apply_chat_template(ep["messages"], tokenize=False)
        lengths.append(len(tok(rendered, add_special_tokens=False)["input_ids"]))

    lengths.sort()
    n = len(lengths)
    print(f"\n[stats] n={n}")
    print(f"[stats] min={lengths[0]}  median={lengths[n // 2]}  "
          f"mean={statistics.mean(lengths):.0f}  max={lengths[-1]}")

    print("\n[percentiles]")
    for p in (10, 25, 50, 75, 90, 95, 99):
        idx = int(n * p / 100)
        print(f"  p{p:>2} = {lengths[idx]}")

    print("\n[truncation rates]")
    for max_len in (300, 512, 1024, 2048, 4096, 8192):
        cut = sum(1 for x in lengths if x > max_len)
        print(f"  max_length={max_len:>5}: cut {cut:>3}/{n} = "
              f"{100 * cut / n:5.1f}%   covers "
              f"{100 * (n - cut) / n:5.1f}%")


if __name__ == "__main__":
    main()
