"""Evaluation orchestrator for the LM Playschool submission.

Wraps `playpen eval <model> --suite all` for one or more models and emits a
comparison table with deltas. Designed to make the baseline-vs-SFT comparison
a one-liner; in iter 1 we run two models, in iter N we run a list.

Usage:
    source env.sh

    # full suite, two models, write comparison to results/iter1.md
    python scripts/eval.py --suite all \\
        --model Qwen3.5-2B --model Qwen3.5-2B-sft \\
        --baseline Qwen3.5-2B --out results/iter1.md

    # single game, just sanity-check
    python scripts/eval.py -g taboo --model Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAYPEN_DIR = PROJECT_ROOT / "playpen"
EVAL_ROOT = PLAYPEN_DIR / "playpen-eval"  # symlink to NVMe


@dataclass
class EvalResult:
    model: str
    eval_dir: Path
    clemscore: float | None
    statscore: float | None

    @classmethod
    def load(cls, model: str, eval_dir: Path) -> "EvalResult":
        path = eval_dir / f"{model}.val.json"
        if not path.exists():
            return cls(model, eval_dir, None, None)
        scores = json.loads(path.read_text())
        return cls(
            model=model,
            eval_dir=eval_dir,
            clemscore=scores.get("clemscore"),
            statscore=scores.get("statscore"),
        )


def run_playpen_eval(model: str, suite: str | None, games: list[str]) -> Path:
    """Invoke `playpen eval <model>` and return the eval dir it wrote to."""
    cmd = ["playpen", "eval", model]
    if suite:
        cmd += ["--suite", suite]
    if games:
        cmd += ["-g", *games]

    before = _existing_eval_dirs()
    print(f"[eval] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PLAYPEN_DIR)
    if proc.returncode != 0:
        sys.exit(f"[eval] playpen eval {model} exited with {proc.returncode}")

    after = _existing_eval_dirs()
    new_dirs = sorted(after - before)
    if not new_dirs:
        # fallback: pick the latest
        new_dirs = [max(after, key=lambda p: p.stat().st_mtime)]
    return new_dirs[-1]


def _existing_eval_dirs() -> set[Path]:
    if not EVAL_ROOT.exists():
        return set()
    return {p for p in EVAL_ROOT.iterdir() if p.is_dir()}


def render_table(
    results: list[EvalResult], baseline: str | None
) -> str:
    """Render a markdown comparison table with deltas vs `baseline`."""
    base = next((r for r in results if r.model == baseline), None)

    head = ["model", "clemscore", "statscore"]
    if base is not None:
        head += ["Δ clem", "Δ stat"]

    rows = []
    for r in results:
        clem = "" if r.clemscore is None else f"{r.clemscore:.2f}"
        stat = "" if r.statscore is None else f"{r.statscore:.2f}"
        cells = [r.model, clem, stat]
        if base is not None:
            d_clem = ""
            d_stat = ""
            if r.clemscore is not None and base.clemscore is not None:
                d_clem = f"{r.clemscore - base.clemscore:+.2f}"
            if r.statscore is not None and base.statscore is not None:
                d_stat = f"{r.statscore - base.statscore:+.2f}"
            cells += [d_clem, d_stat]
        rows.append(cells)

    widths = [max(len(c) for c in col) for col in zip(head, *rows)]
    fmt = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(head), sep, *(fmt(r) for r in rows)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="model name from model_registry.json (repeatable)")
    ap.add_argument("--suite", choices=["all", "clem", "static"],
                    help="full suite (passes --suite to playpen eval)")
    ap.add_argument("-g", "--game", action="append", default=[],
                    help="restrict to specific game(s) (repeatable)")
    ap.add_argument("--baseline",
                    help="model name to compute deltas against")
    ap.add_argument("--out", type=Path,
                    help="write markdown comparison to this path "
                         "(default: results/eval-<timestamp>.md)")
    ap.add_argument("--reuse", type=Path, action="append", default=[],
                    help="reuse an existing eval dir for the corresponding "
                         "--model (skip running). Order matches --model.")
    args = ap.parse_args()

    if args.suite and args.game:
        sys.exit("--suite and -g are mutually exclusive")
    if not args.suite and not args.game:
        sys.exit("pass --suite or -g")
    if not shutil.which("playpen"):
        sys.exit("playpen not on PATH; did you `source env.sh`?")

    out_path = args.out or (
        PROJECT_ROOT / "results" /
        f"eval-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    for i, model in enumerate(args.model):
        reuse = args.reuse[i] if i < len(args.reuse) else None
        if reuse:
            eval_dir = reuse
            print(f"[eval] reusing {eval_dir} for {model}")
        else:
            eval_dir = run_playpen_eval(model, args.suite, args.game)
        results.append(EvalResult.load(model, eval_dir))

    table = render_table(results, args.baseline)
    header = (
        f"# Eval run {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}\n\n"
        f"- suite: `{args.suite or 'games:' + ','.join(args.game)}`\n"
        f"- baseline: `{args.baseline or '(none)'}`\n\n"
    )
    rendered = header + table + "\n"
    out_path.write_text(rendered)

    print()
    print(rendered)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
