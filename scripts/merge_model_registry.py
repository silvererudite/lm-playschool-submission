#!/usr/bin/env python3
"""Idempotently prepend our model entries to playpen's model_registry.json.

Usage:
  python scripts/merge_model_registry.py --upstream playpen/model_registry.json \
      --add config/qwen3.5-2b.entries.json
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--add", required=True, type=Path)
    args = ap.parse_args()

    upstream = json.loads(args.upstream.read_text())
    additions = json.loads(args.add.read_text())

    have = {entry["model_name"] for entry in upstream}
    new = [entry for entry in additions if entry["model_name"] not in have]
    if not new:
        print(f"[merge_model_registry] nothing to add; {args.upstream} already has",
              ", ".join(e["model_name"] for e in additions))
        return

    merged = new + upstream
    args.upstream.write_text(json.dumps(merged, indent=2) + "\n")
    print(f"[merge_model_registry] added {len(new)} entries:",
          ", ".join(e["model_name"] for e in new))


if __name__ == "__main__":
    main()
