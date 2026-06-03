"""Merge a LoRA adapter into its base model and upload to HuggingFace Hub.

Usage:
    source env.sh
    python scripts/merge_and_push.py \\
        --base Qwen/Qwen3.5-2B \\
        --adapter playpen/models/sft+lora/Qwen3.5-2B/checkpoint-LATEST \\
        --repo silvererudite/lm-playschool-qwen3.5-2b-sft \\
        [--private] [--no-push] [--out /tmp/merged]

Requires huggingface.api_key in playpen/key.json (env.sh exports HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_latest_checkpoint(adapter_dir: Path) -> Path:
    """Resolve a path that may be a checkpoint dir, an adapter dir, or
    a parent containing checkpoint-* subdirs."""
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir
    candidates = sorted(
        (p for p in adapter_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No adapter at {adapter_dir} or checkpoint-* subdirectories"
        )
    return candidates[-1]


def merge(base: str, adapter: Path, out: Path, hf_token: str | None) -> Path:
    print(f"[merge] base={base}")
    print(f"[merge] adapter={adapter}")
    print(f"[merge] out={out}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="cpu", token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(base, token=hf_token)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter))
    merged = peft_model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    meta = {
        "base_model": base,
        "adapter_source": str(adapter),
        "adapter_config": json.loads((adapter / "adapter_config.json").read_text()),
    }
    (out / "playschool_provenance.json").write_text(json.dumps(meta, indent=2))
    print(f"[merge] wrote merged model to {out}")
    return out


def push(local: Path, repo: str, private: bool, hf_token: str) -> str:
    print(f"[push] repo={repo} private={private}")
    create_repo(repo, private=private, exist_ok=True, token=hf_token)
    api = HfApi(token=hf_token)
    api.upload_folder(folder_path=str(local), repo_id=repo, repo_type="model")
    url = f"https://huggingface.co/{repo}"
    print(f"[push] done -> {url}")
    return url


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True,
                    help="HF id of the base model, e.g. Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", required=True, type=Path,
                    help="Path to adapter dir or its parent")
    ap.add_argument("--repo", required=True,
                    help="HF repo id to push to, e.g. user/lm-playschool-foo")
    ap.add_argument("--out", type=Path,
                    help="Local merged-model dir (default: <project>/dist/<repo basename>)")
    ap.add_argument("--private", action="store_true",
                    help="Create the HF repo as private")
    ap.add_argument("--no-push", action="store_true",
                    help="Just merge locally, don't upload")
    args = ap.parse_args()

    adapter = find_latest_checkpoint(args.adapter)
    out = args.out or (PROJECT_ROOT / "dist" / Path(args.repo).name)

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if hf_token is None:
        try:
            hf_token = json.loads(
                (PROJECT_ROOT / "playpen" / "key.json").read_text()
            )["huggingface"]["api_key"]
        except Exception as e:
            raise SystemExit(f"No HF token found (env or key.json): {e}")

    merge(args.base, adapter, out, hf_token)
    if not args.no_push:
        push(out, args.repo, args.private, hf_token)


if __name__ == "__main__":
    main()
