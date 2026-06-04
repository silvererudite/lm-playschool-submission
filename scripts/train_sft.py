"""SFT-LoRA trainer for the LM Playschool submission.

Configurable wrapper around the upstream `examples/trl/sft_trainer_lora.py`,
exposing the hyperparameters we expect to tune across iterations:

  * dataset filter (success-only / all / custom outcomes)
  * max_length (upstream default of 300 truncates many game contexts)
  * LoRA rank / alpha / dropout / target_modules
  * epochs, batch size, learning rate
  * eval split fraction

Run single-GPU:
    source env.sh
    playpen run scripts/train_sft.py -l Qwen3.5-2B

Run multi-GPU (data parallel via accelerate; ~4x speedup on 4 A10Gs):
    source env.sh
    accelerate launch --num_processes 4 --num_machines 1 \\
        --mixed_precision bf16 \\
        $(which playpen) run scripts/train_sft.py -l Qwen3.5-2B

Tweak the CONFIG block below; restart playpen run. Loaded by playpen's CLI
which discovers the BasePlaypenTrainer subclass automatically.

Note on multi-GPU: clemcore's huggingface_local backend hardcodes
device_map="auto", which pipeline-parallel-shards the model across all
visible GPUs and runs only one GPU at a time. For DDP we want each rank to
hold a full model copy on its own GPU. We monkeypatch `load_model` at
module import time (which runs *before* clemcore loads the model in
playpen's CLI) so that under accelerate, each rank uses
device_map={"": LOCAL_RANK}.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import trl
from datasets import load_dataset
from peft import LoraConfig

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from playpen import BasePlaypenTrainer


# ----------------------------- DDP patch ----------------------------- #
# clemcore's huggingface backend hardcodes device_map="auto" in
# AutoModelForCausalLM.from_pretrained. Under accelerate/torchrun, "auto"
# pipeline-shards across all visible GPUs — wrong for DDP, where each rank
# wants the full model on its own GPU.
#
# Instead of mirroring clemcore's load_model (which drifts as upstream
# evolves), we intercept AutoModelForCausalLM.from_pretrained and override
# device_map when LOCAL_RANK is set. Smaller blast radius, no copy-paste.

def _patch_device_map_for_ddp() -> None:
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        return  # single-process: keep upstream device_map="auto"

    local_rank = int(local_rank_str)
    from transformers import AutoModelForCausalLM
    _orig_from_pretrained = AutoModelForCausalLM.from_pretrained

    @classmethod
    def _from_pretrained_ddp(cls, *args, **kwargs):
        kwargs["device_map"] = {"": local_rank}
        if local_rank == 0:
            print(f"[train_sft] DDP: from_pretrained device_map -> "
                  f"{{'': {local_rank}}}", flush=True)
        return _orig_from_pretrained.__func__(cls, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = _from_pretrained_ddp


_patch_device_map_for_ddp()


# ----------------------------- configuration ----------------------------- #
# SimpleNamespace instead of @dataclass — the upstream CLI loads this file
# without registering the module in sys.modules, which breaks dataclasses.

CONFIG = SimpleNamespace(
    # data
    dataset_name="colab-potsdam/playpen-data",
    dataset_subset="interactions",
    success_only=True,
    eval_split=0.2,
    seed=42,

    # tokenization / packing
    # iter 1 used 300 (upstream default) — verified to truncate 64% of
    # episodes; see RUNS/iter1.md. 1024 covers ~82%, 2048 ~95%.
    max_length=1024,
    packing=False,
    completion_only_loss=True,

    # optimization. Effective global batch = num_gpus * per_device * grad_accum.
    # Iter 2: 4 GPUs * 1 per_device * 2 grad_accum = 8 (matches iter 1).
    # We tried bs=2 first; that OOMed at seq_len=1024 on A10G even with
    # gradient checkpointing budget headroom — episodes longer than the
    # nominal max push past it. bs=1 + grad_accum=2 is safe.
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    bf16=True,
    # bs=2 seq=1024 OOMs on a 23GB A10G without checkpointing (~22GB used,
    # OOM at first step). With checkpointing, activations get recomputed in
    # backward — slower per step (~30%) but cuts activation memory ~5×.
    gradient_checkpointing=True,

    # eval & logging
    # save_strategy="epoch" with 3 epochs over 6063 steps means the first
    # checkpoint doesn't land until step 2021 — ~1.5h on 4xA10G with no
    # crash recovery before that. Save every 1000 steps with total_limit=2
    # so we always have a recent fallback. (Iter 2 ran with the previous
    # "epoch" setting; this changes future runs only.)
    eval_strategy="epoch",
    logging_steps=25,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,

    # lora
    # iter 1 used modules_to_save=["lm_head", "embed_token"] — Qwen's module
    # is "embed_tokens" (plural) so peft only matched lm_head, then set
    # tie_word_embeddings=False, breaking the tied-weight invariant. See
    # RUNS/iter1.md root cause #1. Empty list = pure LoRA, no full-rank heads.
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="all-linear",
    lora_modules_to_save=[],
)


# ----------------------------- trainer ----------------------------- #

class PlayschoolSftTrainer(BasePlaypenTrainer):
    """LoRA SFT trainer with the configuration above."""

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)

    def _prepare_dataset(self):
        ds = load_dataset(
            CONFIG.dataset_name, CONFIG.dataset_subset, split="train"
        )
        before = len(ds)
        if CONFIG.success_only:
            ds = ds.filter(lambda ep: ep["meta"]["outcome"] == "success")
        print(f"[train_sft] dataset: kept {len(ds)}/{before} episodes "
              f"(success_only={CONFIG.success_only})")

        split = ds.train_test_split(
            CONFIG.eval_split, shuffle=True, seed=CONFIG.seed
        )
        print(f"[train_sft] split: {len(split['train'])} train / "
              f"{len(split['test'])} eval")
        return split

    def _make_sft_config(self, output_dir: Path) -> trl.SFTConfig:
        return trl.SFTConfig(
            output_dir=str(output_dir),
            max_length=CONFIG.max_length,
            packing=CONFIG.packing,
            completion_only_loss=CONFIG.completion_only_loss,
            num_train_epochs=CONFIG.num_train_epochs,
            per_device_train_batch_size=CONFIG.per_device_train_batch_size,
            gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,
            learning_rate=CONFIG.learning_rate,
            warmup_ratio=CONFIG.warmup_ratio,
            weight_decay=CONFIG.weight_decay,
            lr_scheduler_type=CONFIG.lr_scheduler_type,
            bf16=CONFIG.bf16,
            gradient_checkpointing=CONFIG.gradient_checkpointing,
            eval_strategy=CONFIG.eval_strategy,
            logging_steps=CONFIG.logging_steps,
            save_strategy=CONFIG.save_strategy,
            save_steps=CONFIG.save_steps,
            save_total_limit=CONFIG.save_total_limit,
            ddp_find_unused_parameters=False,
            report_to=[],
        )

    def _make_peft_config(self) -> LoraConfig:
        return LoraConfig(
            r=CONFIG.lora_r,
            lora_alpha=CONFIG.lora_alpha,
            lora_dropout=CONFIG.lora_dropout,
            target_modules=CONFIG.lora_target_modules,
            modules_to_save=CONFIG.lora_modules_to_save,
            task_type="CAUSAL_LM",
        )

    def learn(self) -> None:
        dataset = self._prepare_dataset()
        output_dir = Path(f"models/sft+lora/{self.learner.name}")
        sft_cfg = self._make_sft_config(output_dir)
        peft_cfg = self._make_peft_config()

        print(f"[train_sft] config: {CONFIG}")
        print(f"[train_sft] output_dir: {output_dir}")

        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=sft_cfg,
            peft_config=peft_cfg,
        )
        trainer.train()
        print(f"[train_sft] done. Adapters under {output_dir}/checkpoint-*")
