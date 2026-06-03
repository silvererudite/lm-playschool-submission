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

from clemcore.backends import huggingface_local_api as _hf_api
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from playpen import BasePlaypenTrainer


# ----------------------------- DDP patch ----------------------------- #
# Pin the model to the current rank's GPU when launched under accelerate /
# torchrun. The presence of LOCAL_RANK is the standard DDP signal.

def _patch_clemcore_for_ddp() -> None:
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        return  # single-process: keep upstream device_map="auto"

    local_rank = int(local_rank_str)
    _orig_load_model = _hf_api.load_model

    def _load_model_ddp(model_spec):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from clemcore.backends.utils.key_registry import KeyRegistry
        from peft import PeftModel

        model_args = dict(device_map={"": local_rank}, torch_dtype="auto")
        bnb_kwargs = {}
        if "load_in_8bit" in model_spec.model_config:
            bnb_kwargs["load_in_8bit"] = model_spec.model_config["load_in_8bit"]
        if "load_in_4bit" in model_spec.model_config:
            bnb_kwargs["load_in_4bit"] = model_spec.model_config["load_in_4bit"]
        if bnb_kwargs:
            model_args["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
        if model_spec.model_config.get("requires_api_key"):
            key = KeyRegistry.from_json().get_key_for("huggingface")
            model_args["token"] = key["api_key"]
        if model_spec.model_config.get("trust_remote_code"):
            model_args["trust_remote_code"] = True
        if "attn_implementation" in model_spec.model_config:
            model_args["attn_implementation"] = (
                model_spec.model_config["attn_implementation"]
            )

        print(f"[train_sft] DDP: loading {model_spec['huggingface_id']} "
              f"with device_map={{'': {local_rank}}}")
        model = AutoModelForCausalLM.from_pretrained(
            model_spec["huggingface_id"], **model_args
        )
        if "peft_model" in model_spec.model_config:
            model = PeftModel.from_pretrained(
                model, model_spec.model_config["peft_model"]
            )
        return model

    _hf_api.load_model = _load_model_ddp


_patch_clemcore_for_ddp()


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
    max_length=300,        # upstream default; bump to 1024+ in iter 2
    packing=False,
    completion_only_loss=True,

    # optimization. Effective global batch = num_gpus * per_device * grad_accum.
    # Iter 1 single-GPU defaults gave 1*1*8=8. With 4 GPUs we keep ≈8 via
    # 4 * 2 * 1, doubling throughput per GPU to fully use VRAM.
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.0,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=False,

    # eval & logging
    eval_strategy="epoch",
    logging_steps=25,
    save_strategy="epoch",
    save_total_limit=2,

    # lora
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_target_modules="all-linear",
    lora_modules_to_save=["lm_head", "embed_token"],
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
