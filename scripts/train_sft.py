"""SFT-LoRA trainer for the LM Playschool submission.

Configurable wrapper around the upstream `examples/trl/sft_trainer_lora.py`,
exposing the hyperparameters we expect to tune across iterations:

  * dataset filter (success-only / all / custom outcomes)
  * max_length (upstream default of 300 truncates many game contexts)
  * LoRA rank / alpha / dropout / target_modules
  * epochs, batch size, learning rate
  * eval split fraction

Run via:
    source env.sh
    playpen run scripts/train_sft.py -l Qwen3.5-2B

Tweak the CONFIG block below; restart playpen run. Loaded by playpen's CLI
which discovers the BasePlaypenTrainer subclass automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import trl
from datasets import load_dataset
from peft import LoraConfig

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from playpen import BasePlaypenTrainer


# ----------------------------- configuration ----------------------------- #

@dataclass
class TrainConfig:
    # data
    dataset_name: str = "colab-potsdam/playpen-data"
    dataset_subset: str = "interactions"
    success_only: bool = True
    eval_split: float = 0.2
    seed: int = 42

    # tokenization / packing
    max_length: int = 300        # upstream default; bump to 1024+ in iter 2
    packing: bool = False
    completion_only_loss: bool = True

    # optimization
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"

    # eval & logging
    eval_strategy: str = "epoch"
    logging_steps: int = 25
    save_strategy: str = "epoch"
    save_total_limit: int = 2

    # lora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"
    lora_modules_to_save: list[str] = field(
        default_factory=lambda: ["lm_head", "embed_token"]
    )


CONFIG = TrainConfig()


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
            eval_strategy=CONFIG.eval_strategy,
            logging_steps=CONFIG.logging_steps,
            save_strategy=CONFIG.save_strategy,
            save_total_limit=CONFIG.save_total_limit,
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
