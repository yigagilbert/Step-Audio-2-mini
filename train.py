from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.data import PreparedSpeechDataset, StepAudioCollator  # noqa: E402
from stepaudio_luganda.formatting import StepAudioFormatter  # noqa: E402
from stepaudio_luganda.modeling import (  # noqa: E402
    apply_lora,
    load_model,
    load_tokenizer,
    set_audio_trainability,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StepAudioTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


def write_run_metadata(output_dir: Path, cfg: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_config.resolved.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["project"].get("seed", 42)))
    if cfg["training"].get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    processed_dir = Path(cfg["project"]["processed_dir"])
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "validation.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Run data_prep.py first; missing {train_path}")

    tokenizer = load_tokenizer(
        cfg["model"].get("local_path") or cfg["model"]["name_or_path"],
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    formatter = StepAudioFormatter(
        tokenizer=tokenizer,
        system_prompt=cfg["format"]["system_prompt"],
        target_format=cfg["format"]["target_format"],
        max_target_audio_tokens=cfg["format"].get("max_target_audio_tokens"),
    )
    collator = StepAudioCollator(
        formatter=formatter,
        pad_token_id=tokenizer.pad_token_id,
        max_sequence_length=int(cfg["format"].get("max_sequence_length", 16384)),
    )

    model = load_model(
        cfg["model"].get("local_path") or cfg["model"]["name_or_path"],
        cfg["model"],
    )
    model = apply_lora(model, cfg["lora"])
    set_audio_trainability(
        model,
        train_adapter=bool(cfg["lora"].get("train_adapter", False)),
        train_audio_encoder=bool(cfg["lora"].get("train_audio_encoder", False)),
    )

    output_dir = Path(cfg["project"]["output_dir"])
    write_run_metadata(output_dir, cfg)

    train_dataset = PreparedSpeechDataset(train_path)
    eval_dataset = PreparedSpeechDataset(val_path) if val_path.exists() else None

    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=bool(train_cfg["bf16"]),
        dataloader_num_workers=int(train_cfg["dataloader_num_workers"]),
        report_to=train_cfg.get("report_to", "tensorboard"),
        deepspeed=train_cfg.get("deepspeed"),
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        remove_unused_columns=False,
        push_to_hub=bool(train_cfg.get("push_to_hub", False)),
        hub_model_id=train_cfg.get("hub_model_id"),
    )

    trainer = StepAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
