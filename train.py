from __future__ import annotations

import argparse
import json
import os
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
    optimizer_debug: bool = False
    _logged_runtime_optimizer_state: bool = False

    @staticmethod
    def _is_rank_zero() -> bool:
        return os.environ.get("RANK", "0") == "0"

    def _log_optimizer_scheduler_state(self, stage: str) -> None:
        if not self.optimizer_debug or not self._is_rank_zero():
            return
        optimizer = getattr(self, "optimizer", None)
        scheduler = getattr(self, "lr_scheduler", None)
        print(f"[optim-debug] stage={stage}", flush=True)
        print(
            "[optim-debug] "
            f"deepspeed_arg={self.args.deepspeed} "
            f"is_deepspeed_enabled={getattr(self, 'is_deepspeed_enabled', None)} "
            f"args_gradient_accumulation_steps={self.args.gradient_accumulation_steps}",
            flush=True,
        )
        if optimizer is None:
            print("[optim-debug] optimizer=None", flush=True)
        else:
            param_groups = getattr(optimizer, "param_groups", [])
            print(
                f"[optim-debug] optimizer_type={type(optimizer).__name__} "
                f"param_groups={len(param_groups)}",
                flush=True,
            )
            name_by_id = {id(param): name for name, param in self.model.named_parameters()}
            for idx, group in enumerate(param_groups):
                params = list(group.get("params", []))
                total_tensors = len(params)
                trainable_tensors = sum(1 for param in params if getattr(param, "requires_grad", False))
                trainable_elems = sum(
                    int(param.numel()) for param in params if getattr(param, "requires_grad", False)
                )
                sample_names = [name_by_id.get(id(param), "<wrapped>") for param in params[:8]]
                print(
                    "[optim-debug] "
                    f"group={idx} tensors={total_tensors} trainable_tensors={trainable_tensors} "
                    f"trainable_elems={trainable_elems} lr={group.get('lr')} "
                    f"weight_decay={group.get('weight_decay')} sample={sample_names}",
                    flush=True,
                )
        if scheduler is None:
            print("[optim-debug] scheduler=None", flush=True)
        else:
            base_lrs = list(getattr(scheduler, "base_lrs", []) or [])
            try:
                last_lrs = list(scheduler.get_last_lr())
            except Exception as exc:
                last_lrs = [f"<get_last_lr failed: {exc}>"]
            print(
                f"[optim-debug] scheduler_type={type(scheduler).__name__} "
                f"base_lrs={len(base_lrs)} last_lrs={len(last_lrs)} "
                f"base_lrs_values={base_lrs} last_lrs_values={last_lrs}",
                flush=True,
            )

    def create_optimizer(self):
        if self.optimizer is None and self.args.deepspeed:
            trainable_named_params = [
                (name, param) for name, param in self.model.named_parameters() if param.requires_grad
            ]
            if not trainable_named_params:
                raise ValueError("No trainable parameters found for optimizer creation.")
            try:
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            except TypeError:
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(
                [
                    {
                        "params": [param for _, param in trainable_named_params],
                        "weight_decay": self.args.weight_decay,
                    }
                ],
                **optimizer_kwargs,
            )
            if self.optimizer_debug and self._is_rank_zero():
                trainable_tensors = len(trainable_named_params)
                trainable_elems = sum(int(param.numel()) for _, param in trainable_named_params)
                sample_names = [name for name, _ in trainable_named_params[:12]]
                print(
                    "[optim-debug] using_single_deepspeed_optimizer_group=true "
                    f"trainable_tensors={trainable_tensors} trainable_elems={trainable_elems} "
                    f"sample_trainable_names={sample_names}",
                    flush=True,
                )
        else:
            super().create_optimizer()
        self._log_optimizer_scheduler_state("after_create_optimizer")
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        scheduler = super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
        self._log_optimizer_scheduler_state("after_create_scheduler")
        return scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not self._logged_runtime_optimizer_state:
            self._log_optimizer_scheduler_state("first_compute_loss_runtime")
            self._logged_runtime_optimizer_state = True
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


def optional_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def configure_reporting_environment(train_cfg: dict[str, Any]) -> None:
    report_to = train_cfg.get("report_to", "wandb")
    if not report_to:
        return
    if isinstance(report_to, str):
        report_targets = {item.strip().lower() for item in report_to.split(",")}
    else:
        report_targets = {str(item).strip().lower() for item in report_to}

    if "wandb" not in report_targets:
        return

    if "wandb_log_model" in train_cfg:
        os.environ["WANDB_LOG_MODEL"] = str(train_cfg["wandb_log_model"]).lower()
    else:
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")
    if os.environ.get("RANK", "0") == "0":
        print(
            "[wandb-config] "
            f"WANDB_LOG_MODEL={os.environ.get('WANDB_LOG_MODEL')} "
            f"WANDB_WATCH={os.environ.get('WANDB_WATCH')}",
            flush=True,
        )


def resolve_deepspeed_config(train_cfg: dict[str, Any], output_dir: Path) -> str | None:
    source = train_cfg.get("deepspeed")
    if not source:
        return None
    source_path = Path(source)
    with source_path.open("r", encoding="utf-8") as f:
        ds_cfg = json.load(f)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    micro_batch = int(train_cfg["per_device_train_batch_size"])
    ds_cfg["gradient_accumulation_steps"] = grad_accum
    ds_cfg["train_micro_batch_size_per_gpu"] = micro_batch
    ds_cfg["train_batch_size"] = micro_batch * grad_accum * world_size
    ds_cfg["gradient_clipping"] = float(train_cfg["max_grad_norm"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "deepspeed_zero3.resolved.json"
    with resolved_path.open("w", encoding="utf-8") as f:
        json.dump(ds_cfg, f, indent=2)
    if os.environ.get("RANK", "0") == "0":
        print(
            "[deepspeed-config] "
            f"source={source_path} resolved={resolved_path} "
            f"gradient_accumulation_steps={grad_accum} "
            f"micro_batch={micro_batch} world_size={world_size} "
            f"train_batch_size={ds_cfg['train_batch_size']}",
            flush=True,
        )
    return str(resolved_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Override config for smoke tests.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_reporting_environment(cfg["training"])
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
    deepspeed_config = resolve_deepspeed_config(train_cfg=cfg["training"], output_dir=output_dir)
    cfg["training"]["deepspeed_resolved"] = deepspeed_config
    write_run_metadata(output_dir, cfg)

    train_dataset = PreparedSpeechDataset(train_path)
    eval_dataset = PreparedSpeechDataset(val_path) if val_path.exists() else None

    train_cfg = cfg["training"]
    max_steps = args.max_steps if args.max_steps is not None else optional_int(train_cfg.get("max_steps"), -1)
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
        max_steps=max_steps,
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        bf16=bool(train_cfg["bf16"]),
        dataloader_num_workers=int(train_cfg["dataloader_num_workers"]),
        report_to=train_cfg.get("report_to", "wandb"),
        deepspeed=deepspeed_config,
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
    trainer.optimizer_debug = bool(train_cfg.get("debug_optimizer_state", True))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
