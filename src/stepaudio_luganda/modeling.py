from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def torch_dtype(name: str | None) -> torch.dtype:
    if not name:
        return torch.bfloat16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}")


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name_or_path: str, cfg: dict[str, Any]):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(cfg.get("trust_remote_code", True)),
        torch_dtype=torch_dtype(cfg.get("dtype", "bfloat16")),
    )
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    return model


def apply_lora(model, cfg: dict[str, Any]):
    if not cfg.get("enabled", True):
        return model
    lora_config = LoraConfig(
        r=int(cfg.get("r", 64)),
        lora_alpha=int(cfg.get("alpha", 128)),
        lora_dropout=float(cfg.get("dropout", 0.05)),
        bias=str(cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(cfg.get("target_modules", [])),
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def set_audio_trainability(model, train_adapter: bool = False, train_audio_encoder: bool = False) -> None:
    base = getattr(model, "base_model", model)
    if hasattr(base, "model"):
        base = base.model
    for name, param in base.named_parameters():
        if name.startswith("adapter."):
            param.requires_grad = train_adapter
        elif name.startswith("encoder."):
            param.requires_grad = train_audio_encoder
