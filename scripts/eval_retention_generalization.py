from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import sacrebleu
import torch
import yaml
from jiwer import wer
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.audio import audio_cell_to_waveform, log_mel_spectrogram  # noqa: E402
from stepaudio_luganda.data import read_jsonl  # noqa: E402
from stepaudio_luganda.formatting import StepAudioFormatter  # noqa: E402
from stepaudio_luganda.modeling import load_model, load_tokenizer, torch_dtype  # noqa: E402


LANG_TO_NLLB = {
    "arabic": "arb_Arab",
    "chinese": "zho_Hans",
    "english": "eng_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "japanese": "jpn_Jpan",
    "luganda": "lug_Latn",
    "spanish": "spa_Latn",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_prediction_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text.splitlines()[0].strip()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:80].strip("_") or "sample"


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def choose_base_model_path(cfg: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return cfg["model"].get("local_path") or cfg["model"]["name_or_path"]


def resolve_adapter_path(cfg: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return str(Path(cfg["project"]["output_dir"]) / "final")


def prompt_from_template(template: str, source_language: str, target_language: str) -> str:
    return template.format(
        source_language=source_language,
        target_language=target_language,
    )


def extract_outputs(
    tokenizer,
    formatter: StepAudioFormatter,
    token_ids: list[int],
) -> tuple[str, list[int]]:
    text_ids: list[int] = []
    audio_tokens: list[int] = []
    seen_audio = False
    for token_id in token_ids:
        if token_id == formatter.eot_id:
            break
        if token_id == formatter.tts_start_id:
            seen_audio = True
            continue
        if token_id == formatter.tts_end_id:
            break
        if token_id >= formatter.audio_token_offset:
            audio_token = token_id - formatter.audio_token_offset
            if 0 <= audio_token <= formatter.tts_valid_max:
                audio_tokens.append(audio_token)
            seen_audio = True
        elif not seen_audio and token_id < formatter.audio_start_id:
            text_ids.append(token_id)
    text = clean_prediction_text(tokenizer.decode(text_ids, skip_special_tokens=True))
    return text, audio_tokens


def language_script_hit(text: str, target_language: str) -> bool | None:
    if not text:
        return False
    lang = target_language.strip().lower()
    if lang == "chinese":
        return any("\u4e00" <= char <= "\u9fff" for char in text)
    if lang == "japanese":
        return any(
            ("\u3040" <= char <= "\u30ff") or ("\u4e00" <= char <= "\u9fff")
            for char in text
        )
    if lang in {"english", "spanish"}:
        return None
    return None


def get_nested_value(row: dict[str, Any], key: str | None, default: Any = "") -> Any:
    if not key:
        return default
    value: Any = row
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def normalize_reference(value: Any, preferred_keys: list[str] | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        keys = preferred_keys or ["en", "eng", "english", "translation", "text"]
        for key in keys:
            if value.get(key):
                return str(value[key]).strip()
        for item in value.values():
            if item:
                return str(item).strip()
        return ""
    return str(value).strip()


def row_to_mel(row: dict[str, Any], audio_field: str) -> torch.Tensor:
    if row.get("src_mel_path"):
        return torch.load(row["src_mel_path"], map_location="cpu")
    audio_cell = get_nested_value(row, audio_field, None)
    if audio_cell is None:
        raise ValueError(f"Missing audio field {audio_field!r} for row {row.get('id')!r}.")
    waveform = audio_cell_to_waveform(audio_cell, target_rate=16000)
    return log_mel_spectrogram(waveform, n_mels=128, padding=479).cpu()


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    formatter: StepAudioFormatter,
    row: dict[str, Any],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    mel = row_to_mel(row, args.audio_field)
    prompt_ids = formatter.build_prompt(int(mel.shape[1]))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    wavs = mel.unsqueeze(0).to(device=device, dtype=torch.float32)
    wav_lens = torch.tensor([max(1, int(mel.shape[1]) - 2)], dtype=torch.int32, device=device)

    gen_cfg = cfg["generation"]
    do_sample = bool(gen_cfg.get("do_sample", True))
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    if args.deterministic:
        do_sample = False
        temperature = None
        top_p = None
    if args.do_sample is not None:
        do_sample = args.do_sample
    if args.temperature is not None:
        temperature = args.temperature
    if args.top_p is not None:
        top_p = args.top_p

    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "wavs": wavs,
        "wav_lens": wav_lens,
        "max_new_tokens": int(args.max_new_tokens or gen_cfg["max_new_tokens"]),
        "repetition_penalty": float(args.repetition_penalty or gen_cfg["repetition_penalty"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": formatter.eot_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    started = time.perf_counter()
    output = model.generate(**generate_kwargs)
    elapsed_s = time.perf_counter() - started
    new_ids = output[0, len(prompt_ids) :].detach().cpu().tolist()
    prediction, audio_tokens = extract_outputs(tokenizer, formatter, new_ids)
    return {
        "prediction": prediction,
        "audio_tokens": audio_tokens,
        "audio_token_count": len(audio_tokens),
        "generated_token_count": len(new_ids),
        "generation_s": elapsed_s,
    }


def load_hf_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": args.hf_split,
        "token": os.environ.get(args.hf_token_env) if args.hf_token_env else None,
    }
    if args.hf_trust_remote_code:
        kwargs["trust_remote_code"] = True
    dataset = load_dataset(args.hf_dataset, args.hf_config, **kwargs)

    reference_by_id: dict[str, str] = {}
    if args.hf_reference_config:
        ref_dataset = load_dataset(args.hf_dataset, args.hf_reference_config, **kwargs)
        for ref_row in ref_dataset:
            row_id = str(get_nested_value(ref_row, args.id_field, ""))
            reference_by_id[row_id] = normalize_reference(
                get_nested_value(ref_row, args.hf_reference_field, ""),
                preferred_keys=split_csv(args.reference_preferred_keys),
            )

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        if args.limit and len(rows) >= args.limit:
            break
        row = dict(row)
        row_id = str(get_nested_value(row, args.id_field, idx))
        reference = reference_by_id.get(row_id)
        if reference is None:
            reference = normalize_reference(
                get_nested_value(row, args.reference_field, ""),
                preferred_keys=split_csv(args.reference_preferred_keys),
            )
        row["id"] = row_id
        row["reference"] = reference
        row["source_text"] = normalize_reference(get_nested_value(row, args.source_text_field, ""))
        rows.append(row)
    return rows


def load_jsonl_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.input_jsonl)
    normalized = []
    for idx, row in enumerate(rows):
        if args.limit and len(normalized) >= args.limit:
            break
        item = dict(row)
        item["id"] = str(get_nested_value(item, args.id_field, idx))
        item["reference"] = normalize_reference(get_nested_value(item, args.reference_field, ""))
        item["source_text"] = normalize_reference(get_nested_value(item, args.source_text_field, ""))
        normalized.append(item)
    return normalized


def load_composition_rows(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.prepared_jsonl or Path(cfg["project"]["processed_dir"]) / f"{args.split}.jsonl")
    rows = read_jsonl(path)
    normalized = []
    for row in rows:
        if args.limit and len(normalized) >= args.limit:
            break
        item = dict(row)
        item["reference"] = str(row.get("text_eng", "")).strip()
        item["source_text"] = str(row.get("text_lug", "")).strip()
        normalized.append(item)
    return normalized


def load_eval_rows(cfg: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.suite == "composition":
        return load_composition_rows(cfg, args)
    if args.input_jsonl:
        return load_jsonl_rows(args)
    return load_hf_rows(args)


def resolve_nllb_lang(value: str) -> str:
    return LANG_TO_NLLB.get(value.strip().lower(), value)


class NllbTranslator:
    def __init__(self, model_name: str, device: torch.device, dtype_name: str) -> None:
        if dtype_name == "auto":
            dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        else:
            dtype = torch_dtype(dtype_name)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()

    @torch.inference_mode()
    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int,
        max_input_tokens: int,
        max_new_tokens: int,
        num_beams: int,
    ) -> list[str]:
        outputs: list[str] = []
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = src_lang
        if hasattr(self.tokenizer, "lang_code_to_id"):
            forced_bos_token_id = self.tokenizer.lang_code_to_id.get(tgt_lang)
        else:
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        if forced_bos_token_id is None or forced_bos_token_id < 0:
            raise ValueError(f"Could not resolve NLLB target language token {tgt_lang!r}.")
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to(self.device)
            output_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
            )
            outputs.extend(
                clean_prediction_text(text)
                for text in self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            )
        return outputs


def maybe_comet(
    predictions: list[str],
    references: list[str],
    sources: list[str],
    model_name: str | None,
) -> float | None:
    if not model_name:
        return None
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("[WARN] Install unbabel-comet to compute COMET.")
        return None
    checkpoint = download_model(model_name)
    model = load_from_checkpoint(checkpoint)
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
    comet_output = model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    return float(comet_output.system_score)


def compute_metrics(
    rows: list[dict[str, Any]],
    text_field: str,
    metric_prefix: str,
    comet_model: str | None,
) -> dict[str, Any]:
    predictions = [str(row.get(text_field, "") or "") for row in rows]
    references = [str(row.get("reference", "") or "") for row in rows]
    sources = [str(row.get("source_text", "") or "") for row in rows]
    nonempty_pairs = [(p, r, s) for p, r, s in zip(predictions, references, sources) if r.strip()]
    if nonempty_pairs:
        preds, refs, srcs = map(list, zip(*nonempty_pairs))
    else:
        preds, refs, srcs = [], [], []
    metrics: dict[str, Any] = {
        f"{metric_prefix}_count": len(preds),
        f"{metric_prefix}_bleu": sacrebleu.corpus_bleu(preds, [refs]).score if preds else None,
        f"{metric_prefix}_chrf": sacrebleu.corpus_chrf(preds, [refs]).score if preds else None,
        f"{metric_prefix}_wer": wer(refs, preds) if preds else None,
    }
    comet_score = maybe_comet(preds, refs, srcs, comet_model)
    if comet_score is not None:
        metrics[f"{metric_prefix}_comet"] = comet_score
    return metrics


def summarize_group(
    group_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    count = len(group_rows)
    script_hits = [row.get("target_script_hit") for row in group_rows]
    script_hits = [hit for hit in script_hits if hit is not None]
    metrics: dict[str, Any] = {
        "count": count,
        "empty_prediction_rate": (
            sum(1 for row in group_rows if not str(row.get("prediction", "")).strip()) / count
            if count
            else None
        ),
        "mean_generation_s_per_sample": (
            sum(float(row.get("generation_s", 0.0)) for row in group_rows) / count if count else None
        ),
        "valid_audio_token_rate": (
            sum(int(row.get("audio_token_count", 0)) >= args.min_audio_tokens for row in group_rows) / count
            if count
            else None
        ),
    }
    if script_hits:
        metrics["target_script_hit_rate"] = sum(bool(hit) for hit in script_hits) / len(script_hits)
    return metrics


def run_condition(
    condition: str,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    tokenizer,
    device: torch.device,
) -> list[dict[str, Any]]:
    system_prompt = prompt_from_template(
        args.prompt_template,
        source_language=args.source_language,
        target_language=args.target_language,
    )
    formatter = StepAudioFormatter(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        target_format=cfg["format"]["target_format"],
        max_target_audio_tokens=cfg["format"].get("max_target_audio_tokens"),
    )
    model_path = choose_base_model_path(cfg, args.base_model)
    model = load_model(model_path, cfg["model"])
    adapter_loaded = False
    if condition == "adapter":
        adapter_path = resolve_adapter_path(cfg, args.adapter)
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        adapter_loaded = True
    model.to(device).eval()

    outputs: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"{condition}: {args.source_language}->{args.target_language}"):
        try:
            generated = generate_one(model, tokenizer, formatter, row, cfg, args, device)
            prediction = generated["prediction"]
            outputs.append(
                {
                    "id": row.get("id"),
                    "condition": condition,
                    "adapter_loaded": adapter_loaded,
                    "suite": args.suite,
                    "source_language": args.source_language,
                    "target_language": args.target_language,
                    "source_text": row.get("source_text", ""),
                    "reference": row.get("reference", ""),
                    "prediction": prediction,
                    "target_script_hit": language_script_hit(prediction, args.target_language),
                    "audio_token_count": generated["audio_token_count"],
                    "generated_token_count": generated["generated_token_count"],
                    "generation_s": generated["generation_s"],
                }
            )
        except Exception as exc:
            outputs.append(
                {
                    "id": row.get("id"),
                    "condition": condition,
                    "adapter_loaded": adapter_loaded,
                    "suite": args.suite,
                    "source_language": args.source_language,
                    "target_language": args.target_language,
                    "source_text": row.get("source_text", ""),
                    "reference": row.get("reference", ""),
                    "prediction": "",
                    "error": str(exc),
                    "target_script_hit": False,
                    "audio_token_count": 0,
                    "generated_token_count": 0,
                    "generation_s": 0.0,
                }
            )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def default_output_paths(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    out_dir = Path(cfg["project"]["output_dir"]) / "eval" / "retention"
    stem = f"{args.suite}_{safe_name(args.source_language)}_to_{safe_name(args.target_language)}"
    return out_dir / f"{stem}_predictions.jsonl", out_dir / f"{stem}_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Step-Audio pre-fine-tuning capability retention and cross-direction "
            "compositional generalization."
        )
    )
    parser.add_argument("--config", default="configs/h100_nvl_fast_deepspeed.yaml")
    parser.add_argument("--suite", choices=("preservation", "composition"), required=True)
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--conditions", default="base,adapter")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prepared-jsonl", default=None)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--hf-dataset", default="google/fleurs")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--hf-trust-remote-code", action="store_true")
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-reference-config", default=None)
    parser.add_argument("--hf-reference-field", default="transcription")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--source-text-field", default="transcription")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--reference-preferred-keys", default="en,eng,english,translation,text")
    parser.add_argument(
        "--prompt-template",
        default=(
            "You are a professional {source_language}-to-{target_language} speech "
            "translation system. Listen to the {source_language} speech and answer "
            "with natural {target_language} speech."
        ),
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--do-sample", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--min-audio-tokens", type=int, default=10)
    parser.add_argument("--backtranslate", action="store_true")
    parser.add_argument("--backtranslate-model", default="facebook/nllb-200-distilled-1.3B")
    parser.add_argument("--backtranslate-src-lang", default="auto")
    parser.add_argument("--backtranslate-tgt-lang", default="eng_Latn")
    parser.add_argument("--backtranslate-batch-size", type=int, default=8)
    parser.add_argument("--backtranslate-max-input-tokens", type=int, default=256)
    parser.add_argument("--backtranslate-max-new-tokens", type=int, default=256)
    parser.add_argument("--backtranslate-num-beams", type=int, default=4)
    parser.add_argument("--mt-dtype", default="auto")
    parser.add_argument("--comet-model", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--metrics-path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = choose_base_model_path(cfg, args.base_model)
    tokenizer = load_tokenizer(model_path, trust_remote_code=cfg["model"].get("trust_remote_code", True))
    rows = load_eval_rows(cfg, args)
    if not rows:
        raise ValueError("No evaluation rows loaded.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = split_csv(args.conditions)
    unknown_conditions = set(conditions) - {"base", "adapter"}
    if unknown_conditions:
        raise ValueError(f"Unknown condition(s): {sorted(unknown_conditions)}")

    all_outputs: list[dict[str, Any]] = []
    for condition in conditions:
        all_outputs.extend(run_condition(condition, rows, cfg, args, tokenizer, device))

    if args.backtranslate:
        src_lang = (
            resolve_nllb_lang(args.target_language)
            if args.backtranslate_src_lang == "auto"
            else resolve_nllb_lang(args.backtranslate_src_lang)
        )
        tgt_lang = resolve_nllb_lang(args.backtranslate_tgt_lang)
        print(f"Backtranslating {args.target_language} outputs with {args.backtranslate_model}")
        translator = NllbTranslator(args.backtranslate_model, device=device, dtype_name=args.mt_dtype)
        for condition in conditions:
            condition_rows = [row for row in all_outputs if row["condition"] == condition]
            translations = translator.translate(
                [str(row.get("prediction", "")) for row in condition_rows],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                batch_size=args.backtranslate_batch_size,
                max_input_tokens=args.backtranslate_max_input_tokens,
                max_new_tokens=args.backtranslate_max_new_tokens,
                num_beams=args.backtranslate_num_beams,
            )
            for row, backtranslation in zip(condition_rows, translations):
                row["backtranslation"] = backtranslation
                row["backtranslation_model"] = args.backtranslate_model
                row["backtranslation_src_lang"] = src_lang
                row["backtranslation_tgt_lang"] = tgt_lang

    metrics: dict[str, Any] = {
        "suite": args.suite,
        "source_language": args.source_language,
        "target_language": args.target_language,
        "conditions": conditions,
        "limit": args.limit,
        "base_model": model_path,
        "adapter": resolve_adapter_path(cfg, args.adapter),
    }
    for condition in conditions:
        group_rows = [row for row in all_outputs if row["condition"] == condition]
        prefix = condition
        metrics[prefix] = summarize_group(group_rows, args)
        if args.suite == "preservation" or args.target_language.strip().lower() == "english":
            metrics[prefix].update(
                compute_metrics(group_rows, "prediction", "direct", args.comet_model)
            )
        if args.backtranslate:
            metrics[prefix].update(
                compute_metrics(group_rows, "backtranslation", "roundtrip", args.comet_model)
            )

    predictions_path, metrics_path = default_output_paths(cfg, args)
    predictions_path = Path(args.output_jsonl or predictions_path)
    metrics_path = Path(args.metrics_path or metrics_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in all_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
