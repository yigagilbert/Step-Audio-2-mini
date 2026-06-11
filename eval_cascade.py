from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import sacrebleu
import torch
import yaml
from jiwer import wer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.data import read_jsonl  # noqa: E402


NLLB_LANG_ALIASES = {
    "lug": "lug_Latn",
    "luganda": "lug_Latn",
    "ganda": "lug_Latn",
    "eng": "eng_Latn",
    "en": "eng_Latn",
    "english": "eng_Latn",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text.splitlines()[0].strip()


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_nllb_lang(value: str) -> str:
    return NLLB_LANG_ALIASES.get(value.strip().lower(), value)


def resolve_device(device_arg: str | None) -> tuple[torch.device, int]:
    if device_arg:
        device = torch.device(device_arg)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_device = -1
    if device.type == "cuda":
        pipeline_device = device.index if device.index is not None else 0
    return device, pipeline_device


def source_wav_path(row: dict[str, Any], processed_dir: Path, split: str) -> Path:
    if row.get("src_wav_path"):
        return Path(row["src_wav_path"])
    return processed_dir / split / "wav" / f"{row['id']}.lug.wav"


def require_source_wavs(rows: list[dict[str, Any]], processed_dir: Path, split: str) -> None:
    missing = []
    for row in rows:
        wav_path = source_wav_path(row, processed_dir, split)
        if not wav_path.exists():
            missing.append(wav_path)
    if missing:
        first = missing[0]
        raise FileNotFoundError(
            f"Missing {len(missing)} source wav file(s); first missing: {first}. "
            "Run data_prep.py with dataset.audio.save_resampled_wavs=true before cascade eval."
        )


def build_asr_pipeline(asr_model: str, device: torch.device, pipeline_device: int):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        torch_dtype=dtype,
        device=pipeline_device,
    )


def transcribe_batch(asr, wav_paths: list[Path], batch_size: int) -> list[str]:
    outputs = asr(
        [str(path) for path in wav_paths],
        batch_size=batch_size,
        return_timestamps=False,
    )
    texts = []
    for output in outputs:
        if isinstance(output, dict):
            texts.append(clean_text(str(output.get("text", ""))))
        else:
            texts.append(clean_text(str(output)))
    return texts


def resolve_mt_dtype(device: torch.device, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def load_mt(mt_model: str, src_lang: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(mt_model, src_lang=src_lang)
    model_kwargs = {"attn_implementation": "eager"}
    for dtype_key in ("dtype", "torch_dtype"):
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                mt_model,
                **{dtype_key: dtype},
                **model_kwargs,
            )
            break
        except TypeError:
            continue
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(mt_model, torch_dtype=dtype)
    model.to(device).eval()
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    return tokenizer, model


@torch.no_grad()
def translate_batch(
    tokenizer,
    model,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    device: torch.device,
    max_input_tokens: int,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    if not texts:
        return []
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_lang)
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos_token_id is None or forced_bos_token_id < 0:
        raise ValueError(f"Could not resolve target language token {tgt_lang!r}.")
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    output_ids = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
    )
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [clean_text(text) for text in decoded]


def maybe_comet(
    predictions: list[str],
    references: list[str],
    sources: list[str],
    model_name: str | None,
):
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
    predictions: list[str],
    references: list[str],
    sources: list[str],
    comet_model: str | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score if predictions else 0.0,
        "chrf": sacrebleu.corpus_chrf(predictions, [references]).score if predictions else 0.0,
        "wer_on_text_channel": wer(references, predictions) if predictions else 1.0,
        "count": len(predictions),
    }
    norm_preds = [normalize_text(text) for text in predictions]
    norm_refs = [normalize_text(text) for text in references]
    metrics.update(
        {
            "normalized_bleu": (
                sacrebleu.corpus_bleu(norm_preds, [norm_refs]).score if norm_preds else 0.0
            ),
            "normalized_chrf": (
                sacrebleu.corpus_chrf(norm_preds, [norm_refs]).score if norm_preds else 0.0
            ),
            "normalized_wer_on_text_channel": wer(norm_refs, norm_preds) if norm_preds else 1.0,
            "empty_prediction_rate": (
                sum(1 for prediction in predictions if not prediction.strip()) / len(predictions)
                if predictions
                else 1.0
            ),
        }
    )
    comet_score = maybe_comet(predictions, references, sources, comet_model)
    if comet_score is not None:
        metrics["comet"] = comet_score
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an ASR+MT cascade on the prepared split."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--asr-model", default="Sunbird/asr-whisper-large-v3-salt")
    parser.add_argument("--mt-model", default="Sunbird/translate-nllb-3.3b-salt")
    parser.add_argument("--src-lang", default="lug_Latn", help="NLLB source language code.")
    parser.add_argument("--tgt-lang", default="eng_Latn", help="NLLB target language code.")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--mt-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Torch dtype for the MT model. auto uses bf16 on supported CUDA GPUs.",
    )
    parser.add_argument("--asr-batch-size", type=int, default=8)
    parser.add_argument("--mt-batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--comet-model", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--metrics-path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["project"]["processed_dir"])
    rows = read_jsonl(processed_dir / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    require_source_wavs(rows, processed_dir, args.split)

    output_dir = Path(cfg["project"]["output_dir"]) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = Path(
        args.output_jsonl or output_dir / f"cascade_{args.split}_predictions.jsonl"
    )
    metrics_path = Path(args.metrics_path or output_dir / f"cascade_{args.split}_metrics.json")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    src_lang = resolve_nllb_lang(args.src_lang)
    tgt_lang = resolve_nllb_lang(args.tgt_lang)
    device, pipeline_device = resolve_device(args.device)
    mt_dtype = resolve_mt_dtype(device, args.mt_dtype)

    print(f"Loading ASR model: {args.asr_model}")
    asr = build_asr_pipeline(args.asr_model, device, pipeline_device)
    print(f"Loading MT model: {args.mt_model} ({src_lang} -> {tgt_lang}, {mt_dtype})")
    mt_tokenizer, mt_model = load_mt(args.mt_model, src_lang, device, mt_dtype)

    outputs: list[dict[str, Any]] = []
    t_asr_total = 0.0
    t_mt_total = 0.0
    for start in tqdm(range(0, len(rows), args.asr_batch_size), desc="Cascade"):
        batch_rows = rows[start : start + args.asr_batch_size]
        wav_paths = [source_wav_path(row, processed_dir, args.split) for row in batch_rows]

        t0 = time.time()
        asr_texts = transcribe_batch(asr, wav_paths, args.asr_batch_size)
        t_asr_total += time.time() - t0

        batch_predictions: list[str] = []
        t0 = time.time()
        for mt_start in range(0, len(asr_texts), args.mt_batch_size):
            mt_texts = asr_texts[mt_start : mt_start + args.mt_batch_size]
            batch_predictions.extend(
                translate_batch(
                    mt_tokenizer,
                    mt_model,
                    mt_texts,
                    src_lang,
                    tgt_lang,
                    device,
                    args.max_input_tokens,
                    args.max_new_tokens,
                    args.num_beams,
                )
            )
        t_mt_total += time.time() - t0

        zipped_outputs = zip(batch_rows, wav_paths, asr_texts, batch_predictions)
        for row, wav_path, asr_text, prediction in zipped_outputs:
            outputs.append(
                {
                    "id": row["id"],
                    "source_wav": str(wav_path),
                    "source": row.get("text_lug", ""),
                    "asr_text": asr_text,
                    "reference": row["text_eng"],
                    "prediction": prediction,
                }
            )

    predictions = [row["prediction"] for row in outputs]
    references = [row["reference"] for row in outputs]
    sources = [row["source"] for row in outputs]
    metrics = compute_metrics(predictions, references, sources, args.comet_model)
    metrics.update(
        {
            "asr_model": args.asr_model,
            "mt_model": args.mt_model,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "mt_dtype": str(mt_dtype).replace("torch.", ""),
            "asr_batch_size": args.asr_batch_size,
            "mt_batch_size": args.mt_batch_size,
            "num_beams": args.num_beams,
            "mean_asr_s_per_sample": t_asr_total / len(outputs) if outputs else 0.0,
            "mean_mt_s_per_sample": t_mt_total / len(outputs) if outputs else 0.0,
        }
    )

    with predictions_path.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
