from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sacrebleu
import torch
import yaml
from jiwer import wer
from peft import PeftModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.data import read_jsonl  # noqa: E402
from stepaudio_luganda.formatting import StepAudioFormatter  # noqa: E402
from stepaudio_luganda.modeling import load_model, load_tokenizer  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_prediction_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text.splitlines()[0].strip()


def extract_outputs(
    token_ids: list[int],
    formatter: StepAudioFormatter,
) -> tuple[list[int], list[int]]:
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
    return text_ids, audio_tokens


@torch.no_grad()
def generate_one(model, tokenizer, formatter, row, cfg, device: torch.device) -> dict[str, Any]:
    mel = torch.load(row["src_mel_path"], map_location="cpu")
    prompt_ids = formatter.build_prompt(int(mel.shape[1]))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    wavs = mel.unsqueeze(0).to(device=device, dtype=torch.float32)
    wav_lens = torch.tensor([max(1, int(mel.shape[1]) - 2)], dtype=torch.int32, device=device)
    gen_cfg = cfg["generation"]
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        wavs=wavs,
        wav_lens=wav_lens,
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        temperature=float(gen_cfg["temperature"]),
        top_p=float(gen_cfg["top_p"]),
        repetition_penalty=float(gen_cfg["repetition_penalty"]),
        do_sample=bool(gen_cfg["do_sample"]),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=formatter.eot_id,
    )
    new_ids = output[0, len(prompt_ids) :].detach().cpu().tolist()
    text_ids, audio_tokens = extract_outputs(new_ids, formatter)
    pred_text = clean_prediction_text(tokenizer.decode(text_ids, skip_special_tokens=True))
    return {
        "id": row["id"],
        "reference": row["text_eng"],
        "prediction": pred_text,
        "audio_tokens": audio_tokens,
    }


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


def should_load_adapter(adapter_path: str, default_adapter_path: str) -> bool:
    if Path(adapter_path).exists():
        return True
    return adapter_path != default_adapter_path


def choose_base_model_path(cfg: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return cfg["model"].get("local_path") or cfg["model"]["name_or_path"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base Step-Audio model path or Hub repo ID; defaults to config local_path/name.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter path or Hub repo ID; defaults to output_dir/final.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--comet-model", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["project"]["processed_dir"])
    rows = read_jsonl(processed_dir / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    model_path = choose_base_model_path(cfg, args.base_model)
    tokenizer = load_tokenizer(
        model_path,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    formatter = StepAudioFormatter(
        tokenizer,
        system_prompt=cfg["format"]["system_prompt"],
        target_format=cfg["format"]["target_format"],
        max_target_audio_tokens=cfg["format"].get("max_target_audio_tokens"),
    )
    model = load_model(model_path, cfg["model"])
    default_adapter_path = str(Path(cfg["project"]["output_dir"]) / "final")
    adapter_path = args.adapter or default_adapter_path
    if should_load_adapter(adapter_path, default_adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    outputs = []
    for row in tqdm(rows, desc="Generating"):
        outputs.append(generate_one(model, tokenizer, formatter, row, cfg, device))

    preds = [o["prediction"] for o in outputs]
    refs = [o["reference"] for o in outputs]
    srcs = [row.get("text_lug", "") for row in rows]
    metrics = {
        "bleu": sacrebleu.corpus_bleu(preds, [refs]).score if preds else 0.0,
        "chrf": sacrebleu.corpus_chrf(preds, [refs]).score if preds else 0.0,
        "wer_on_text_channel": wer(refs, preds) if preds else 1.0,
        "count": len(outputs),
    }
    comet_score = maybe_comet(preds, refs, srcs, args.comet_model)
    if comet_score is not None:
        metrics["comet"] = comet_score

    out_dir = Path(cfg["project"]["output_dir"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{args.split}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / f"{args.split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
