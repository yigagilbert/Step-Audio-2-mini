from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.data import read_jsonl  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()]
    return ("".join(keep)[:100]).strip("_") or "sample"


def read_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize cascade text predictions with SpeechT5."
    )
    parser.add_argument("--config", default="configs/h100_nvl_fast_deepspeed.yaml")
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Cascade predictions JSONL. Defaults to "
            "output_dir/eval/cascade_validation_predictions.jsonl."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--speaker-index", type=int, default=7306)
    parser.add_argument("--max-chars", type=int, default=450)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["project"]["output_dir"])
    processed_dir = Path(cfg["project"]["processed_dir"])
    predictions_path = Path(
        args.predictions or output_dir / "eval" / f"cascade_{args.split}_predictions.jsonl"
    )
    sample_dir = Path(args.output_dir or output_dir / "eval" / "cascade_audio_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)

    prepared_rows = {
        row["id"]: row
        for row in read_jsonl(processed_dir / f"{args.split}.jsonl")
    }
    predictions = read_prediction_jsonl(predictions_path)
    if args.limit is not None:
        predictions = predictions[: args.limit]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device).eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()
    speaker_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_index = min(max(args.speaker_index, 0), len(speaker_dataset) - 1)
    speaker_embeddings = torch.tensor(
        speaker_dataset[speaker_index]["xvector"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    manifest = []
    for idx, row in enumerate(tqdm(predictions, desc="Cascade TTS")):
        text = str(row.get("prediction") or row.get("hyp") or "").strip()
        if not text:
            continue
        text = text[: args.max_chars]
        inputs = processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.inference_mode():
            speech = model.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)
        row_id = str(row.get("id", idx))
        wav_path = sample_dir / f"{idx:04d}_{safe_name(row_id)}.wav"
        torchaudio.save(str(wav_path), speech.detach().cpu().unsqueeze(0), 16000)
        manifest.append(
            {
                "id": row_id,
                "hyp_audio": str(wav_path),
                "wav": str(wav_path),
                "ref_audio": str(processed_dir / args.split / "wav" / f"{row_id}.eng.wav"),
                "source": prepared_rows.get(row_id, {}).get("text_lug", row.get("source", "")),
                "reference": row.get("reference") or row.get("ref"),
                "prediction": row.get("prediction") or row.get("hyp"),
                "tts_model": "microsoft/speecht5_tts",
                "speaker_index": speaker_index,
            }
        )

    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_jsonl_path = sample_dir / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(manifest)} wav files to {sample_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote JSONL manifest to {manifest_jsonl_path}")


if __name__ == "__main__":
    main()
