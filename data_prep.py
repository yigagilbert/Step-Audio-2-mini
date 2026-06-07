from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Audio, DatasetDict, load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.audio import (  # noqa: E402
    audio_cell_to_waveform,
    energy_trim,
    log_mel_spectrogram,
    save_waveform,
)
from stepaudio_luganda.data import write_jsonl  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_id(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "sample"


class TargetSpeechTokenizer:
    def __init__(self, model_path: str | Path, device: str = "cuda") -> None:
        import s3tokenizer

        self.s3tokenizer = s3tokenizer
        tokenizer_path = Path(model_path) / "token2wav" / "speech_tokenizer_v2_25hz.onnx"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Missing {tokenizer_path}. Clone or download stepfun-ai/Step-Audio-2-mini first."
            )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.audio_tokenizer = s3tokenizer.load_model(str(tokenizer_path))
        if self.device.type == "cuda" and hasattr(self.audio_tokenizer, "cuda"):
            self.audio_tokenizer = self.audio_tokenizer.cuda()
        elif hasattr(self.audio_tokenizer, "to"):
            self.audio_tokenizer = self.audio_tokenizer.to(self.device)
        self.audio_tokenizer = self.audio_tokenizer.eval()

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> list[int]:
        audio = waveform.detach().cpu().to(torch.float32)
        mels = self.s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = self.s3tokenizer.padding([mels])
        tokens, token_lens = self.audio_tokenizer.quantize(
            mels.to(self.device),
            mels_lens.to(self.device),
        )
        length = int(token_lens[0]) if token_lens is not None else int(tokens.shape[1])
        return [int(x) for x in tokens[0, :length].detach().cpu().tolist()]


def validate_schema(dataset, required_columns: list[str]) -> None:
    columns = set(dataset.column_names)
    missing = [col for col in required_columns if col not in columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def keep_row(row: dict[str, Any], filters: dict[str, Any], target_format: str) -> bool:
    if not row.get("text_lug") or not row.get("text_eng"):
        return False
    checks = [
        float(row.get("dur_ratio", 0.0)) >= float(filters["min_dur_ratio"]),
        float(row.get("dur_ratio", 999.0)) <= float(filters["max_dur_ratio"]),
        float(row.get("src_speech_ratio", 0.0)) >= float(filters["min_src_speech_ratio"]),
        float(row.get("tgt_speech_ratio", 0.0)) >= float(filters["min_tgt_speech_ratio"]),
        float(row.get("src_dur_s", 999.0)) <= float(filters["max_src_dur_s"]),
        float(row.get("tgt_dur_s", 999.0)) <= float(filters["max_tgt_dur_s"]),
    ]
    if target_format != "text_only":
        checks.append(row.get("audio_eng") is not None)
    return all(checks)


def prepare_split(
    split_name: str,
    dataset,
    cfg: dict[str, Any],
    target_tokenizer: TargetSpeechTokenizer | None,
) -> list[dict[str, Any]]:
    project = cfg["project"]
    dataset_cfg = cfg["dataset"]
    target_format = cfg["format"]["target_format"]
    processed_dir = Path(project["processed_dir"])
    split_dir = processed_dir / split_name
    mel_dir = split_dir / "mels"
    wav_dir = split_dir / "wav"
    mel_dir.mkdir(parents=True, exist_ok=True)
    if dataset_cfg["audio"].get("save_resampled_wavs", True):
        wav_dir.mkdir(parents=True, exist_ok=True)

    max_samples = dataset_cfg["preprocessing"].get("max_samples_per_split", {}).get(split_name)
    rows: list[dict[str, Any]] = []
    skipped = 0
    seen_ids: set[str] = set()

    for row in tqdm(dataset, desc=f"Preparing {split_name}"):
        if max_samples and len(rows) >= int(max_samples):
            break
        if not keep_row(row, dataset_cfg["filters"], target_format):
            skipped += 1
            continue
        row_id = safe_id(row.get("id", len(rows)))
        if row_id in seen_ids:
            row_id = f"{row_id}_{len(rows)}"
        seen_ids.add(row_id)
        try:
            src = audio_cell_to_waveform(row["audio_lug"], target_rate=dataset_cfg["audio"]["sample_rate"])
            if dataset_cfg["vad"].get("enabled", True):
                src = energy_trim(src, top_db=int(dataset_cfg["vad"].get("top_db", 35)))
            src_mel = log_mel_spectrogram(src, n_mels=128, padding=479)
            src_mel_path = mel_dir / f"{row_id}.pt"
            torch.save(src_mel.cpu(), src_mel_path)

            target_audio_tokens: list[int] = []
            if target_format != "text_only":
                if target_tokenizer is None:
                    raise RuntimeError("target_format requires target audio tokenization.")
                tgt = audio_cell_to_waveform(row["audio_eng"], target_rate=dataset_cfg["audio"]["sample_rate"])
                if dataset_cfg["vad"].get("enabled", True):
                    tgt = energy_trim(tgt, top_db=int(dataset_cfg["vad"].get("top_db", 35)))
                target_audio_tokens = target_tokenizer.encode(tgt)
                if dataset_cfg["audio"].get("save_resampled_wavs", True):
                    save_waveform(wav_dir / f"{row_id}.eng.wav", tgt, dataset_cfg["audio"]["sample_rate"])
            if dataset_cfg["audio"].get("save_resampled_wavs", True):
                save_waveform(wav_dir / f"{row_id}.lug.wav", src, dataset_cfg["audio"]["sample_rate"])

            rows.append(
                {
                    "id": row_id,
                    "src_mel_path": str(src_mel_path),
                    "src_mel_frames": int(src_mel.shape[1]),
                    "target_audio_tokens": target_audio_tokens,
                    "text_lug": row["text_lug"],
                    "text_eng": row["text_eng"],
                    "src_dur_s": float(row.get("src_dur_s", 0.0)),
                    "tgt_dur_s": float(row.get("tgt_dur_s", 0.0)),
                    "dur_ratio": float(row.get("dur_ratio", 0.0)),
                    "src_speech_ratio": float(row.get("src_speech_ratio", 0.0)),
                    "tgt_speech_ratio": float(row.get("tgt_speech_ratio", 0.0)),
                }
            )
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping {row_id}: {exc}")

    write_jsonl(processed_dir / f"{split_name}.jsonl", rows)
    print(f"{split_name}: wrote {len(rows)} rows, skipped {skipped}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg["dataset"]
    token = os.environ.get(dataset_cfg.get("token_env", "HF_TOKEN"))
    dataset = load_dataset(dataset_cfg["name"], token=token)
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a DatasetDict with train/validation/test splits.")

    target_format = cfg["format"]["target_format"]
    target_tokenizer = None
    if target_format != "text_only":
        target_tokenizer = TargetSpeechTokenizer(cfg["model"].get("local_path") or cfg["model"]["name_or_path"], args.device)

    for out_split, hf_split in dataset_cfg["splits"].items():
        if hf_split not in dataset:
            print(f"[WARN] Dataset split {hf_split!r} not found; skipping {out_split}.")
            continue
        split = dataset[hf_split].cast_column("audio_lug", Audio(sampling_rate=dataset_cfg["audio"]["sample_rate"]))
        if target_format != "text_only":
            split = split.cast_column("audio_eng", Audio(sampling_rate=dataset_cfg["audio"]["sample_rate"]))
        validate_schema(split, dataset_cfg["required_columns"])
        prepare_split(out_split, split, cfg, target_tokenizer)


if __name__ == "__main__":
    main()
