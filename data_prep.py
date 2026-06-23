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
    if row.get("audio_lug") is None:
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


def direction_prompt(cfg: dict[str, Any], direction: dict[str, Any]) -> str:
    if direction.get("system_prompt"):
        return str(direction["system_prompt"])
    template = cfg["format"].get("system_prompt_template")
    if template:
        return str(template).format(
            source_language=direction["source_language"],
            target_language=direction["target_language"],
        )
    return str(cfg["format"]["system_prompt"])


def default_direction_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "lug_to_eng",
            "source_audio": "audio_lug",
            "target_audio": "audio_eng",
            "source_text": "text_lug",
            "target_text": "text_eng",
            "source_language": "Luganda",
            "target_language": "English",
            "source_wav_suffix": "lug",
            "target_wav_suffix": "eng",
        }
    ]


def direction_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    directions = cfg["dataset"].get("preprocessing", {}).get("directions")
    if not directions:
        return default_direction_specs()
    required = {
        "name",
        "source_audio",
        "target_audio",
        "source_text",
        "target_text",
        "source_language",
        "target_language",
    }
    normalized = []
    for direction in directions:
        missing = sorted(required - set(direction))
        if missing:
            raise ValueError(f"Direction {direction!r} is missing required keys: {missing}")
        item = dict(direction)
        item.setdefault("source_wav_suffix", safe_id(item["source_language"]).lower())
        item.setdefault("target_wav_suffix", safe_id(item["target_language"]).lower())
        normalized.append(item)
    return normalized


def waveform_for_field(
    row: dict[str, Any],
    field: str,
    dataset_cfg: dict[str, Any],
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    if field not in cache:
        waveform = audio_cell_to_waveform(
            row[field],
            target_rate=dataset_cfg["audio"]["sample_rate"],
        )
        if dataset_cfg["vad"].get("enabled", True):
            waveform = energy_trim(waveform, top_db=int(dataset_cfg["vad"].get("top_db", 35)))
        cache[field] = waveform
    return cache[field]


def prepare_split(
    split_name: str,
    dataset,
    cfg: dict[str, Any],
    target_tokenizer: TargetSpeechTokenizer | None,
    tokenize_target_audio: bool = True,
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
    directions = direction_specs(cfg)

    for row in tqdm(dataset, desc=f"Preparing {split_name}"):
        if max_samples and len(rows) >= int(max_samples):
            break
        if not keep_row(row, dataset_cfg["filters"], target_format):
            skipped += 1
            continue
        base_id = safe_id(row.get("id", len(rows)))
        waveform_cache: dict[str, torch.Tensor] = {}
        try:
            for direction in directions:
                row_id = f"{base_id}__{safe_id(direction['name'])}"
                if row_id in seen_ids:
                    row_id = f"{row_id}_{len(rows)}"
                seen_ids.add(row_id)

                src = waveform_for_field(row, direction["source_audio"], dataset_cfg, waveform_cache)
                src_mel = log_mel_spectrogram(src, n_mels=128, padding=479)
                src_mel_path = mel_dir / f"{row_id}.pt"
                torch.save(src_mel.cpu(), src_mel_path)

                target_audio_tokens: list[int] = []
                target_wav_path = ""
                if target_format != "text_only":
                    if tokenize_target_audio and target_tokenizer is None:
                        raise RuntimeError("target_format requires target audio tokenization.")
                    tgt = waveform_for_field(row, direction["target_audio"], dataset_cfg, waveform_cache)
                    if tokenize_target_audio:
                        target_audio_tokens = target_tokenizer.encode(tgt)
                    if dataset_cfg["audio"].get("save_resampled_wavs", True):
                        target_wav_path = str(wav_dir / f"{row_id}.target.wav")
                        save_waveform(
                            target_wav_path,
                            tgt,
                            dataset_cfg["audio"]["sample_rate"],
                        )
                        save_waveform(
                            wav_dir / f"{row_id}.{direction['target_wav_suffix']}.wav",
                            tgt,
                            dataset_cfg["audio"]["sample_rate"],
                        )

                source_wav_path = ""
                if dataset_cfg["audio"].get("save_resampled_wavs", True):
                    source_wav_path = str(wav_dir / f"{row_id}.source.wav")
                    save_waveform(
                        source_wav_path,
                        src,
                        dataset_cfg["audio"]["sample_rate"],
                    )
                    save_waveform(
                        wav_dir / f"{row_id}.{direction['source_wav_suffix']}.wav",
                        src,
                        dataset_cfg["audio"]["sample_rate"],
                    )

                rows.append(
                    {
                        "id": row_id,
                        "base_id": base_id,
                        "direction": direction["name"],
                        "source_language": direction["source_language"],
                        "target_language": direction["target_language"],
                        "system_prompt": direction_prompt(cfg, direction),
                        "src_mel_path": str(src_mel_path),
                        "src_mel_frames": int(src_mel.shape[1]),
                        "source_wav_path": source_wav_path,
                        "target_wav_path": target_wav_path,
                        "target_audio_tokens": target_audio_tokens,
                        "source_text": str(row.get(direction["source_text"], "")).strip(),
                        "target_text": str(row.get(direction["target_text"], "")).strip(),
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
            print(f"[WARN] Skipping {base_id}: {exc}")

    write_jsonl(processed_dir / f"{split_name}.jsonl", rows)
    print(f"{split_name}: wrote {len(rows)} rows, skipped {skipped}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated output split names to prepare, for example validation.",
    )
    parser.add_argument(
        "--skip-target-tokenization",
        action="store_true",
        help="Save target wavs but skip target audio-token extraction; useful for eval-only prep.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    dataset_cfg = cfg["dataset"]
    token = os.environ.get(dataset_cfg.get("token_env", "HF_TOKEN"))
    selected_splits = None
    if args.splits:
        selected_splits = {split.strip() for split in args.splits.split(",") if split.strip()}
        unknown_splits = selected_splits - set(dataset_cfg["splits"])
        if unknown_splits:
            raise ValueError(
                f"Unknown split(s) requested: {sorted(unknown_splits)}. "
                f"Known output splits: {sorted(dataset_cfg['splits'])}"
            )

    target_format = cfg["format"]["target_format"]
    target_tokenizer = None
    tokenize_target_audio = not args.skip_target_tokenization
    if target_format != "text_only" and tokenize_target_audio:
        target_tokenizer = TargetSpeechTokenizer(
            cfg["model"].get("local_path") or cfg["model"]["name_or_path"],
            args.device,
        )

    if selected_splits is None:
        dataset = load_dataset(dataset_cfg["name"], token=token)
        if not isinstance(dataset, DatasetDict):
            raise ValueError("Expected a DatasetDict with train/validation/test splits.")
        split_items = [
            (out_split, hf_split, dataset[hf_split])
            for out_split, hf_split in dataset_cfg["splits"].items()
            if hf_split in dataset
        ]
    else:
        split_items = [
            (
                out_split,
                dataset_cfg["splits"][out_split],
                load_dataset(
                    dataset_cfg["name"],
                    split=dataset_cfg["splits"][out_split],
                    token=token,
                ),
            )
            for out_split in dataset_cfg["splits"]
            if out_split in selected_splits
        ]

    for out_split, hf_split, split in split_items:
        split = split.cast_column(
            "audio_lug",
            Audio(sampling_rate=dataset_cfg["audio"]["sample_rate"], decode=False),
        )
        if target_format != "text_only":
            split = split.cast_column(
                "audio_eng",
                Audio(sampling_rate=dataset_cfg["audio"]["sample_rate"], decode=False),
            )
        validate_schema(split, dataset_cfg["required_columns"])
        prepare_split(
            out_split,
            split,
            cfg,
            target_tokenizer,
            tokenize_target_audio=tokenize_target_audio,
        )


if __name__ == "__main__":
    main()
