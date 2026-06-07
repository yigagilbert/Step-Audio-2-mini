from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .constants import IGNORE_INDEX
from .formatting import StepAudioFormatter


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class PreparedSpeechDataset(Dataset):
    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)
        self.rows = read_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class StepAudioCollator:
    def __init__(
        self,
        formatter: StepAudioFormatter,
        pad_token_id: int,
        max_sequence_length: int = 16384,
    ) -> None:
        self.formatter = formatter
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        formatted = []
        mels = []
        for row in rows:
            mel = torch.load(row["src_mel_path"], map_location="cpu")
            if mel.ndim != 2:
                raise ValueError(f"Expected 2D mel tensor, got {tuple(mel.shape)} for {row['id']}")
            sample = self.formatter.format_sft(
                mel_frames=int(mel.shape[1]),
                text_eng=row.get("text_eng", ""),
                audio_tokens=row.get("target_audio_tokens", []),
            )
            if len(sample.input_ids) > self.max_sequence_length:
                sample.input_ids = sample.input_ids[: self.max_sequence_length]
                sample.labels = sample.labels[: self.max_sequence_length]
            formatted.append(sample)
            mels.append(mel)

        max_len = max(len(s.input_ids) for s in formatted)
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), max_len), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)

        for i, sample in enumerate(formatted):
            length = len(sample.input_ids)
            input_ids[i, :length] = torch.tensor(sample.input_ids, dtype=torch.long)
            labels[i, :length] = torch.tensor(sample.labels, dtype=torch.long)
            attention_mask[i, :length] = 1

        max_frames = max(m.shape[1] for m in mels)
        wavs = torch.zeros((len(rows), 128, max_frames), dtype=torch.float32)
        wav_lens = torch.zeros((len(rows),), dtype=torch.int32)
        for i, mel in enumerate(mels):
            wavs[i, :, : mel.shape[1]] = mel
            wav_lens[i] = max(1, int(mel.shape[1]) - 2)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "wavs": wavs,
            "wav_lens": wav_lens,
        }
