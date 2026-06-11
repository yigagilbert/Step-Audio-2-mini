from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data_prep import keep_row, load_config, safe_id, validate_schema  # noqa: E402
from stepaudio_luganda.audio import audio_cell_to_waveform, energy_trim, save_waveform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a small random set of validation audio samples from the configured "
            "Hugging Face dataset for quick VM/deployment testing."
        )
    )
    parser.add_argument("--config", default="configs/h100_nvl_fast_deepspeed.yaml")
    parser.add_argument("--split", default="validation", help="Configured split name to sample.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to project.seed from the config, or 42.",
    )
    parser.add_argument(
        "--output-dir",
        default="test_samples/validation_audio",
        help="Folder where WAV files and manifests will be written.",
    )
    parser.add_argument(
        "--no-filters",
        action="store_true",
        help="Do not apply the same duration/speech-ratio filters used by data_prep.py.",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not apply the configured energy VAD trim before saving WAVs.",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Save only Luganda source audio. By default, English reference audio is saved too.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than zero.")

    cfg = load_config(args.config)
    project_cfg = cfg["project"]
    dataset_cfg = cfg["dataset"]
    format_cfg = cfg["format"]
    audio_cfg = dataset_cfg["audio"]
    vad_cfg = dataset_cfg.get("vad", {})

    if args.split not in dataset_cfg["splits"]:
        raise ValueError(
            f"Unknown split '{args.split}'. Known splits: {sorted(dataset_cfg['splits'])}"
        )

    seed = args.seed if args.seed is not None else int(project_cfg.get("seed") or 42)
    sample_rate = int(audio_cfg.get("sample_rate") or 16000)
    token = os.environ.get(dataset_cfg.get("token_env", "HF_TOKEN"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_split = dataset_cfg["splits"][args.split]
    dataset = load_dataset(dataset_cfg["name"], split=hf_split, token=token)
    dataset = dataset.cast_column("audio_lug", Audio(sampling_rate=sample_rate, decode=False))
    include_target = not args.source_only and format_cfg.get("target_format") != "text_only"
    if include_target:
        dataset = dataset.cast_column("audio_eng", Audio(sampling_rate=sample_rate, decode=False))

    validate_schema(dataset, dataset_cfg["required_columns"])
    shuffled_indices = list(range(len(dataset)))
    import random

    random.Random(seed).shuffle(shuffled_indices)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    skipped = 0

    for idx in shuffled_indices:
        if len(records) >= args.count:
            break

        row = dataset[int(idx)]
        if not args.no_filters and not keep_row(row, dataset_cfg["filters"], format_cfg["target_format"]):
            skipped += 1
            continue

        row_id = safe_id(row.get("id", idx))
        if row_id in seen_ids:
            row_id = f"{row_id}_{idx}"
        seen_ids.add(row_id)

        sample_prefix = f"{len(records) + 1:02d}_{row_id}"
        lug_path = output_dir / f"{sample_prefix}.lug.wav"
        eng_path = output_dir / f"{sample_prefix}.eng.wav"

        try:
            lug_waveform = audio_cell_to_waveform(row["audio_lug"], target_rate=sample_rate)
            if not args.no_trim and vad_cfg.get("enabled", True):
                lug_waveform = energy_trim(lug_waveform, top_db=int(vad_cfg.get("top_db", 35)))
            save_waveform(lug_path, lug_waveform, sample_rate)

            saved_eng_path = None
            if include_target:
                eng_waveform = audio_cell_to_waveform(row["audio_eng"], target_rate=sample_rate)
                if not args.no_trim and vad_cfg.get("enabled", True):
                    eng_waveform = energy_trim(eng_waveform, top_db=int(vad_cfg.get("top_db", 35)))
                save_waveform(eng_path, eng_waveform, sample_rate)
                saved_eng_path = str(eng_path)

            records.append(
                {
                    "id": row_id,
                    "dataset": dataset_cfg["name"],
                    "split": args.split,
                    "hf_split": hf_split,
                    "row_index": int(idx),
                    "audio_lug": str(lug_path),
                    "audio_eng": saved_eng_path,
                    "text_lug": row.get("text_lug", ""),
                    "text_eng": row.get("text_eng", ""),
                    "src_dur_s": float(row.get("src_dur_s", 0.0) or 0.0),
                    "tgt_dur_s": float(row.get("tgt_dur_s", 0.0) or 0.0),
                    "sample_rate": sample_rate,
                }
            )
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping row {idx} ({row_id}): {exc}")

    if len(records) < args.count:
        print(f"[WARN] Requested {args.count} samples but only wrote {len(records)}.")

    write_jsonl(output_dir / "manifest.jsonl", records)
    write_json(
        output_dir / "manifest.json",
        {
            "dataset": dataset_cfg["name"],
            "split": args.split,
            "hf_split": hf_split,
            "count": len(records),
            "requested_count": args.count,
            "seed": seed,
            "sample_rate": sample_rate,
            "filters_applied": not args.no_filters,
            "trim_applied": not args.no_trim and vad_cfg.get("enabled", True),
            "source_only": args.source_only,
            "skipped": skipped,
            "records": records,
        },
    )

    print(f"Wrote {len(records)} samples to {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
