from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.paths import resolve_prompt_wav  # noqa: E402
from stepaudio_luganda.data import read_jsonl as read_prepared_jsonl  # noqa: E402
from stepaudio_luganda.torchcodec_compat import patch_torchaudio_bytesio_save  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:100] or "sample"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Eval JSONL from eval.py; defaults to output_dir/eval/validation_predictions.jsonl.",
    )
    parser.add_argument("--stepaudio2-repo", default="Step-Audio2")
    parser.add_argument("--prompt-wav", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-audio-tokens", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["project"]["output_dir"])
    processed_dir = Path(cfg["project"]["processed_dir"])
    predictions_path = Path(
        args.predictions or output_dir / "eval" / "validation_predictions.jsonl"
    )
    sample_dir = Path(args.output_dir or output_dir / "eval" / "audio_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = {
        row["id"]: row
        for row in read_prepared_jsonl(processed_dir / f"{args.split}.jsonl")
    }

    stepaudio2_repo = Path(args.stepaudio2_repo).resolve()
    sys.path.insert(0, str(stepaudio2_repo))
    patch_torchaudio_bytesio_save()
    from token2wav import Token2wav  # noqa: PLC0415

    model_path = Path(cfg["model"].get("local_path") or cfg["model"]["name_or_path"])
    token2wav = Token2wav(str(model_path / "token2wav"))
    prompt_wav = resolve_prompt_wav(
        args.prompt_wav or cfg["generation"]["prompt_wav"],
        model_path=model_path,
        stepaudio2_repo=stepaudio2_repo,
        root=ROOT,
    )
    print(f"Using prompt wav: {prompt_wav}")

    written = 0
    manifest = []
    for idx, row in enumerate(read_jsonl(predictions_path)):
        audio_tokens = [int(token) for token in row.get("audio_tokens", [])]
        if len(audio_tokens) < args.min_audio_tokens:
            continue
        name = f"{idx:04d}_{safe_name(str(row.get('id', idx)))}.wav"
        wav_path = sample_dir / name
        wav_path.write_bytes(token2wav(audio_tokens, prompt_wav=str(prompt_wav)))
        manifest.append(
            {
                "id": row.get("id"),
                "hyp_audio": str(wav_path),
                "wav": str(wav_path),
                "ref_audio": str(
                    processed_dir / args.split / "wav" / f"{row.get('id')}.eng.wav"
                ),
                "source": prepared_rows.get(str(row.get("id")), {}).get("text_lug", ""),
                "reference": row.get("reference"),
                "prediction": row.get("prediction"),
                "audio_tokens": len(audio_tokens),
            }
        )
        written += 1
        if args.limit is not None and written >= args.limit:
            break

    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_jsonl_path = sample_dir / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {written} wav files to {sample_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote JSONL manifest to {manifest_jsonl_path}")


if __name__ == "__main__":
    main()
