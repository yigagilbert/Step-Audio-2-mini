from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

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


def existing_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:128].startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def validate_token2wav_assets(model_path: Path) -> None:
    onnx_path = model_path / "token2wav" / "speech_tokenizer_v2_25hz.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Missing {onnx_path}. Run `git -C {model_path} lfs pull` or pass "
            "--base-model stepfun-ai/Step-Audio-2-mini."
        )
    if is_lfs_pointer(onnx_path) or onnx_path.stat().st_size < 1024 * 1024:
        raise RuntimeError(
            f"{onnx_path} is not a real ONNX file. It is likely a Git LFS pointer "
            "or incomplete download. Run `git lfs install && "
            f"git -C {model_path} lfs pull`, or pass "
            "--base-model stepfun-ai/Step-Audio-2-mini."
        )


def resolve_base_model_path(model_ref: str) -> Path:
    local_path = existing_path(model_ref)
    if local_path:
        return local_path
    print(f"Downloading or reusing cached token2wav assets from: {model_ref}")
    return Path(
        snapshot_download(
            repo_id=model_ref,
            allow_patterns=["token2wav/*", "assets/*"],
        )
    )


def choose_base_model_path(cfg: dict[str, Any], override: str | None) -> Path:
    refs = []
    if override:
        refs.append(override)
    else:
        local_ref = cfg["model"].get("local_path")
        if local_ref:
            refs.append(str(local_ref))
        refs.append(str(cfg["model"]["name_or_path"]))

    errors = []
    for ref in refs:
        model_path = resolve_base_model_path(ref)
        try:
            validate_token2wav_assets(model_path)
            return model_path
        except Exception as exc:
            errors.append(f"{ref}: {exc}")
            if override:
                raise
            print(f"[WARN] Could not use base model assets from {ref}: {exc}")
    raise RuntimeError("No usable token2wav assets found:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base Step-Audio model path or Hub repo ID; defaults to config local_path/name.",
    )
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

    model_path = choose_base_model_path(cfg, args.base_model)
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
