from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize audio for the best checkpoints selected by eval_rank_ablation.py "
            "and run speech-level metrics on the resulting manifests."
        )
    )
    parser.add_argument("--summary", required=True, help="summary.json from eval_rank_ablation.py.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <summary_dir>/best_audio_eval.",
    )
    parser.add_argument("--stepaudio2-repo", default="Step-Audio2")
    parser.add_argument("--base-model", default=None)
    parser.add_argument(
        "--prepared-jsonl",
        default=None,
        help="Prepared validation JSONL for aligned advanced metrics.",
    )
    parser.add_argument("--split", default=None, help="Override split from summary metadata.")
    parser.add_argument("--limit", type=int, default=None, help="Number of WAVs to synthesize per system.")
    parser.add_argument("--min-audio-tokens", type=int, default=None)
    parser.add_argument("--speech-model", default="microsoft/wavlm-large")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Optional system names to include, e.g. r16 r32 main-r64.",
    )
    parser.add_argument(
        "--extra-audio-system",
        action="append",
        default=[],
        metavar="NAME=MANIFEST",
        help="Additional pre-synthesized audio manifest to include in speech metrics.",
    )
    parser.add_argument(
        "--extra-text-system",
        action="append",
        default=[],
        metavar="NAME=PREDICTIONS",
        help="Additional text prediction JSONL to include in BLASER text metrics.",
    )
    parser.add_argument(
        "--run-blaser",
        action="store_true",
        help="Also run BLASER on the selected text prediction JSONLs.",
    )
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-speech-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "rows" not in data:
        raise ValueError(f"Expected summary object with rows: {path}")
    return data


def display_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_and_log(cmd: list[str], log_path: Path, dry_run: bool = False) -> None:
    print(display_cmd(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(display_cmd(cmd) + "\n", encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {display_cmd(cmd)}")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=path")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("System name cannot be empty.")
    return name, Path(raw_path).expanduser()


def selected_rows(summary: dict[str, Any], systems: set[str] | None) -> list[dict[str, Any]]:
    rows = []
    for row in summary["rows"]:
        if row.get("kind") != "adapter":
            continue
        if not row.get("is_best_for_rank"):
            continue
        if systems and row.get("system") not in systems:
            continue
        rows.append(row)
    if not rows:
        raise ValueError("No selected best checkpoint rows found in summary.")
    return rows


def default_prepared_jsonl(rows: list[dict[str, Any]], split: str) -> Path:
    first_config = Path(str(rows[0]["config"]))
    if not first_config.is_absolute():
        first_config = ROOT / first_config
    if first_config.exists():
        try:
            import yaml

            cfg = yaml.safe_load(first_config.read_text(encoding="utf-8"))
            processed_dir = Path(cfg["project"]["processed_dir"])
            if not processed_dir.is_absolute():
                processed_dir = ROOT / processed_dir
            return processed_dir / f"{split}.jsonl"
        except Exception:
            pass
    return ROOT / "data" / "processed" / "luganda_english_cleaned_v1" / f"{split}.jsonl"


def synthesize_row(
    row: dict[str, Any],
    output_dir: Path,
    stepaudio2_repo: str,
    base_model: str | None,
    split: str,
    limit: int | None,
    min_audio_tokens: int,
    dry_run: bool,
) -> Path:
    system_name = str(row["system"])
    checkpoint = str(row["checkpoint"])
    audio_dir = output_dir / "audio" / f"{system_name}__{checkpoint}"
    cmd = [
        sys.executable,
        "scripts/synthesize_eval_audio.py",
        "--config",
        str(row["config"]),
        "--predictions",
        str(row["predictions_path"]),
        "--stepaudio2-repo",
        stepaudio2_repo,
        "--output-dir",
        str(audio_dir),
        "--split",
        split,
        "--min-audio-tokens",
        str(min_audio_tokens),
    ]
    if base_model:
        cmd.extend(["--base-model", base_model])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    run_and_log(cmd, output_dir / "logs" / f"synthesize_{system_name}__{checkpoint}.log", dry_run)
    return audio_dir / "manifest.jsonl"


def write_selection(
    rows: list[dict[str, Any]],
    manifest_paths: dict[str, Path],
    output_dir: Path,
) -> None:
    payload = []
    for row in rows:
        item = dict(row)
        item["audio_manifest_path"] = str(manifest_paths.get(str(row["system"]), ""))
        payload.append(item)
    (output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in payload for key in row.keys()})
    with (output_dir / "selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload)


def run_speech_metrics(
    manifests: dict[str, Path],
    output_dir: Path,
    prepared_jsonl: Path,
    speech_model: str,
    device: str | None,
    dry_run: bool,
) -> None:
    if not manifests:
        raise ValueError("No audio manifests were provided for speech metrics.")
    cmd = [
        sys.executable,
        "eval_advanced_metrics.py",
        "--output",
        str(output_dir / "speech_metrics.json"),
        "--prepared-jsonl",
        str(prepared_jsonl),
        "--speech-model",
        speech_model,
        "--align-ids",
        "--skip-blaser",
    ]
    if device:
        cmd.extend(["--device", device])
    for name, manifest in manifests.items():
        cmd.extend(["--system", f"{name}={manifest}"])
    run_and_log(cmd, output_dir / "logs" / "speech_metrics.log", dry_run)


def run_blaser_metrics(
    prediction_paths: dict[str, Path],
    output_dir: Path,
    prepared_jsonl: Path,
    device: str | None,
    dry_run: bool,
) -> None:
    if not prediction_paths:
        raise ValueError("No prediction JSONLs were provided for BLASER metrics.")
    cmd = [
        sys.executable,
        "eval_advanced_metrics.py",
        "--output",
        str(output_dir / "blaser_metrics.json"),
        "--prepared-jsonl",
        str(prepared_jsonl),
        "--align-ids",
        "--skip-speechbertscore",
        "--skip-mcd",
    ]
    if device:
        cmd.extend(["--device", device])
    for name, predictions in prediction_paths.items():
        cmd.extend(["--system", f"{name}={predictions}"])
    run_and_log(cmd, output_dir / "logs" / "blaser_metrics.log", dry_run)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).expanduser()
    summary = read_summary(summary_path)
    summary_dir = summary_path.parent
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else summary_dir / "best_audio_eval"
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = set(args.systems) if args.systems else None
    rows = selected_rows(summary, systems)
    metadata = summary.get("metadata", {})
    split = args.split or str(metadata.get("split", "validation"))
    min_audio_tokens = args.min_audio_tokens or int(metadata.get("min_audio_tokens", 10))
    prepared_jsonl = (
        Path(args.prepared_jsonl).expanduser()
        if args.prepared_jsonl
        else default_prepared_jsonl(rows, split)
    )
    if not prepared_jsonl.exists() and not args.dry_run:
        raise FileNotFoundError(f"Missing prepared JSONL: {prepared_jsonl}")

    manifests: dict[str, Path] = {}
    if not args.skip_synthesis:
        for row in rows:
            manifests[str(row["system"])] = synthesize_row(
                row=row,
                output_dir=output_dir,
                stepaudio2_repo=args.stepaudio2_repo,
                base_model=args.base_model,
                split=split,
                limit=args.limit,
                min_audio_tokens=min_audio_tokens,
                dry_run=args.dry_run,
            )
    else:
        for row in rows:
            manifest = output_dir / "audio" / f"{row['system']}__{row['checkpoint']}" / "manifest.jsonl"
            manifests[str(row["system"])] = manifest

    for value in args.extra_audio_system:
        name, path = parse_named_path(value)
        manifests[name] = path

    prediction_paths = {str(row["system"]): Path(str(row["predictions_path"])) for row in rows}
    for value in args.extra_text_system:
        name, path = parse_named_path(value)
        prediction_paths[name] = path

    write_selection(rows, manifests, output_dir)

    if not args.skip_speech_metrics:
        run_speech_metrics(
            manifests=manifests,
            output_dir=output_dir,
            prepared_jsonl=prepared_jsonl,
            speech_model=args.speech_model,
            device=args.device,
            dry_run=args.dry_run,
        )

    if args.run_blaser:
        run_blaser_metrics(
            prediction_paths=prediction_paths,
            output_dir=output_dir,
            prepared_jsonl=prepared_jsonl,
            device=args.device,
            dry_run=args.dry_run,
        )

    print(f"Wrote selection files to {output_dir}")


if __name__ == "__main__":
    main()
