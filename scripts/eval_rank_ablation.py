from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

HIGHER_IS_BETTER = {
    "bleu",
    "chrf",
    "normalized_bleu",
    "normalized_chrf",
    "comet",
    "blaser_2_0_ref",
    "blaser_2_0_qe",
    "valid_audio_token_rate",
}
LOWER_IS_BETTER = {
    "wer_on_text_channel",
    "normalized_wer_on_text_channel",
    "empty_prediction_rate",
}
BEST_FALLBACKS = ("comet", "chrf", "bleu", "valid_audio_token_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrospective Step-Audio rank-ablation evaluations over a YAML/JSON plan. "
            "Each checkpoint gets unique prediction, metric, and log files."
        )
    )
    parser.add_argument("--plan", help="YAML/JSON evaluation plan.")
    parser.add_argument(
        "--write-template",
        help="Write an editable rank-ablation evaluation plan template and exit.",
    )
    parser.add_argument("--output-dir", help="Override plan output_dir.")
    parser.add_argument("--split", help="Override plan split.")
    parser.add_argument("--limit", type=int, help="Override plan limit.")
    parser.add_argument("--comet-model", help="Override plan comet_model.")
    parser.add_argument(
        "--best-metric",
        default=None,
        help="Metric used to select best checkpoint per rank. Defaults to comet if present, then chrF/BLEU.",
    )
    parser.add_argument(
        "--min-audio-tokens",
        type=int,
        default=None,
        help="Minimum generated audio tokens counted as valid for audio-token-rate summaries.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing prediction/metric files when both exist.",
    )
    parser.add_argument(
        "--include-cascade",
        action="store_true",
        help="Run the optional cascade block from the plan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running model evaluation.",
    )
    return parser.parse_args()


def load_structured(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing plan file: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML plans. Use JSON or install pyyaml.") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Plan must be a mapping/object: {path}")
    return data


def write_template(path: Path) -> None:
    template = """# Edit paths to match your VM. All evaluations use the same split/limit/decoding
# settings from each model config, and every output is written under output_dir.
output_dir: outputs/ablation/rank_eval_200
split: validation
limit: 200
comet_model: Unbabel/wmt22-comet-da
min_audio_tokens: 20
best_metric: comet

systems:
  - name: r16
    rank: 16
    config: /tmp/lora_r16_single_5000.yaml
    trainable_params: 38535168
    trainable_percent: 0.4613
    checkpoints:
      - name: checkpoint-3000
        adapter: outputs/ablation/lora-r16-stable-5k/checkpoint-3000
      - name: checkpoint-4000
        adapter: outputs/ablation/lora-r16-stable-5k/checkpoint-4000
      - name: checkpoint-5000
        adapter: outputs/ablation/lora-r16-stable-5k/checkpoint-5000
      - name: final
        adapter: outputs/ablation/lora-r16-stable-5k/final

  - name: r32
    rank: 32
    config: /tmp/lora_r32_single_5000.yaml
    trainable_params: 77070336
    trainable_percent: 0.9186
    checkpoints:
      - name: checkpoint-3000
        adapter: outputs/ablation/lora-r32-stable-5k/checkpoint-3000
      - name: checkpoint-4000
        adapter: outputs/ablation/lora-r32-stable-5k/checkpoint-4000
      - name: checkpoint-5000
        adapter: outputs/ablation/lora-r32-stable-5k/checkpoint-5000
      - name: final
        adapter: outputs/ablation/lora-r32-stable-5k/final

  - name: main-r64
    rank: 64
    config: configs/h100_nvl_fast_deepspeed.yaml
    trainable_params: 154140672
    trainable_percent: 1.82
    checkpoints:
      - name: hf-final
        adapter: yigagilbert/stepaudio2-mini-luganda-english-s2st-lora

cascade:
  enabled: false
  name: cascade
  config: configs/h100_nvl_fast_deepspeed.yaml
  asr_model: Sunbird/asr-whisper-large-v3-salt
  mt_model: Sunbird/translate-nllb-3.3b-salt
  src_lang: lug_Latn
  tgt_lang: eng_Latn
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")
    print(f"Wrote template plan to {path}")


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return clean.strip("-") or "unnamed"


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


def resolve_existing_config(config: str) -> Path:
    path = Path(config).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {config}")
    return path


def adapter_is_remote(adapter: str) -> bool:
    return (
        "/" in adapter
        and not adapter.startswith(".")
        and not adapter.startswith("/")
        and not adapter.startswith("outputs/")
        and not adapter.startswith("data/")
        and not adapter.startswith("checkpoints/")
    )


def validate_adapter(adapter: str) -> None:
    if adapter_is_remote(adapter):
        return
    path = Path(adapter).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Missing adapter/checkpoint path: {adapter}. "
            "If this is a Hugging Face repo ID, use owner/repo form."
        )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prediction_stats(path: Path, min_audio_tokens: int) -> dict[str, Any]:
    rows = iter_jsonl(path)
    if not rows:
        return {
            "prediction_count": 0,
            "empty_prediction_rate": 1.0,
            "valid_audio_token_rate": 0.0,
            "mean_audio_tokens": 0.0,
            "median_audio_tokens": 0.0,
        }
    audio_lengths = []
    empty_predictions = 0
    valid_audio = 0
    for row in rows:
        prediction = str(row.get("prediction", ""))
        if not prediction.strip():
            empty_predictions += 1
        audio_tokens = row.get("audio_tokens", [])
        if isinstance(audio_tokens, list):
            audio_len = len(audio_tokens)
        elif isinstance(audio_tokens, int):
            audio_len = audio_tokens
        else:
            audio_len = 0
        audio_lengths.append(audio_len)
        if audio_len >= min_audio_tokens:
            valid_audio += 1
    ordered = sorted(audio_lengths)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = float(ordered[mid])
    else:
        median = float((ordered[mid - 1] + ordered[mid]) / 2.0)
    return {
        "prediction_count": len(rows),
        "empty_prediction_rate": empty_predictions / len(rows),
        "valid_audio_token_rate": valid_audio / len(rows),
        "mean_audio_tokens": sum(audio_lengths) / len(audio_lengths),
        "median_audio_tokens": median,
    }


def checkpoint_items(plan: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for system in plan.get("systems", []):
        if not isinstance(system, dict):
            raise ValueError("Each system must be a mapping/object.")
        system_name = system.get("name")
        if not system_name:
            raise ValueError("Every system requires a name.")
        if "config" not in system:
            raise ValueError(f"System {system_name!r} requires config.")
        checkpoints = system.get("checkpoints") or []
        if not checkpoints:
            raise ValueError(f"System {system_name!r} has no checkpoints.")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Checkpoint under {system_name!r} must be an object.")
            if "adapter" not in checkpoint:
                raise ValueError(f"Checkpoint under {system_name!r} requires adapter.")
            items.append(
                {
                    "kind": "adapter",
                    "system": str(system_name),
                    "rank": system.get("rank"),
                    "config": str(system["config"]),
                    "checkpoint": str(checkpoint.get("name") or checkpoint["adapter"]),
                    "adapter": str(checkpoint["adapter"]),
                    "base_model": checkpoint.get("base_model") or system.get("base_model"),
                    "trainable_params": system.get("trainable_params"),
                    "trainable_percent": system.get("trainable_percent"),
                    "notes": system.get("notes", ""),
                }
            )
    return items


def cascade_item(plan: dict[str, Any]) -> dict[str, Any] | None:
    cascade = plan.get("cascade")
    if not isinstance(cascade, dict) or not cascade.get("enabled"):
        return None
    return {
        "kind": "cascade",
        "system": str(cascade.get("name", "cascade")),
        "rank": None,
        "config": str(cascade.get("config", "config.yaml")),
        "checkpoint": "cascade",
        "adapter": None,
        "base_model": None,
        "trainable_params": None,
        "trainable_percent": None,
        "notes": cascade.get("notes", ""),
        "asr_model": cascade.get("asr_model"),
        "mt_model": cascade.get("mt_model"),
        "src_lang": cascade.get("src_lang"),
        "tgt_lang": cascade.get("tgt_lang"),
    }


def build_eval_cmd(
    item: dict[str, Any],
    split: str,
    limit: int | None,
    comet_model: str | None,
    predictions_path: Path,
    metrics_path: Path,
) -> list[str]:
    if item["kind"] == "cascade":
        cmd = [
            sys.executable,
            "eval_cascade.py",
            "--config",
            item["config"],
            "--split",
            split,
            "--output-jsonl",
            str(predictions_path),
            "--metrics-path",
            str(metrics_path),
        ]
        for option, value in (
            ("--asr-model", item.get("asr_model")),
            ("--mt-model", item.get("mt_model")),
            ("--src-lang", item.get("src_lang")),
            ("--tgt-lang", item.get("tgt_lang")),
        ):
            if value:
                cmd.extend([option, str(value)])
    else:
        cmd = [
            sys.executable,
            "eval.py",
            "--config",
            item["config"],
            "--split",
            split,
            "--adapter",
            item["adapter"],
            "--output-jsonl",
            str(predictions_path),
            "--metrics-path",
            str(metrics_path),
        ]
        if item.get("base_model"):
            cmd.extend(["--base-model", str(item["base_model"])])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if comet_model:
        cmd.extend(["--comet-model", comet_model])
    return cmd


def metric_order(preferred_metric: str | None) -> list[str]:
    candidates = [preferred_metric] if preferred_metric else []
    candidates.extend(metric for metric in BEST_FALLBACKS if metric not in candidates)
    return [metric for metric in candidates if metric]


def mark_best(rows: list[dict[str, Any]], preferred_metric: str | None) -> None:
    for row in rows:
        row["is_best_for_rank"] = False
        row["best_metric_used"] = None
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "adapter":
            continue
        rank = row.get("rank")
        group_key = f"rank-{rank}" if rank is not None else str(row.get("system"))
        groups.setdefault(group_key, []).append(row)

    for group_rows in groups.values():
        metric = None
        candidates = []
        for candidate_metric in metric_order(preferred_metric):
            metric_candidates = []
            for row in group_rows:
                value = row.get(candidate_metric)
                if isinstance(value, (int, float)):
                    metric_candidates.append((row, candidate_metric, float(value)))
            if metric_candidates:
                metric = candidate_metric
                candidates = metric_candidates
                break
        if not candidates:
            continue
        if metric in LOWER_IS_BETTER:
            best = min(candidates, key=lambda item: item[2])
        else:
            best = max(candidates, key=lambda item: item[2])
        best[0]["is_best_for_rank"] = True
        for row, used_metric, _ in candidates:
            row["best_metric_used"] = used_metric


def write_summary(rows: list[dict[str, Any]], output_dir: Path, metadata: dict[str, Any]) -> None:
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    payload = {"metadata": metadata, "rows": rows}
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    if fieldnames:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


def evaluate_item(
    item: dict[str, Any],
    output_dir: Path,
    split: str,
    limit: int | None,
    comet_model: str | None,
    min_audio_tokens: int,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    config_path = resolve_existing_config(item["config"])
    item["config"] = str(config_path)
    if item["kind"] == "adapter":
        validate_adapter(item["adapter"])

    stem = f"{slug(item['system'])}__{slug(item['checkpoint'])}"
    predictions_path = output_dir / "predictions" / f"{stem}_predictions.jsonl"
    metrics_path = output_dir / "metrics" / f"{stem}_metrics.json"
    log_path = output_dir / "logs" / f"{stem}.log"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and predictions_path.exists() and metrics_path.exists():
        print(f"Reusing existing outputs for {stem}")
    else:
        cmd = build_eval_cmd(item, split, limit, comet_model, predictions_path, metrics_path)
        run_and_log(cmd, log_path, dry_run=dry_run)

    metrics = {} if dry_run else read_json(metrics_path)
    stats = {} if dry_run else prediction_stats(predictions_path, min_audio_tokens)
    row = dict(item)
    row.update(metrics)
    row.update(stats)
    row.update(
        {
            "split": split,
            "limit": limit,
            "comet_model": comet_model,
            "predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    if args.write_template:
        write_template(Path(args.write_template).expanduser())
        return
    if not args.plan:
        raise SystemExit("--plan is required unless --write-template is used.")

    plan = load_structured(Path(args.plan).expanduser())
    output_dir = Path(args.output_dir or plan.get("output_dir", "outputs/ablation/rank_eval")).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split = args.split or str(plan.get("split", "validation"))
    limit = args.limit if args.limit is not None else plan.get("limit")
    if limit is not None:
        limit = int(limit)
    comet_model = args.comet_model if args.comet_model is not None else plan.get("comet_model")
    min_audio_tokens = (
        args.min_audio_tokens
        if args.min_audio_tokens is not None
        else int(plan.get("min_audio_tokens", 10))
    )
    best_metric = args.best_metric or plan.get("best_metric")

    items = checkpoint_items(plan)
    cascade = cascade_item(plan)
    if cascade and args.include_cascade:
        items.append(cascade)
    elif cascade:
        print("Cascade block is enabled in plan but was not run; pass --include-cascade to run it.")

    rows = [
        evaluate_item(
            item=item,
            output_dir=output_dir,
            split=split,
            limit=limit,
            comet_model=comet_model,
            min_audio_tokens=min_audio_tokens,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        for item in items
    ]
    if not args.dry_run:
        mark_best(rows, best_metric)

    metadata = {
        "plan": str(Path(args.plan).expanduser()),
        "split": split,
        "limit": limit,
        "comet_model": comet_model,
        "min_audio_tokens": min_audio_tokens,
        "best_metric": best_metric,
        "git_commit": current_git_commit(),
    }
    write_summary(rows, output_dir, metadata)


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
