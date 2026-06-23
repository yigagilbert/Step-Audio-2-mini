from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KEYS = [
    "_step",
    "_runtime",
    "train/global_step",
    "train/epoch",
    "train/loss",
    "train/grad_norm",
    "train/learning_rate",
    "loss",
    "grad_norm",
    "learning_rate",
    "epoch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export W&B histories for rank-ablation runs to CSV, with optional simple plots. "
            "The wandb package/API is imported only after arguments are parsed."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=ENTITY/PROJECT/RUN_ID",
        help="W&B run path. NAME= is optional but recommended. May be passed multiple times.",
    )
    parser.add_argument("--output-dir", default="outputs/ablation/wandb_exports")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_KEYS,
        help="History keys to request from W&B.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optional history sampling count. By default scan_history exports all matching rows.",
    )
    parser.add_argument("--plot", action="store_true", help="Write simple PNG convergence plots.")
    return parser.parse_args()


def slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return clean.strip("-") or "run"


def parse_run(value: str) -> tuple[str, str]:
    if "=" in value:
        name, run_path = value.split("=", 1)
        name = name.strip()
        run_path = run_path.strip()
    else:
        run_path = value.strip()
        name = run_path.rstrip("/").split("/")[-1]
    if run_path.count("/") != 2:
        raise argparse.ArgumentTypeError(
            f"Expected W&B path ENTITY/PROJECT/RUN_ID, got {run_path!r}."
        )
    return name or slug(run_path), run_path


def export_run(name: str, run_path: str, keys: list[str], samples: int | None, output_dir: Path) -> Path:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb before exporting histories: uv add wandb") from exc

    try:
        api = wandb.Api()
        run = api.run(run_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not access W&B run {run_path!r}. Check credentials with `wandb login`."
        ) from exc

    raw_rows: list[dict[str, Any]]
    if samples:
        history = run.history(samples=samples, pandas=False)
        raw_rows = [dict(row) for row in history]
    else:
        raw_rows = [dict(row) for row in run.scan_history()]

    key_set = set(keys)
    key_set.update({"_step", "_runtime", "_timestamp"})
    rows = []
    for raw_row in raw_rows:
        filtered = {key: raw_row.get(key) for key in key_set if key in raw_row}
        if filtered:
            rows.append(filtered)

    for row in rows:
        row.setdefault("run_name", name)
        row.setdefault("wandb_path", run_path)
        row.setdefault("display_name", getattr(run, "display_name", ""))
        row.setdefault("state", getattr(run, "state", ""))

    output_path = output_dir / f"{slug(name)}_history.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "name": name,
        "wandb_path": run_path,
        "display_name": getattr(run, "display_name", ""),
        "state": getattr(run, "state", ""),
        "summary": dict(getattr(run, "summary", {})),
        "config": dict(getattr(run, "config", {})),
        "history_csv": str(output_path),
    }
    (output_dir / f"{slug(name)}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def plot_histories(csv_paths: list[Path], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not installed; skipping optional W&B plots.")
        return

    specs = [
        ("loss_vs_step.png", "Loss", ("train/loss", "loss")),
        ("grad_norm_vs_step.png", "Gradient Norm", ("train/grad_norm", "grad_norm")),
        (
            "learning_rate_vs_step.png",
            "Learning Rate",
            ("train/learning_rate", "learning_rate"),
        ),
    ]
    for filename, title, y_keys in specs:
        plt.figure(figsize=(8, 5))
        plotted = False
        for path in csv_paths:
            rows = read_csv(path)
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                x = first_float(row, ("train/global_step", "_step"))
                y = first_float(row, y_keys)
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            if xs and ys:
                plotted = True
                plt.plot(xs, ys, label=path.stem.replace("_history", ""))
        if plotted:
            plt.title(title)
            plt.xlabel("step")
            plt.ylabel(title.lower())
            plt.legend()
            plt.tight_layout()
            out_path = output_dir / filename
            plt.savefig(out_path, dpi=160)
            print(f"Wrote {out_path}")
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [parse_run(value) for value in args.run]
    csv_paths = [
        export_run(name, run_path, args.keys, args.samples, output_dir)
        for name, run_path in run_specs
    ]
    index = [{"name": name, "wandb_path": run_path, "history_csv": str(path)} for (name, run_path), path in zip(run_specs, csv_paths)]
    (output_dir / "wandb_export_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.plot:
        plot_histories(csv_paths, output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
