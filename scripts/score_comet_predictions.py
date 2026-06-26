from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from jiwer import wer


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_for_row(row: dict[str, Any], prepared_by_id: dict[str, dict[str, Any]]) -> str:
    for key in ("source", "source_text", "src", "text_lug"):
        value = row.get(key)
        if value:
            return str(value)
    prepared = prepared_by_id.get(str(row.get("id")), {})
    for key in ("source_text", "text_lug", "text_eng"):
        value = prepared.get(key)
        if value:
            return str(value)
    return ""


def maybe_comet(
    predictions: list[str],
    references: list[str],
    sources: list[str],
    model_name: str,
) -> float:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError("Install unbabel-comet to compute COMET.") from exc
    checkpoint = download_model(model_name)
    model = load_from_checkpoint(checkpoint)
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
    comet_output = model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    return float(comet_output.system_score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute COMET and text metrics from an existing prediction JSONL."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    parser.add_argument(
        "--prepared-jsonl",
        default=None,
        help="Optional prepared split JSONL used to fill missing source text by id.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    prepared_by_id = {}
    if args.prepared_jsonl:
        prepared_by_id = {
            str(row["id"]): row
            for row in read_jsonl(args.prepared_jsonl)
            if row.get("id") is not None
        }

    predictions = [str(row.get("prediction", "") or "") for row in rows]
    references = [str(row.get("reference", "") or "") for row in rows]
    sources = [source_for_row(row, prepared_by_id) for row in rows]

    metrics = {
        "predictions": str(args.predictions),
        "count": len(rows),
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score if predictions else 0.0,
        "chrf": sacrebleu.corpus_chrf(predictions, [references]).score if predictions else 0.0,
        "wer_on_text_channel": wer(references, predictions) if predictions else 1.0,
        "comet_model": args.comet_model,
        "comet": maybe_comet(predictions, references, sources, args.comet_model),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
