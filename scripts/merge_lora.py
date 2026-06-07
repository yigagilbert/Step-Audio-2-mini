from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.modeling import load_model, load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", default="outputs/merged-stepaudio2-luganda")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_path = cfg["model"].get("local_path") or cfg["model"]["name_or_path"]
    adapter_path = args.adapter or str(Path(cfg["project"]["output_dir"]) / "final")
    model = load_model(base_path, cfg["model"])
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    tokenizer = load_tokenizer(base_path, trust_remote_code=cfg["model"].get("trust_remote_code", True))
    tokenizer.save_pretrained(output)
    print(f"Wrote merged model to {output}")


if __name__ == "__main__":
    main()
