from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.modeling import load_model, load_tokenizer  # noqa: E402


FULL_MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: {base_model}
library_name: transformers
tags:
- audio
- speech-translation
- speech-to-speech
- luganda
- english
- stepaudio2
- merged-lora
model-index:
- name: Step-Audio 2 Mini Luganda-to-English S2ST
  results:
  - task:
      type: speech-translation
      name: Luganda-to-English speech translation
    dataset:
      type: yigagilbert/luganda-english-cleaned-v1-split
      name: Luganda-English Cleaned v1 Split
      split: validation
    metrics:
    - type: bleu
      name: BLEU
      value: 32.530
    - type: chrf
      name: chrF
      value: 54.535
    - type: wer
      name: WER on generated English text
      value: 0.574
    - type: comet
      name: COMET
      value: 0.717
    - type: blaser_2_0_ref
      name: BLASER 2.0 ref
      value: 3.762
    - type: blaser_2_0_qe
      name: BLASER 2.0 QE
      value: 3.723
---

# Step-Audio 2 Mini Luganda-to-English S2ST

This repository contains a full merged model for Luganda speech input to English speech
translation output. It was created by merging the LoRA adapter `{adapter}` into
`{base_model}`.

The separate adapter-only repository should remain available for users who prefer PEFT
loading or want the smaller adapter artifact. This repository is intended for simpler
deployment and inference where loading a single model repo is preferable.

## Intended Use

Research and development for Luganda-to-English speech translation. Validate outputs
with native speakers before production use.

## Source Model and Adapter

- Base model: `{base_model}`
- LoRA adapter: `{adapter}`
- Merge script: `scripts/push_full_model_to_hub.py`

## Evaluation Summary

The table below summarizes the held-out validation evaluation used during development.
All text metrics were computed on 200 validation examples. Speech metrics were computed
on the aligned 197-example subset for which generated audio existed.

This full model was created by merging the LoRA adapter into the base model. The metrics
below were generated with the adapter-loaded fine-tuned model before merge; the merged
model contains the same adapted weights and is expected to match these results aside
from normal deterministic or runtime differences. Re-run evaluation directly on this
repository before a strict release if exact reproducibility is required.

### Text and Semantic Metrics

| System | Loading Form | Count | BLEU higher | chrF higher | WER lower | COMET higher | BLASER ref higher | BLASER QE higher |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base Step-Audio-2-mini | Base only | 200 | 0.012 | 5.152 | 10.702 | 0.386 | 1.713 | 2.164 |
| Fine-tuned Step-Audio | Base + LoRA adapter | 200 | 32.530 | 54.535 | 0.574 | 0.717 | 3.762 | 3.723 |
| This full merged model | Same adapted weights, merged | 200 | 32.530* | 54.535* | 0.574* | 0.717* | 3.762* | 3.723* |
| Cascade baseline | ASR + MT + TTS | 200 | 36.778 | 57.971 | 0.521 | 0.737 | 3.839 | 3.776 |

`*` The full merged row reflects the adapter evaluation because this repository is
produced by folding the evaluated LoRA adapter into the same base model.

### Speech Metrics

| System | Count | chrF higher | SpeechBERT P higher | SpeechBERT R higher | SpeechBERT F1 higher | MCD lower |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned Step-Audio | 197 | 54.582 | 0.644 | 0.648 | 0.645 | 629.718 |
| This full merged model | 197 | 54.582* | 0.644* | 0.648* | 0.645* | 629.718* |
| Cascade baseline | 197 | 57.893 | 0.603 | 0.622 | 0.612 | 613.212 |

The unfine-tuned base model emitted no valid speech-token sequences in this 200-sample
run, so SpeechBERTScore and MCD were not computed for it.

### Interpretation

The fine-tuned Step-Audio model substantially improves over the base model and becomes
a viable end-to-end Luganda-to-English speech translation system. The cascade remains
stronger on text and semantic metrics, while the fine-tuned Step-Audio system is simpler
to deploy and scored higher on the WavLM-based SpeechBERTScore F1 proxy.

## Notes

If this repository includes `token2wav/`, those assets are provided to support waveform
synthesis from generated audio tokens. Some inference clients may still use the official
Step-Audio2 runtime code for token-to-waveform conversion.

## License

The training code and adapter metadata are Apache-2.0. Because this merged repository
contains base-model weights, users must also comply with the base model license and any
dataset licensing constraints.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a Step-Audio LoRA adapter into the base model and push a full HF repo."
    )
    parser.add_argument("--config", default="configs/h100_nvl_fast_deepspeed.yaml")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path or Hub repo ID. Defaults to config model.local_path/name_or_path.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Adapter folder or Hub repo ID. Defaults to project.output_dir/final.",
    )
    parser.add_argument("--output", default="outputs/merged-stepaudio2-luganda")
    parser.add_argument("--repo-id", required=True, help="Destination HF model repo ID.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Push the existing --output folder without rebuilding the merged model.",
    )
    parser.add_argument(
        "--include-token2wav-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy token2wav/assets from the base model into the merged folder.",
    )
    parser.add_argument(
        "--overwrite-readme",
        action="store_true",
        help="Overwrite an existing README.md in the merged model folder.",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Upload only README.md/model card instead of the full merged model folder.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload merged Step-Audio Luganda-English model",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def existing_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def resolve_base_model_ref(cfg: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    local_path = cfg["model"].get("local_path")
    if local_path and Path(local_path).exists():
        return str(local_path)
    return str(cfg["model"]["name_or_path"])


def resolve_adapter_ref(cfg: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return str(Path(cfg["project"]["output_dir"]) / "final")


def write_model_card(output: Path, base_model: str, adapter: str, overwrite: bool) -> None:
    readme = output / "README.md"
    if readme.exists() and not overwrite:
        return
    readme.write_text(
        FULL_MODEL_CARD_TEMPLATE.format(base_model=base_model, adapter=adapter),
        encoding="utf-8",
    )


def copy_if_exists(src_root: Path, output: Path, relative_path: str) -> None:
    src = src_root / relative_path
    if not src.exists():
        return
    dst = output / relative_path
    if dst.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_token2wav_assets(base_model: str, output: Path) -> None:
    base_path = existing_path(base_model)
    if base_path is None:
        base_path = Path(
            snapshot_download(
                repo_id=base_model,
                allow_patterns=["token2wav/*", "assets/*", "flow.yaml"],
            )
        )

    copy_if_exists(base_path, output, "token2wav")
    copy_if_exists(base_path, output, "assets")
    copy_if_exists(base_path, output, "flow.yaml")


def merge_model(cfg: dict[str, Any], base_model: str, adapter: str, output: Path) -> None:
    model_cfg = dict(cfg["model"])
    model_cfg["gradient_checkpointing"] = False
    model = load_model(base_model, model_cfg)
    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)

    tokenizer = load_tokenizer(
        base_model,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    tokenizer.save_pretrained(output)


def push_folder(output: Path, repo_id: str, private: bool, commit_message: str) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(output),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )


def push_readme(readme: Path, repo_id: str, private: bool) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card with evaluation results",
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base_model = resolve_base_model_ref(cfg, args.base_model)
    adapter = resolve_adapter_ref(cfg, args.adapter)
    output = Path(args.output)

    if args.readme_only:
        output.mkdir(parents=True, exist_ok=True)
    elif args.skip_merge:
        if not output.exists():
            raise FileNotFoundError(output)
    else:
        print(f"Merging adapter '{adapter}' into base model '{base_model}'")
        merge_model(cfg, base_model, adapter, output)

    if args.include_token2wav_assets and not args.readme_only:
        print("Copying token2wav assets into the full model folder")
        copy_token2wav_assets(base_model, output)

    write_model_card(output, base_model=base_model, adapter=adapter, overwrite=args.overwrite_readme)
    readme = output / "README.md"

    if args.readme_only:
        print(f"Updating model card at https://huggingface.co/{args.repo_id}")
        push_readme(readme, args.repo_id, private=args.private)
        print(f"Updated model card at https://huggingface.co/{args.repo_id}")
        return

    print(f"Pushing {output} to https://huggingface.co/{args.repo_id}")
    push_folder(output, args.repo_id, private=args.private, commit_message=args.commit_message)
    print(f"Pushed full model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
