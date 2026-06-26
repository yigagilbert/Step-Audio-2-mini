from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: stepfun-ai/Step-Audio-2-mini
library_name: peft
tags:
- audio
- speech-translation
- speech-to-speech
- luganda
- english
- stepaudio2
model-index:
- name: Step-Audio 2 Mini Luganda-to-English S2ST LoRA
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

# Step-Audio 2 Mini Luganda-to-English S2ST LoRA

This repository contains a LoRA adapter for `stepfun-ai/Step-Audio-2-mini` trained for
Luganda speech input to English speech output. The adapter requires the base model and
Step-Audio 2 `token2wav` assets for waveform synthesis.

Adapter repository: `{repo_id}`

## Intended use

Research and development for Luganda-to-English speech translation. Validate with native
speakers before production use.

## Training data

Configured for `yigagilbert/luganda-english-cleaned-v1-split` with columns:
`audio_lug`, `audio_eng`, `text_lug`, `text_eng`, `id`, `src_dur_s`, `tgt_dur_s`,
`dur_ratio`, `src_speech_ratio`, `tgt_speech_ratio`.

## Evaluation

The table below summarizes the held-out validation evaluation used during development.
All text metrics were computed on 200 validation examples. Speech metrics were computed
on the aligned 197-example subset for which generated audio existed.

The merged full-model repository contains the same adapted weights as this adapter after
LoRA merge, so it is expected to match these results aside from normal deterministic or
runtime differences. Re-run evaluation directly on the merged repository before a strict
release if exact reproducibility is required.

### Text and Semantic Metrics

| System | Loading Form | Count | BLEU higher | chrF higher | WER lower | COMET higher | BLASER ref higher | BLASER QE higher |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base Step-Audio-2-mini | Base only | 200 | 0.012 | 5.152 | 10.702 | 0.386 | 1.713 | 2.164 |
| Fine-tuned Step-Audio | Base + this LoRA adapter | 200 | 32.530 | 54.535 | 0.574 | 0.717 | 3.762 | 3.723 |
| Full merged fine-tuned model | Same adapted weights, merged | 200 | 32.530* | 54.535* | 0.574* | 0.717* | 3.762* | 3.723* |
| Cascade baseline | ASR + MT + TTS | 200 | 36.778 | 57.971 | 0.521 | 0.737 | 3.839 | 3.776 |

`*` The merged full-model row reflects the adapter evaluation because the merged
repository is produced by folding this LoRA adapter into the same base model.

### Speech Metrics

| System | Count | chrF higher | SpeechBERT P higher | SpeechBERT R higher | SpeechBERT F1 higher | MCD lower |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned Step-Audio | 197 | 54.582 | 0.644 | 0.648 | 0.645 | 629.718 |
| Full merged fine-tuned model | 197 | 54.582* | 0.644* | 0.648* | 0.645* | 629.718* |
| Cascade baseline | 197 | 57.893 | 0.603 | 0.622 | 0.612 | 613.212 |

The unfine-tuned base model emitted no valid speech-token sequences in this 200-sample
run, so SpeechBERTScore and MCD were not computed for it.

### Interpretation

The LoRA adapter changes the base model from an unusable zero-shot system into a
functioning Luganda-to-English speech translation model. The cascade remains stronger
on text and semantic metrics, while the fine-tuned Step-Audio system is operationally
simpler and scored higher on the WavLM-based SpeechBERTScore F1 proxy.

## License

Adapter code and metadata are Apache-2.0. Check the dataset license separately before
redistribution.
"""

BIDIRECTIONAL_MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: stepfun-ai/Step-Audio-2-mini
library_name: peft
tags:
- audio
- speech-translation
- speech-to-speech
- luganda
- english
- bidirectional
- stepaudio2
model-index:
- name: Step-Audio 2 Mini Luganda-English Bidirectional S2ST LoRA
  results:
  - task:
      type: speech-translation
      name: English-to-Luganda speech translation
    dataset:
      type: yigagilbert/luganda-english-cleaned-v1-split
      name: Luganda-English Cleaned v1 Split
      split: validation
    metrics:
    - type: bleu
      name: BLEU
      value: 11.155
    - type: chrf
      name: chrF
      value: 45.331
    - type: wer
      name: WER on generated Luganda text
      value: 0.811
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
      value: 19.032
    - type: chrf
      name: chrF
      value: 42.447
    - type: wer
      name: WER on generated English text
      value: 0.737
---

# Step-Audio 2 Mini Luganda-English Bidirectional S2ST LoRA

This repository contains a LoRA adapter for `stepfun-ai/Step-Audio-2-mini` trained
with direction prompts for both Luganda-to-English and English-to-Luganda
speech-to-speech translation.

Adapter repository: `{repo_id}`

## Intended Use

Research and development for bidirectional Luganda-English speech translation. Validate
with native speakers before production use.

## Training Data

The adapter was trained from `yigagilbert/luganda-english-cleaned-v1-split` by expanding
each aligned pair into two supervised examples:

- Luganda speech input -> English text and English speech tokens
- English speech input -> Luganda text and Luganda speech tokens

Each example uses a direction-specific system prompt so that one adapter can learn both
translation directions.

## Evaluation

The checkpoint uploaded here was evaluated on 200 held-out validation examples per
direction.

| Direction | Count | BLEU higher | chrF higher | WER lower |
|---|---:|---:|---:|---:|
| English -> Luganda | 200 | 11.155 | 45.331 | 0.811 |
| Luganda -> English | 200 | 19.032 | 42.447 | 0.737 |

Compared with the earlier Luganda-to-English-only LoRA adapter, this bidirectional
adapter adds English-to-Luganda capability, but Luganda-to-English quality is lower than
the specialized one-direction adapter. This is expected: one adapter is sharing capacity
across both translation directions.

## Loading

Load this as a PEFT adapter on top of `stepfun-ai/Step-Audio-2-mini`.

## Limitations

- Automatic metrics for Luganda can understate useful translations because of spelling
  and morphology variation.
- Speech quality should be checked by listening to synthesized samples.
- This adapter is a research artifact and should not be used for high-stakes translation
  without human validation.

## License

Adapter code and metadata are Apache-2.0. Check the dataset and base model licenses
separately before redistribution.
"""


def write_model_card(folder: Path, repo_id: str, overwrite: bool, card_type: str) -> Path:
    readme = folder / "README.md"
    if not readme.exists() or overwrite:
        template = (
            BIDIRECTIONAL_MODEL_CARD_TEMPLATE
            if card_type == "bidirectional"
            else MODEL_CARD_TEMPLATE
        )
        readme.write_text(template.format(repo_id=repo_id), encoding="utf-8")
    return readme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="outputs/stepaudio2-luganda-lora/final")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--card-type",
        choices=("lug_to_eng", "bidirectional"),
        default="lug_to_eng",
        help="Model card template to write before upload.",
    )
    parser.add_argument(
        "--overwrite-readme",
        action="store_true",
        help="Overwrite README.md/model card in --folder before uploading.",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Upload only README.md/model card instead of the full adapter folder.",
    )
    parser.add_argument("--commit-message", default="Upload Step-Audio Luganda-English adapter")
    args = parser.parse_args()

    folder = Path(args.folder)
    if args.readme_only:
        folder.mkdir(parents=True, exist_ok=True)
    elif not folder.exists():
        raise FileNotFoundError(folder)
    readme = write_model_card(
        folder,
        repo_id=args.repo_id,
        overwrite=args.overwrite_readme,
        card_type=args.card_type,
    )

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    if args.readme_only:
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update model card with evaluation results",
        )
        print(f"Updated model card at https://huggingface.co/{args.repo_id}")
    else:
        api.upload_folder(
            folder_path=str(folder),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
        print(f"Pushed {folder} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
