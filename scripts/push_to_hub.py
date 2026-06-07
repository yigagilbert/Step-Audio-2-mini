from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


MODEL_CARD = """---
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
---

# Step-Audio 2 Mini Luganda-to-English S2ST LoRA

This repository contains a LoRA adapter for `stepfun-ai/Step-Audio-2-mini` trained for
Luganda speech input to English speech output. The adapter requires the base model and
Step-Audio 2 `token2wav` assets for waveform synthesis.

## Intended use

Research and development for Luganda-to-English speech translation. Validate with native
speakers before production use.

## Training data

Configured for `yigagilbert/luganda-english-cleaned-v1-split` with columns:
`audio_lug`, `audio_eng`, `text_lug`, `text_eng`, `id`, `src_dur_s`, `tgt_dur_s`,
`dur_ratio`, `src_speech_ratio`, `tgt_speech_ratio`.

## License

Adapter code and metadata are Apache-2.0. Check the dataset license separately before
redistribution.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="outputs/stepaudio2-luganda-lora/final")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    readme = folder / "README.md"
    if not readme.exists():
        readme.write_text(MODEL_CARD, encoding="utf-8")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.repo_id, repo_type="model")
    print(f"Pushed {folder} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
