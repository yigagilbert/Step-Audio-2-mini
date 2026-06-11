# H100 Evaluation Runbook

This runbook sets up a new H100 machine to evaluate:

1. The Hugging Face-hosted Step-Audio LoRA adapter
2. The ASR + MT + TTS cascade baseline

It prepares only the validation split and computes:

- BLEU
- chrF
- WER on the English text channel
- COMET, optional
- BLASER 2.0
- SpeechBERTScore-style WavLM F1
- MCD

## Metric Notes

- `chrF` is computed with SacreBLEU on generated English text.
- `BLASER 2.0` is computed through Meta SONAR text embeddings using
  `blaser_2_0_ref` and `blaser_2_0_qe`.
- `SpeechBERTScore` is implemented as reference-aware BERTScore over WavLM speech
  frame embeddings.
- `MCD` is computed as MFCC + DTW mel-cepstral distortion. Lower is better.
- Cascade audio metrics require a TTS stage. This runbook uses `microsoft/speecht5_tts`
  so the cascade is `ASR -> MT -> TTS`.

BLASER/SONAR setup follows the official SONAR package guidance:
`sonar-space` requires a matching `fairseq2` wheel for the installed PyTorch/CUDA build.
See https://github.com/facebookresearch/SONAR.

## 1. Clone Repos

```bash
git clone https://github.com/yigagilbert/Step-Audio-2-mini.git
cd Step-Audio-2-mini

git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-2-mini
git -C Step-Audio-2-mini lfs pull
git clone https://github.com/stepfun-ai/Step-Audio2.git Step-Audio2
```

## 2. Create the `.venv`

```bash
./scripts/setup_h100_eval_env.sh
source .venv/bin/activate
```

By default the setup script installs `torch==2.5.1` and `torchaudio==2.5.1` from the
CUDA 12.1 PyTorch wheel index. This avoids accidentally installing PyPI wheels that
expect CUDA 13 runtime libraries such as `libcudart.so.13`.

To choose another supported CUDA 12 wheel index:

```bash
PYTORCH_VERSION=2.5.1 \
TORCHAUDIO_VERSION=2.5.1 \
PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124 \
./scripts/setup_h100_eval_env.sh
```

If `fairseq2` fails to install, check the PyTorch/CUDA version:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
PY
```

Then install the matching wheel from the `fairseq2` package index documented by SONAR.

### Repair an Existing `.venv` with a CUDA 13 torchaudio Error

If importing `torchaudio` fails with:

```text
OSError: libcudart.so.13: cannot open shared object file
```

reinstall matching CUDA 12.x PyTorch wheels inside the active venv:

```bash
source .venv/bin/activate
python -m pip uninstall -y torch torchaudio torchvision
python -m pip install \
  torch==2.5.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
import torchaudio
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchaudio", torchaudio.__version__)
print("cuda available", torch.cuda.is_available())
PY
```

Then rerun the validation-only data preparation command.

## 3. Log In to Hugging Face

Required if the adapter or dataset is private/gated.

```bash
huggingface-cli login
```

Set the adapter repo once:

```bash
export ADAPTER_REPO_ID="yigagilbert/stepaudio2-mini-luganda-english-s2st-lora"
```

## 4. Prepare Only the Validation Split

This saves:

- `validation.jsonl`
- source Luganda mels for Step-Audio eval
- source Luganda wavs for cascade ASR
- reference English wavs for MCD/SpeechBERTScore

It skips target audio tokenization because we are not training.
`data_prep.py` loads only the requested split and casts Hugging Face audio with
`decode=False`, so it avoids the `torchcodec` decoder path that can fail when a
TorchCodec wheel expects CUDA 13 libraries.

```bash
python data_prep.py \
  --config configs/h100_nvl_fast_deepspeed.yaml \
  --device cuda \
  --splits validation \
  --skip-target-tokenization
```

Sanity check:

```bash
wc -l data/processed/luganda_english_cleaned_v1/validation.jsonl
ls data/processed/luganda_english_cleaned_v1/validation/wav | head
```

## 5. Evaluate the Fine-Tuned Step-Audio Adapter

Use `--limit 200` for quick apples-to-apples comparison, or omit it for full validation.

```bash
mkdir -p outputs/stepaudio2-luganda-lora/eval

python eval.py \
  --config configs/h100_nvl_fast_deepspeed.yaml \
  --base-model Step-Audio-2-mini \
  --split validation \
  --adapter "$ADAPTER_REPO_ID" \
  --limit 200 \
  --comet-model Unbabel/wmt22-comet-da \
  | tee outputs/stepaudio2-luganda-lora/eval/stepaudio_metrics_200.txt
```

If tokenizer loading fails with `expected value at line 1 column 1`, the local
`Step-Audio-2-mini/` model folder is incomplete or contains Git LFS pointer files. Fix
it with:

```bash
git -C Step-Audio-2-mini lfs pull
head -n 1 Step-Audio-2-mini/tokenizer.json
```

If the first line starts with `version https://git-lfs.github.com/spec`, LFS still has
not pulled the real file. You can bypass the local folder and load the base model from
Hugging Face instead:

```bash
python eval.py \
  --config configs/h100_nvl_fast_deepspeed.yaml \
  --base-model stepfun-ai/Step-Audio-2-mini \
  --split validation \
  --adapter "$ADAPTER_REPO_ID" \
  --limit 200 \
  --comet-model Unbabel/wmt22-comet-da
```

Then synthesize generated audio for speech metrics:

```bash
python scripts/synthesize_eval_audio.py \
  --config configs/h100_nvl_fast_deepspeed.yaml \
  --base-model Step-Audio-2-mini \
  --split validation \
  --predictions outputs/stepaudio2-luganda-lora/eval/validation_predictions.jsonl \
  --stepaudio2-repo Step-Audio2 \
  --output-dir outputs/stepaudio2-luganda-lora/eval/stepaudio_audio_samples \
  --limit 200
```

For full validation, omit `--limit`.

If synthesis fails with `google.protobuf.message.DecodeError: Error parsing message`
while loading `speech_tokenizer_v2_25hz.onnx`, the local ONNX file is probably a Git
LFS pointer or incomplete download. Check it:

```bash
ls -lh Step-Audio-2-mini/token2wav/speech_tokenizer_v2_25hz.onnx
head -n 3 Step-Audio-2-mini/token2wav/speech_tokenizer_v2_25hz.onnx
```

If it starts with `version https://git-lfs.github.com/spec`, repair it:

```bash
git lfs install
git -C Step-Audio-2-mini lfs pull
```

Or bypass the local folder and download the token2wav assets from Hugging Face:

```bash
python scripts/synthesize_eval_audio.py \
  --config configs/h100_nvl_fast_deepspeed.yaml \
  --base-model stepfun-ai/Step-Audio-2-mini \
  --split validation \
  --predictions outputs/stepaudio2-luganda-lora/eval/validation_predictions.jsonl \
  --stepaudio2-repo Step-Audio2 \
  --output-dir outputs/stepaudio2-luganda-lora/eval/stepaudio_audio_samples \
  --limit 200
```

## 6. Evaluate the Cascade Text Pipeline

```bash
python eval_cascade.py \
  --config configs/h100_nvl_fast_deepspeed.yaml \
  --split validation \
  --limit 200 \
  --comet-model Unbabel/wmt22-comet-da \
  | tee outputs/stepaudio2-luganda-lora/eval/cascade_metrics_200.txt
```

Then synthesize cascade English text into English speech for speech metrics:

```bash
python scripts/synthesize_cascade_tts.py \
  --config configs/h100_nvl_fast_deepspeed.yaml \
  --split validation \
  --predictions outputs/stepaudio2-luganda-lora/eval/cascade_validation_predictions.jsonl \
  --output-dir outputs/stepaudio2-luganda-lora/eval/cascade_audio_samples \
  --limit 200
```

For full validation, omit `--limit`.

## 7. Run Advanced Metrics

```bash
python eval_advanced_metrics.py \
  --system stepaudio=outputs/stepaudio2-luganda-lora/eval/stepaudio_audio_samples/manifest.jsonl \
  --system cascade=outputs/stepaudio2-luganda-lora/eval/cascade_audio_samples/manifest.jsonl \
  --output outputs/stepaudio2-luganda-lora/eval/advanced_metrics_200.json
```

This computes all advanced metrics by default. To skip expensive metrics:

```bash
python eval_advanced_metrics.py \
  --system stepaudio=outputs/stepaudio2-luganda-lora/eval/stepaudio_audio_samples/manifest.jsonl \
  --system cascade=outputs/stepaudio2-luganda-lora/eval/cascade_audio_samples/manifest.jsonl \
  --skip-blaser \
  --skip-speechbertscore \
  --output outputs/stepaudio2-luganda-lora/eval/quick_audio_metrics_200.json
```

## 8. Expected Output Files

Step-Audio:

```text
outputs/stepaudio2-luganda-lora/eval/validation_predictions.jsonl
outputs/stepaudio2-luganda-lora/eval/validation_metrics.json
outputs/stepaudio2-luganda-lora/eval/stepaudio_audio_samples/manifest.jsonl
```

Cascade:

```text
outputs/stepaudio2-luganda-lora/eval/cascade_validation_predictions.jsonl
outputs/stepaudio2-luganda-lora/eval/cascade_validation_metrics.json
outputs/stepaudio2-luganda-lora/eval/cascade_audio_samples/manifest.jsonl
```

Comparison:

```text
outputs/stepaudio2-luganda-lora/eval/advanced_metrics_200.json
```

## Interpretation

Text metrics:

- Higher BLEU, chrF, COMET, and BLASER are better.
- Lower WER is better.

Speech metrics:

- Higher SpeechBERTScore F1 is better.
- Lower MCD is better.

MCD is sensitive to speaker, duration, and prosody. Since the cascade uses SpeechT5 and
the fine-tuned model uses Step-Audio token2wav, MCD should be treated as an audio
similarity diagnostic rather than a pure translation-quality score.
