# Step-Audio 2 Mini Luganda-to-English S2ST LoRA Pipeline

Repository: https://github.com/yigagilbert/Step-Audio-2-mini

## Research summary

Step-Audio 2 is an end-to-end multimodal audio LLM: it consumes raw speech-derived
features through an audio encoder/adaptor and generates text plus discrete audio tokens
that are converted to waveform by `token2wav`. The official repo describes Step-Audio 2
mini as Apache-2.0, provides inference scripts, a vLLM Docker command, and examples for
S2TT/S2ST, but does not provide official fine-tuning code.

Primary source notes:

- Official repo: `Step-Audio 2 mini` is Apache-2.0, uses Python >=3.10, PyTorch >=2.3,
  `transformers==4.49.0`, and recommends the StepFun vLLM backend for fast streaming
  inference: https://github.com/stepfun-ai/Step-Audio2
- Model card: `stepfun-ai/Step-Audio-2-mini` is an 8B BF16 custom-code model, tagged
  Any-to-Any / English / Chinese / `step_audio_2`, and exposes Transformers loading with
  `trust_remote_code=True`: https://huggingface.co/stepfun-ai/Step-Audio-2-mini
- Public remote config: hidden size 3584, 28 layers, 28 attention heads, 4 KV heads,
  max sequence length 16384, 128-mel audio encoder, and vocab size 158720.
- Public remote model code inserts encoded audio features at `<audio_start>` positions
  and emits logits only, so this repo uses a custom Trainer loss instead of relying on
  a built-in `labels` argument.
- Technical report: SFT is large-scale multi-task, one epoch over roughly 4B text/audio
  tokens; for S2ST the paper reports CoVoST 2/CVSS use and CVSS BLEU for Step-Audio 2
  mini of 29.08 average: https://arxiv.org/html/2507.16632v3
- GitHub issue #67 asks for fine-tuning code/audio tokenization details and is open, so
  this template is a public-code-compatible LoRA pipeline, not an official StepFun SFT
  reproduction: https://github.com/stepfun-ai/Step-Audio2/issues/67

Important correction: the 16.7 Hz linguistic plus 25 Hz acoustic dual-codebook tokenizer
with 2:3 interleaving is documented for the earlier Step-Audio tokenizer family. The
public Step-Audio 2 mini code/model files expose CosyVoice 2-style output audio tokens
`<audio_0>` onward, with `<audio_0>` at token id `151696`, `token2wav`, and special
tokens such as `<audio_start>`, `<audio_patch>`, `<tts_start>`, and `<tts_end>`. This
repo therefore does not invent an unavailable official SFT interleaver; it uses the
public token format and makes the response packing explicit in `config.yaml`.

## Pipeline diagram

```text
HF dataset
  -> schema validation and quality filters
  -> 16 kHz mono load/resample + optional energy trim
  -> Luganda source log-mel cache for Step-Audio 2 audio encoder
  -> English target speech tokenization via Step-Audio-2-mini/token2wav/speech_tokenizer_v2_25hz.onnx
  -> Step-Audio chat prompt + labels
  -> LoRA SFT on Qwen2 backbone modules with bf16 + ZeRO-3
  -> BLEU/WER/optional COMET eval on generated English text channel
  -> local or merged-model inference, then token2wav waveform synthesis
```

## Repo structure

```text
.
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- config.yaml
+-- deepspeed_zero3.json
+-- accelerate_config.yaml
+-- data_prep.py
+-- train.py
+-- eval.py
+-- inference_example.py
+-- streaming_vllm_example.py
+-- scripts/
|   +-- merge_lora.py
|   +-- push_to_hub.py
+-- src/stepaudio_luganda/
    +-- audio.py
    +-- constants.py
    +-- data.py
    +-- formatting.py
    +-- modeling.py
```

## Linux install with uv

```bash
# Ubuntu/Debian host packages.
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg libsndfile1 build-essential tmux nvtop

# Install uv if it is not already available.
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Clone this training repo.
git lfs install
git clone https://github.com/yigagilbert/Step-Audio-2-mini.git
cd Step-Audio-2-mini

# Create and activate a project-local Python environment.
uv venv --python 3.10 .venv
source .venv/bin/activate

# Recommended on H100/Linux: install CUDA PyTorch first, then the rest.
uv pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt

# Download Step-Audio 2 mini model assets and official inference helpers.
git clone https://huggingface.co/stepfun-ai/Step-Audio-2-mini
git clone https://github.com/stepfun-ai/Step-Audio2.git Step-Audio2

# Required if the dataset is private or gated.
export HF_TOKEN=hf_...
```

If your remote GPU image uses CUDA 12.4 instead of CUDA 12.1, use the matching PyTorch
wheel index from https://pytorch.org/get-started/locally/. If `s3tokenizer` or
`onnxruntime` need CUDA-specific wheels on your host, install those with `uv pip install`
inside `.venv`, then rerun `uv pip install -r requirements.txt`.

Quick environment check:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Logging and GPU monitoring

Weights & Biases is the default training logger. Log in once on the remote GPU host:

```bash
uv run wandb login
export WANDB_PROJECT=stepaudio2-luganda-s2st
export WANDB_LOG_MODEL=checkpoint
```

The active backend is controlled in `config.yaml`:

```yaml
training:
  report_to: wandb
```

For offline runs, use:

```bash
export WANDB_MODE=offline
```

Use `nvtop` in a second SSH/tmux pane while preprocessing or training:

```bash
nvtop
```

`nvtop` gives live GPU utilization, VRAM, temperature, power draw, and per-process GPU
usage. Keep it open during the first full run to confirm the H100 is saturated and that
ZeRO-3/LoRA memory stays within budget.

## Data preparation

The dataset is expected to contain exactly these columns:

```text
audio_lug, audio_eng, text_lug, text_eng, id, src_dur_s, tgt_dur_s,
dur_ratio, src_speech_ratio, tgt_speech_ratio
```

Run:

```bash
uv run python data_prep.py --config config.yaml --device cuda
```

Defaults keep pairs with `0.5 <= dur_ratio <= 2.0`, source/target speech ratios >=0.7,
source duration <=25s, and target duration <=30s. The 25s source cap matches the public
inference code's audio chunking constraint around the encoder context; segment longer
utterances before SFT if you want to keep them.

If preprocessing logs `Unsupported audio cell type:
<class 'datasets.features._torchcodec.AudioDecoder'>`, update to the latest repo code
and restart `data_prep.py`. Newer Hugging Face `datasets` versions return TorchCodec
decoder objects for `Audio` columns; this repo decodes them through
`AudioDecoder.get_all_samples()` before resampling to mono 16 kHz.

For `yigagilbert/luganda-english-cleaned-v1-split`, a normal filtered run may produce
only `train.jsonl` and `validation.jsonl`. If `data_prep.py` prints `Dataset split 'test'
not found`, use `validation` for evaluation unless you create a separate held-out test
manifest later.

## Hyperparameters

| Setting | Default | Rationale |
|---|---:|---|
| LoRA rank | 64 | Good first pass on one H100 95 GB |
| LoRA alpha | 128 | Standard 2x rank scaling |
| Target modules | q_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Backbone adaptation without full fine-tune |
| Batch size | 1 | Audio sequences are long |
| Grad accumulation | 16 | Stable effective batch on one H100 |
| LR | 1e-4 | Conservative LoRA SFT rate |
| Epochs | 2 | Start low; monitor eval BLEU/WER |
| Precision | bf16 | Native H100 path |
| Sequence length | 16384 | Public config maximum |
| DeepSpeed | ZeRO-3 | Keeps optimizer/state memory manageable |

## Train

```bash
uv run torchrun --nproc_per_node=1 train.py --config config.yaml
```

Expected first-run behavior on an H100 NVL: preprocessing is dominated by target speech
tokenization; training 50 hours for one epoch is likely in the 4-8 hour range depending
on average utterance length, disk speed, and evaluation frequency.

Recommended tmux runbook for remote GPUs:

```bash
tmux new -s stepaudio2-train
```

Inside the tmux session:

```bash
cd ~/jupyterlab-env/Step-Audio-2-mini
source .venv/bin/activate

export WANDB_PROJECT=stepaudio2-luganda-s2st
export WANDB_LOG_MODEL=checkpoint
export TOKENIZERS_PARALLELISM=false

mkdir -p logs
uv run torchrun --nproc_per_node=1 train.py --config config.yaml 2>&1 | tee logs/train-$(date +%Y%m%d-%H%M%S).log
```

Useful tmux controls:

```text
Ctrl-b d        detach and leave training running
tmux attach -t stepaudio2-train
tmux ls
tmux kill-session -t stepaudio2-train
```

Open a second tmux pane or SSH tab for GPU monitoring:

```bash
nvtop
```

Before starting the full run, sanity-check the processed manifests:

```bash
wc -l data/processed/luganda_english_cleaned_v1/train.jsonl
wc -l data/processed/luganda_english_cleaned_v1/validation.jsonl
```

Smoke-test optimizer, scheduler, DeepSpeed, and LoRA wiring for 20 update steps:

```bash
WANDB_MODE=offline uv run torchrun --nproc_per_node=1 train.py --config config.yaml --max-steps 20
```

Healthy optimizer diagnostics should show one DeepSpeed optimizer group and one
scheduler LR value:

```text
[deepspeed-config] ... gradient_accumulation_steps=16 ... train_batch_size=16
[optim-debug] using_single_deepspeed_optimizer_group=true ...
[optim-debug] stage=after_create_optimizer
[optim-debug] optimizer_type=AdamW param_groups=1
[optim-debug] stage=after_create_scheduler
[optim-debug] scheduler_type=LambdaLR base_lrs=1 last_lrs=1
```

## Evaluate

```bash
uv run python eval.py --config config.yaml --split validation --adapter outputs/stepaudio2-luganda-lora/final

# Optional COMET if installed:
uv run python eval.py --config config.yaml --split validation --comet-model Unbabel/wmt22-comet-da
```

The evaluator reports BLEU and WER over the generated English text channel. For final
S2ST acceptance, also run native-speaker review and ASR-based WER on generated wavs.

## Inference

```bash
uv run python inference_example.py \
  --config config.yaml \
  --audio path/to/luganda.wav \
  --adapter outputs/stepaudio2-luganda-lora/final \
  --stepaudio2-repo Step-Audio2 \
  --output outputs/luganda_to_english.wav
```

For low-latency service, merge the adapter and serve with the official Step-Audio 2 vLLM
Docker image after validating that your vLLM build supports the merged custom model:

```bash
uv run python scripts/merge_lora.py --config config.yaml --output outputs/merged-stepaudio2-luganda
```

Then adapt the official command from the Step-Audio2 README with
`--audio-parser step_audio_2_tts_ta4`, `--tokenizer-mode step_audio_2`, and
`--trust-remote-code`.

Streaming client:

```bash
uv run python streaming_vllm_example.py \
  --config config.yaml \
  --audio path/to/luganda.wav \
  --api-url http://localhost:8000/v1/chat/completions \
  --model-name step-audio-2-mini \
  --stepaudio2-repo Step-Audio2 \
  --output outputs/streaming_luganda_to_english.wav
```

## Push to Hugging Face

```bash
uv run python scripts/push_to_hub.py \
  --folder outputs/stepaudio2-luganda-lora/final \
  --repo-id your-org/stepaudio2-mini-luganda-english-s2st-lora
```

The pushed adapter card is Apache-2.0. Check and document the dataset license before
publishing trained weights.
