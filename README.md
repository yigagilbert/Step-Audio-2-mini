# Step-Audio 2 Mini Luganda-to-English S2ST LoRA Pipeline

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

## Install

```bash
conda create -n stepaudio2-luganda python=3.10 -y
conda activate stepaudio2-luganda
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-2-mini
git clone https://github.com/stepfun-ai/Step-Audio2.git Step-Audio2

# Required if the dataset is private or gated.
export HF_TOKEN=hf_...
```

If `s3tokenizer` or `onnxruntime` wheels need a CUDA-specific install on your H100 host,
install those wheels first, then rerun `pip install -r requirements.txt`.

## Data preparation

The dataset is expected to contain exactly these columns:

```text
audio_lug, audio_eng, text_lug, text_eng, id, src_dur_s, tgt_dur_s,
dur_ratio, src_speech_ratio, tgt_speech_ratio
```

Run:

```bash
python data_prep.py --config config.yaml --device cuda
```

Defaults keep pairs with `0.5 <= dur_ratio <= 2.0`, source/target speech ratios >=0.7,
source duration <=25s, and target duration <=30s. The 25s source cap matches the public
inference code's audio chunking constraint around the encoder context; segment longer
utterances before SFT if you want to keep them.

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
torchrun --nproc_per_node=1 train.py --config config.yaml
```

Expected first-run behavior on an H100 NVL: preprocessing is dominated by target speech
tokenization; training 50 hours for one epoch is likely in the 4-8 hour range depending
on average utterance length, disk speed, and evaluation frequency.

## Evaluate

```bash
python eval.py --config config.yaml --split test --adapter outputs/stepaudio2-luganda-lora/final

# Optional COMET if installed:
python eval.py --config config.yaml --split test --comet-model Unbabel/wmt22-comet-da
```

The evaluator reports BLEU and WER over the generated English text channel. For final
S2ST acceptance, also run native-speaker review and ASR-based WER on generated wavs.

## Inference

```bash
python inference_example.py \
  --config config.yaml \
  --audio path/to/luganda.wav \
  --adapter outputs/stepaudio2-luganda-lora/final \
  --stepaudio2-repo Step-Audio2 \
  --output outputs/luganda_to_english.wav
```

For low-latency service, merge the adapter and serve with the official Step-Audio 2 vLLM
Docker image after validating that your vLLM build supports the merged custom model:

```bash
python scripts/merge_lora.py --config config.yaml --output outputs/merged-stepaudio2-luganda
```

Then adapt the official command from the Step-Audio2 README with
`--audio-parser step_audio_2_tts_ta4`, `--tokenizer-mode step_audio_2`, and
`--trust-remote-code`.

Streaming client:

```bash
python streaming_vllm_example.py \
  --config config.yaml \
  --audio path/to/luganda.wav \
  --api-url http://localhost:8000/v1/chat/completions \
  --model-name step-audio-2-mini \
  --stepaudio2-repo Step-Audio2 \
  --output outputs/streaming_luganda_to_english.wav
```

## Push to Hugging Face

```bash
python scripts/push_to_hub.py \
  --folder outputs/stepaudio2-luganda-lora/final \
  --repo-id your-org/stepaudio2-mini-luganda-english-s2st-lora
```

The pushed adapter card is Apache-2.0. Check and document the dataset license before
publishing trained weights.
# Step-Audio-2-mini
