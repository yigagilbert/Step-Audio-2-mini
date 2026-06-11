# Using the Luganda-to-English Step-Audio Adapter in Google Colab

This guide shows how to use a Hugging Face-hosted LoRA adapter for
`stepfun-ai/Step-Audio-2-mini` in a Google Colab notebook. It translates one Luganda
audio file into English text and synthesized English speech.

The notebook user does not need this training repository. They only need:

- A Hugging Face adapter repo ID, for example
  `your-org/stepaudio2-mini-luganda-english-s2st-lora`
- Access to the base model `stepfun-ai/Step-Audio-2-mini`
- One Luganda audio file, preferably under 25 seconds
- A GPU Colab runtime

## Hardware Notes

Step-Audio 2 Mini is an 8B model. Use a GPU runtime with enough VRAM.

Recommended:

- Colab Pro/Pro+ with A100, L4, or another high-memory GPU
- High-RAM runtime if available

T4 runtimes may run out of memory.

## 1. Select a GPU Runtime

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Then verify the GPU:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

## 2. Install Dependencies

Run this in a Colab cell:

```python
%%capture
!pip install -U \
  "transformers==4.49.0" \
  "peft>=0.12.0" \
  "accelerate>=0.34.0" \
  "huggingface_hub>=0.24.6" \
  "librosa>=0.10.2.post1" \
  "soundfile>=0.12.1" \
  "sentencepiece>=0.2.0" \
  "s3tokenizer" \
  "diffusers" \
  "hyperpyyaml" \
  "onnxruntime"

!git clone --depth 1 https://github.com/stepfun-ai/Step-Audio2.git /content/Step-Audio2
```

Restart the runtime if Colab asks you to after installation.

## 3. Log In to Hugging Face

If the adapter repo is private, add an `HF_TOKEN` secret in Colab:

```text
Colab left sidebar -> Secrets -> Add new secret -> HF_TOKEN
```

Then run:

```python
from huggingface_hub import login

try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except Exception:
    hf_token = None

if hf_token:
    login(token=hf_token)
else:
    print("No HF_TOKEN secret found. Public repos can still load without login.")
```

## 4. Configure the Model and Input Audio

Replace `ADAPTER_REPO_ID` with the Hugging Face repo that contains your LoRA adapter.

```python
BASE_MODEL_ID = "stepfun-ai/Step-Audio-2-mini"
ADAPTER_REPO_ID = "your-org/stepaudio2-mini-luganda-english-s2st-lora"

INPUT_AUDIO = "/content/luganda.wav"
OUTPUT_TEXT = "/content/english_translation.txt"
OUTPUT_AUDIO = "/content/english_translation.wav"
OUTPUT_JSON = "/content/translation_metadata.json"

SYSTEM_PROMPT = (
    "You are a professional Luganda-to-English speech translation system. "
    "Listen to the Luganda speech and answer with natural English speech."
)

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.05
DO_SAMPLE = True
```

Upload an audio file through the Colab UI or use:

```python
from google.colab import files

uploaded = files.upload()
if uploaded:
    INPUT_AUDIO = "/content/" + next(iter(uploaded.keys()))
    print(INPUT_AUDIO)
```

## 5. Load the Model

This cell downloads the base model and adapter, then loads them with PEFT.

```python
import json
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def patch_torchaudio_bytesio_save():
    import io

    original_save = torchaudio.save
    if getattr(original_save, "_stepaudio_bytesio_patch", False):
        return

    def save(uri, src, sample_rate: int, *args, **kwargs):
        if isinstance(uri, io.BytesIO):
            suffix = f".{str(kwargs.get('format') or 'wav').lstrip('.')}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                original_save(str(tmp_path), src, sample_rate, *args, **kwargs)
                uri.write(tmp_path.read_bytes())
                uri.seek(0)
                return None
            finally:
                tmp_path.unlink(missing_ok=True)
        return original_save(uri, src, sample_rate, *args, **kwargs)

    save._stepaudio_bytesio_patch = True
    torchaudio.save = save


def torch_dtype_for_runtime():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch_dtype_for_runtime()

base_dir = snapshot_download(BASE_MODEL_ID)
print("Base model snapshot:", base_dir)

tokenizer = AutoTokenizer.from_pretrained(
    base_dir,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_dir,
    trust_remote_code=True,
    torch_dtype=dtype,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO_ID)
model.to(device).eval()

sys.path.insert(0, "/content/Step-Audio2")
patch_torchaudio_bytesio_save()
from token2wav import Token2wav

token2wav = Token2wav(str(Path(base_dir) / "token2wav"))
```

## 6. Define Audio and Prompt Helpers

These helpers reproduce the Step-Audio 2 prompt format used during fine-tuning.

```python
AUDIO_START = "<audio_start>"
AUDIO_END = "<audio_end>"
AUDIO_PATCH = "<audio_patch>"
TTS_START = "<tts_start>"
TTS_END = "<tts_end>"
BOT = "<|BOT|>"
EOT = "<|EOT|>"
DEFAULT_AUDIO_TOKEN_OFFSET = 151696
DEFAULT_TTS_VALID_MAX = 6560


def load_audio(path, target_rate=16000):
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.to(torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.squeeze()
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_rate)(waveform)
    return waveform.contiguous()


def mel_filters(n_mels=128):
    return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels))


def log_mel_spectrogram(audio, n_mels=128, padding=479):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(np.asarray(audio))
    audio = audio.to(dtype=torch.float32)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400, device=audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = mel_filters(n_mels).to(audio.device)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def compute_token_num(max_feature_len):
    max_feature_len = max_feature_len - 2
    encoder_output_dim = (max_feature_len + 1) // 2 // 2
    padding = 1
    kernel_size = 3
    stride = 2
    return (encoder_output_dim + 2 * padding - kernel_size) // stride + 1


def token_id(token, fallback=None):
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or value == tokenizer.unk_token_id:
        if fallback is not None:
            return fallback
        raise ValueError(f"Tokenizer does not define {token!r}")
    return int(value)


audio_start_id = token_id(AUDIO_START)
audio_end_id = token_id(AUDIO_END)
audio_patch_id = token_id(AUDIO_PATCH)
tts_start_id = token_id(TTS_START)
tts_end_id = token_id(TTS_END)
eot_id = token_id(EOT)
audio_token_offset = token_id("<audio_0>", DEFAULT_AUDIO_TOKEN_OFFSET)


def encode_text(text):
    return tokenizer.encode(text, add_special_tokens=False)


def audio_placeholder_ids(mel_frames):
    patch_count = compute_token_num(mel_frames)
    return [audio_start_id] + [audio_patch_id] * patch_count + [audio_end_id]


def build_prompt(mel_frames):
    ids = []
    ids += encode_text(f"{BOT}system\n{SYSTEM_PROMPT}{EOT}")
    ids += encode_text(f"{BOT}human\n")
    ids += audio_placeholder_ids(mel_frames)
    ids += [eot_id]
    ids += encode_text(f"{BOT}assistant\n")
    return ids


def clean_prediction_text(text):
    text = text.strip()
    if not text:
        return text
    return text.splitlines()[0].strip()


def extract_audio_and_text(token_ids):
    text_ids = []
    audio_tokens = []
    seen_audio = False

    for token in token_ids:
        if token == eot_id:
            break
        if token == tts_start_id:
            seen_audio = True
            continue
        if token == tts_end_id:
            break
        if token >= audio_token_offset:
            audio_token = token - audio_token_offset
            if 0 <= audio_token <= DEFAULT_TTS_VALID_MAX:
                audio_tokens.append(audio_token)
            seen_audio = True
        elif not seen_audio and token < audio_start_id:
            text_ids.append(token)

    text = clean_prediction_text(tokenizer.decode(text_ids, skip_special_tokens=True))
    return text, audio_tokens
```

## 7. Translate One Audio File

```python
audio = load_audio(INPUT_AUDIO, target_rate=16000)
mel = log_mel_spectrogram(audio, n_mels=128, padding=479)

prompt_ids = build_prompt(int(mel.shape[1]))
input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
attention_mask = torch.ones_like(input_ids)
wavs = mel.unsqueeze(0).to(device=device, dtype=torch.float32)
wav_lens = torch.tensor([max(1, int(mel.shape[1]) - 2)], dtype=torch.int32, device=device)

with torch.no_grad():
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        wavs=wavs,
        wav_lens=wav_lens,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eot_id,
    )

new_ids = generated[0, len(prompt_ids):].detach().cpu().tolist()
english_text, audio_tokens = extract_audio_and_text(new_ids)

print("English text:")
print(english_text)
print("Audio tokens:", len(audio_tokens))

if not audio_tokens:
    raise RuntimeError("The model returned no audio tokens. Try increasing MAX_NEW_TOKENS.")
```

## 8. Save English Text and English Audio

```python
prompt_wav = "/content/Step-Audio2/assets/default_female.wav"
wav_bytes = token2wav(audio_tokens, prompt_wav=prompt_wav)

Path(OUTPUT_TEXT).write_text(english_text + "\n", encoding="utf-8")
Path(OUTPUT_AUDIO).write_bytes(wav_bytes)
Path(OUTPUT_JSON).write_text(
    json.dumps(
        {
            "base_model": BASE_MODEL_ID,
            "adapter": ADAPTER_REPO_ID,
            "input_audio": INPUT_AUDIO,
            "prediction": english_text,
            "audio_tokens": len(audio_tokens),
            "output_text": OUTPUT_TEXT,
            "output_audio": OUTPUT_AUDIO,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("Wrote:", OUTPUT_TEXT)
print("Wrote:", OUTPUT_AUDIO)
print("Wrote:", OUTPUT_JSON)
```

Listen inside Colab:

```python
from IPython.display import Audio, display

display(Audio(OUTPUT_AUDIO))
```

Download the outputs:

```python
from google.colab import files

files.download(OUTPUT_TEXT)
files.download(OUTPUT_AUDIO)
files.download(OUTPUT_JSON)
```

## Troubleshooting

### CUDA Out of Memory

Use an A100 or L4 runtime if available. T4 may not have enough VRAM for the base model,
adapter, and audio generation together.

### Private Adapter Repo Fails to Load

Confirm that:

- You added `HF_TOKEN` to Colab secrets
- The token has read access to the adapter repo
- You ran the login cell before loading the model

### `token2wav.py` Not Found

Run the dependency cell again and confirm:

```python
!ls /content/Step-Audio2/token2wav.py
```

### No Audio Tokens Returned

Try:

- Increasing `MAX_NEW_TOKENS`
- Using a shorter input audio file
- Confirming that the adapter was trained with `target_format: text_then_audio`

### Poor Translation Quality

Check that:

- The input is Luganda speech
- The audio is clear and shorter than roughly 25 seconds
- The adapter repo is the selected best checkpoint, not an older or weaker checkpoint

For repeatable outputs, set:

```python
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0
```
