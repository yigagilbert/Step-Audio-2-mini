from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import wave
from pathlib import Path
from typing import Any

import requests
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.audio import load_audio  # noqa: E402


CHUNK_SIZE = 25


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def wav_bytes_from_tensor(audio: torch.Tensor, sample_rate: int = 16000) -> bytes:
    audio_i16 = (audio.cpu().numpy().clip(-1.0, 1.0) * 32767.0).astype("int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())
    return buf.getvalue()


class StepAudio2VLLMClient:
    audio_token_re = re.compile(r"<audio_(\d+)>")

    def __init__(self, api_url: str, model_name: str) -> None:
        self.api_url = api_url
        self.model_name = model_name

    def audio_content(self, path: str | Path) -> list[dict[str, Any]]:
        audio = load_audio(path, target_rate=16000)
        chunks = []
        for start in range(0, audio.numel(), 25 * 16000):
            chunk = audio[start : start + 25 * 16000]
            if chunk.numel() == 0:
                continue
            chunks.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(wav_bytes_from_tensor(chunk)).decode("utf-8"),
                        "format": "wav",
                    },
                }
            )
        return chunks

    def stream(self, messages: list[dict[str, Any]], **sampling):
        payload = dict(sampling)
        payload["model"] = self.model_name
        payload["messages"] = messages
        payload["stream"] = True
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
        headers = {"Content-Type": "application/json"}
        with requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=600) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                delta = json.loads(line)["choices"][0]["delta"]
                text = delta.get("tts_content", {}).get("tts_text") or delta.get("content") or ""
                audio_str = delta.get("tts_content", {}).get("tts_audio") or ""
                audio = [int(x) for x in self.audio_token_re.findall(audio_str)]
                yield text, audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model-name", default="step-audio-2-mini")
    parser.add_argument("--stepaudio2-repo", default="Step-Audio2")
    parser.add_argument("--output", default="outputs/streaming_luganda_to_english.wav")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sys.path.insert(0, str(Path(args.stepaudio2_repo).resolve()))
    from token2wav import Token2wav

    model_path = cfg["model"].get("local_path") or cfg["model"]["name_or_path"]
    prompt_wav = cfg["generation"]["prompt_wav"]
    token2wav = Token2wav(str(Path(model_path) / "token2wav"))
    token2wav.set_stream_cache(prompt_wav)

    client = StepAudio2VLLMClient(args.api_url, args.model_name)
    messages = [
        {"role": "system", "content": cfg["format"]["system_prompt"]},
        {"role": "human", "content": client.audio_content(args.audio)},
        {"role": "assistant", "content": "", "eot": False},
    ]
    sampling = {
        "max_tokens": int(cfg["generation"]["max_new_tokens"]),
        "temperature": float(cfg["generation"]["temperature"]),
        "top_p": float(cfg["generation"]["top_p"]),
        "repetition_penalty": float(cfg["generation"]["repetition_penalty"]),
        "skip_special_tokens": False,
        "parallel_tool_calls": False,
    }

    output_pcm = Path(args.output).with_suffix(".pcm")
    output_pcm.parent.mkdir(parents=True, exist_ok=True)
    output_pcm.unlink(missing_ok=True)
    buffer: list[int] = []
    for text, audio_tokens in client.stream(messages, **sampling):
        if text:
            print(text, end="", flush=True)
        if audio_tokens:
            buffer.extend([x for x in audio_tokens if 0 <= x <= 6560])
            lookahead = getattr(token2wav.flow, "pre_lookahead_len", 0)
            while len(buffer) >= CHUNK_SIZE + lookahead:
                chunk = buffer[: CHUNK_SIZE + lookahead]
                with output_pcm.open("ab") as f:
                    f.write(token2wav.stream(chunk, prompt_wav))
                buffer = buffer[CHUNK_SIZE:]
    if buffer:
        with output_pcm.open("ab") as f:
            f.write(token2wav.stream(buffer, prompt_wav, last_chunk=True))

    if not output_pcm.exists():
        raise RuntimeError("The vLLM server returned no audio tokens; check the prompt and max_tokens.")
    wav_path = Path(args.output)
    with output_pcm.open("rb") as f:
        pcm = f.read()
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm)
    print(f"\nWrote {wav_path}")


if __name__ == "__main__":
    main()
