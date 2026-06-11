from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stepaudio_luganda.audio import load_audio, log_mel_spectrogram  # noqa: E402
from stepaudio_luganda.formatting import StepAudioFormatter  # noqa: E402
from stepaudio_luganda.modeling import load_model, load_tokenizer  # noqa: E402
from stepaudio_luganda.paths import resolve_prompt_wav  # noqa: E402
from stepaudio_luganda.torchcodec_compat import patch_torchaudio_bytesio_save  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_prediction_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text.splitlines()[0].strip()


def existing_path(value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def resolve_base_model_path(model_ref: str) -> str:
    local_path = existing_path(model_ref)
    if local_path:
        return str(local_path)
    print(f"Downloading or reusing cached base model snapshot: {model_ref}")
    return snapshot_download(repo_id=model_ref)


def choose_base_model_ref(args_base_model: str | None, model_cfg: dict[str, Any]) -> str:
    if args_base_model:
        return args_base_model
    local_path = existing_path(model_cfg.get("local_path"))
    if local_path:
        return str(local_path)
    return str(model_cfg["name_or_path"])


def extract_audio_and_text(tokenizer, formatter, token_ids: list[int]) -> tuple[str, list[int]]:
    text_ids: list[int] = []
    audio_tokens: list[int] = []
    seen_audio = False
    for token_id in token_ids:
        if token_id == formatter.eot_id:
            break
        if token_id == formatter.tts_start_id:
            seen_audio = True
            continue
        if token_id == formatter.tts_end_id:
            break
        if token_id >= formatter.audio_token_offset:
            audio_token = token_id - formatter.audio_token_offset
            if 0 <= audio_token <= formatter.tts_valid_max:
                audio_tokens.append(audio_token)
            seen_audio = True
        elif not seen_audio and token_id < formatter.audio_start_id:
            text_ids.append(token_id)
    text = clean_prediction_text(tokenizer.decode(text_ids, skip_special_tokens=True))
    return text, audio_tokens


def import_token2wav(stepaudio2_repo: str | Path):
    repo_path = Path(stepaudio2_repo).expanduser()
    if not (repo_path / "token2wav.py").exists():
        raise FileNotFoundError(
            f"Missing {repo_path / 'token2wav.py'}. Clone the official Step-Audio2 repo "
            "or pass --stepaudio2-repo to its local path."
        )
    sys.path.insert(0, str(repo_path.resolve()))
    patch_torchaudio_bytesio_save()
    from token2wav import Token2wav  # noqa: PLC0415

    return Token2wav


def bool_from_config_or_arg(config_value: Any, override: bool | None) -> bool:
    if override is not None:
        return override
    return bool(config_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate one Luganda audio file with a Hugging Face LoRA adapter."
    )
    parser.add_argument("--config", default="configs/h100_nvl_fast_deepspeed.yaml")
    parser.add_argument("--audio", required=True, help="Input Luganda audio path.")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Hugging Face LoRA adapter repo ID or local adapter path.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base Step-Audio model path/repo. Defaults to config local_path/name_or_path.",
    )
    parser.add_argument("--stepaudio2-repo", default="Step-Audio2")
    parser.add_argument("--prompt-wav", default=None)
    parser.add_argument("--output-audio", default="outputs/hf_single_translation.wav")
    parser.add_argument("--output-text", default="outputs/hf_single_translation.txt")
    parser.add_argument("--output-json", default="outputs/hf_single_translation.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = dict(cfg["model"])
    base_ref = choose_base_model_ref(args.base_model, model_cfg)
    base_model_path = resolve_base_model_path(base_ref)

    tokenizer = load_tokenizer(
        base_model_path,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    formatter = StepAudioFormatter(
        tokenizer,
        system_prompt=cfg["format"]["system_prompt"],
        target_format=cfg["format"]["target_format"],
        max_target_audio_tokens=cfg["format"].get("max_target_audio_tokens"),
    )

    print(f"Loading base model: {base_model_path}")
    model = load_model(base_model_path, model_cfg)
    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    audio = load_audio(args.audio, target_rate=16000)
    mel = log_mel_spectrogram(audio, n_mels=128, padding=479)
    prompt_ids = formatter.build_prompt(int(mel.shape[1]))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    wavs = mel.unsqueeze(0).to(device=device, dtype=torch.float32)
    wav_lens = torch.tensor([max(1, int(mel.shape[1]) - 2)], dtype=torch.int32, device=device)

    gen_cfg = cfg["generation"]
    max_new_tokens = args.max_new_tokens or int(gen_cfg["max_new_tokens"])
    temperature = (
        args.temperature if args.temperature is not None else float(gen_cfg["temperature"])
    )
    top_p = args.top_p if args.top_p is not None else float(gen_cfg["top_p"])
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else float(gen_cfg["repetition_penalty"])
    )
    do_sample = bool_from_config_or_arg(gen_cfg["do_sample"], args.do_sample)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            wavs=wavs,
            wav_lens=wav_lens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=formatter.eot_id,
        )

    new_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
    text, audio_tokens = extract_audio_and_text(tokenizer, formatter, new_ids)
    if not text:
        print("[WARN] Model returned empty text.")
    if not audio_tokens:
        raise RuntimeError(
            "Model returned no audio tokens. Try increasing --max-new-tokens "
            "or check target_format."
        )

    Token2wav = import_token2wav(args.stepaudio2_repo)
    token2wav = Token2wav(str(Path(base_model_path) / "token2wav"))
    prompt_wav = resolve_prompt_wav(
        args.prompt_wav or gen_cfg["prompt_wav"],
        model_path=base_model_path,
        stepaudio2_repo=args.stepaudio2_repo,
        root=ROOT,
    )
    wav_bytes = token2wav(audio_tokens, prompt_wav=str(prompt_wav))

    output_audio = Path(args.output_audio)
    output_text = Path(args.output_text)
    output_json = Path(args.output_json)
    for path in [output_audio, output_text, output_json]:
        path.parent.mkdir(parents=True, exist_ok=True)

    output_audio.write_bytes(wav_bytes)
    output_text.write_text(text + "\n", encoding="utf-8")
    output_json.write_text(
        json.dumps(
            {
                "input_audio": str(Path(args.audio).expanduser()),
                "adapter": args.adapter,
                "base_model": base_model_path,
                "prompt_wav": str(prompt_wav),
                "prediction": text,
                "audio_tokens": len(audio_tokens),
                "output_audio": str(output_audio),
                "output_text": str(output_text),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(text)
    print(f"Wrote English audio: {output_audio}")
    print(f"Wrote English text: {output_text}")
    print(f"Wrote metadata: {output_json}")


if __name__ == "__main__":
    main()
