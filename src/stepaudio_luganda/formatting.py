from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from .audio import compute_token_num
from .constants import (
    AUDIO_END,
    AUDIO_PATCH,
    AUDIO_START,
    BOT,
    DEFAULT_AUDIO_TOKEN_OFFSET,
    DEFAULT_TTS_VALID_MAX,
    EOT,
    IGNORE_INDEX,
    TTS_END,
    TTS_START,
)


@dataclass
class FormattedSample:
    input_ids: list[int]
    labels: list[int]
    prompt_length: int


class StepAudioFormatter:
    """Builds Step-Audio 2 style chat sequences for SFT and generation."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        system_prompt: str,
        target_format: str = "text_then_audio",
        max_target_audio_tokens: int | None = None,
        tts_valid_max: int = DEFAULT_TTS_VALID_MAX,
    ) -> None:
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.target_format = target_format
        self.max_target_audio_tokens = max_target_audio_tokens
        self.tts_valid_max = tts_valid_max

        self.audio_start_id = self._token_id(AUDIO_START)
        self.audio_end_id = self._token_id(AUDIO_END)
        self.audio_patch_id = self._token_id(AUDIO_PATCH)
        self.tts_start_id = self._token_id(TTS_START)
        self.tts_end_id = self._token_id(TTS_END)
        self.eot_id = self._token_id(EOT)
        self.audio_token_offset = self._token_id("<audio_0>", DEFAULT_AUDIO_TOKEN_OFFSET)

    def _token_id(self, token: str, fallback: int | None = None) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            if fallback is not None:
                return fallback
            raise ValueError(f"Tokenizer does not define required token {token!r}")
        return int(token_id)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def audio_placeholder_ids(self, mel_frames: int) -> list[int]:
        patch_count = compute_token_num(mel_frames)
        return [self.audio_start_id] + [self.audio_patch_id] * patch_count + [self.audio_end_id]

    def build_prompt(self, mel_frames: int) -> list[int]:
        ids: list[int] = []
        ids += self.encode_text(f"{BOT}system\n{self.system_prompt}{EOT}")
        ids += self.encode_text(f"{BOT}human\n")
        ids += self.audio_placeholder_ids(mel_frames)
        ids += [self.eot_id]
        ids += self.encode_text(f"{BOT}assistant\n")
        return ids

    def audio_tokens_to_vocab_ids(self, audio_tokens: list[int]) -> list[int]:
        filtered = [
            int(t)
            for t in audio_tokens
            if isinstance(t, int) or (isinstance(t, str) and str(t).isdigit())
        ]
        filtered = [t for t in filtered if 0 <= t <= self.tts_valid_max]
        if self.max_target_audio_tokens:
            filtered = filtered[: self.max_target_audio_tokens]
        return [self.audio_token_offset + t for t in filtered]

    def build_target(self, text_eng: str, audio_tokens: list[int]) -> list[int]:
        text_eng = (text_eng or "").strip()
        if self.target_format == "text_only":
            return self.encode_text(text_eng) + [self.eot_id]

        audio_ids = [self.tts_start_id] + self.audio_tokens_to_vocab_ids(audio_tokens) + [self.tts_end_id]
        if self.target_format == "audio_only":
            return audio_ids + [self.eot_id]
        if self.target_format == "text_then_audio":
            text_ids = self.encode_text(text_eng + "\n") if text_eng else []
            return text_ids + audio_ids + [self.eot_id]
        raise ValueError(f"Unsupported target_format={self.target_format!r}")

    def format_sft(
        self,
        mel_frames: int,
        text_eng: str,
        audio_tokens: list[int],
    ) -> FormattedSample:
        prompt = self.build_prompt(mel_frames)
        target = self.build_target(text_eng=text_eng, audio_tokens=audio_tokens)
        return FormattedSample(
            input_ids=prompt + target,
            labels=[IGNORE_INDEX] * len(prompt) + target,
            prompt_length=len(prompt),
        )
