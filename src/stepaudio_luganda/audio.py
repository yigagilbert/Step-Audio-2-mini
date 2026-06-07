from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence


def _mel_filters(n_mels: int) -> torch.Tensor:
    if n_mels not in {80, 128}:
        raise ValueError(f"Unsupported n_mels={n_mels}; Step-Audio 2 mini uses 128.")
    return torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels))


def load_audio(path: str | Path, target_rate: int = 16000, max_length: int | None = None) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = _to_mono_resampled(waveform, sample_rate, target_rate)
    if max_length is not None and waveform.numel() > max_length:
        waveform = waveform[:max_length]
    return waveform.contiguous()


def _as_float_tensor(audio: Any) -> torch.Tensor:
    tensor = torch.as_tensor(audio)
    if tensor.numel() == 0:
        return tensor.to(dtype=torch.float32)
    if tensor.dtype.is_floating_point:
        return tensor.to(dtype=torch.float32)
    if tensor.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        info = torch.iinfo(tensor.dtype)
        scale = float(max(abs(info.min), abs(info.max)))
        return tensor.to(dtype=torch.float32) / scale
    return tensor.to(dtype=torch.float32)


def _to_mono_resampled(audio: Any, sample_rate: int, target_rate: int) -> torch.Tensor:
    waveform = _as_float_tensor(audio)
    if waveform.ndim == 0:
        raise ValueError("Audio sample decoded to a scalar.")
    waveform = waveform.squeeze()
    if waveform.ndim == 2:
        # TorchCodec/torchaudio usually use [channels, frames], while some arrays use
        # [frames, channels]. Use the smaller dimension as the channel axis.
        waveform = waveform.mean(dim=0) if waveform.shape[0] <= waveform.shape[1] else waveform.mean(dim=1)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono or 2D audio, got shape {tuple(waveform.shape)}.")
    sample_rate = int(sample_rate or target_rate)
    if sample_rate != target_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_rate)(waveform)
    return waveform.contiguous()


def _decode_audio_bytes(blob: bytes, target_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(io.BytesIO(blob))
    return _to_mono_resampled(waveform, sample_rate, target_rate)


def _torchcodec_decoder_to_waveform(decoder: Any, target_rate: int) -> torch.Tensor:
    samples = decoder.get_all_samples()
    data = getattr(samples, "data", None)
    sample_rate = getattr(samples, "sample_rate", None)
    if data is None and isinstance(samples, dict):
        data = samples.get("data")
        if data is None:
            data = samples.get("array")
        sample_rate = samples.get("sample_rate") or samples.get("sampling_rate")
    if data is None:
        raise TypeError("TorchCodec audio decoder returned samples without a data field.")
    return _to_mono_resampled(data, int(sample_rate or target_rate), target_rate)


def audio_cell_to_waveform(cell: Any, target_rate: int = 16000) -> torch.Tensor:
    """Accept HF Audio cells, TorchCodec decoders, file paths, or arrays."""
    if isinstance(cell, (str, Path)):
        return load_audio(cell, target_rate=target_rate)
    if torch.is_tensor(cell) or isinstance(cell, np.ndarray):
        return _to_mono_resampled(cell, target_rate, target_rate)
    if hasattr(cell, "get_all_samples"):
        return _torchcodec_decoder_to_waveform(cell, target_rate=target_rate)
    if isinstance(cell, dict):
        if cell.get("array") is not None:
            sr = int(cell.get("sampling_rate") or target_rate)
            return _to_mono_resampled(cell["array"], sr, target_rate)
        if cell.get("bytes") is not None:
            return _decode_audio_bytes(cell["bytes"], target_rate=target_rate)
        if cell.get("path"):
            return load_audio(cell["path"], target_rate=target_rate)
    raise TypeError(f"Unsupported audio cell type: {type(cell)!r}")


def energy_trim(waveform: torch.Tensor, top_db: int = 35) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    trimmed, _ = librosa.effects.trim(waveform.cpu().numpy(), top_db=top_db)
    return torch.from_numpy(np.ascontiguousarray(trimmed)).to(torch.float32)


def log_mel_spectrogram(
    audio: torch.Tensor | np.ndarray | str | Path,
    n_mels: int = 128,
    padding: int = 479,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(audio, (str, Path)):
        audio = load_audio(audio)
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(np.asarray(audio))
    audio = audio.to(dtype=torch.float32)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400, device=audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = _mel_filters(n_mels).to(audio.device)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def compute_token_num(max_feature_len: int) -> int:
    max_feature_len = max_feature_len - 2
    encoder_output_dim = (max_feature_len + 1) // 2 // 2
    padding = 1
    kernel_size = 3
    stride = 2
    return (encoder_output_dim + 2 * padding - kernel_size) // stride + 1


def pad_mels(mels: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([m.size(1) - 2 for m in mels], dtype=torch.int32)
    padded = pad_sequence([m.t() for m in mels], batch_first=True, padding_value=0.0)
    return padded.transpose(1, 2), lengths


def save_waveform(path: str | Path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform.unsqueeze(0).cpu(), sample_rate=sample_rate)
