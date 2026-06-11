from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import sacrebleu
import torch
import torchaudio
from scipy.spatial.distance import cdist
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def system_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=manifest_path")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("System name cannot be empty.")
    return name, Path(path)


def get_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def get_path(record: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = record.get(key)
        if value:
            return Path(value)
    return None


def filter_to_common_ids(
    systems: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    id_sets = []
    for name, records in systems.items():
        ids = {str(record.get("id")) for record in records if record.get("id") is not None}
        if not ids:
            raise ValueError(f"System {name!r} has no records with an 'id' field.")
        id_sets.append(ids)
    common_ids = set.intersection(*id_sets) if id_sets else set()
    if not common_ids:
        raise ValueError("No common record IDs found across systems.")

    filtered = {}
    for name, records in systems.items():
        filtered[name] = [
            record for record in records if str(record.get("id")) in common_ids
        ]
    return filtered, sorted(common_ids)


def load_audio(path: Path, target_rate: int = 16000) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.to(torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.squeeze()
    if int(sample_rate) != target_rate:
        waveform = torchaudio.transforms.Resample(int(sample_rate), target_rate)(waveform)
    return waveform.contiguous()


def compute_chrf(records: list[dict[str, Any]]) -> float:
    predictions = [get_text(row, "prediction", "hyp", "hyp_text") for row in records]
    references = [get_text(row, "reference", "ref", "ref_text") for row in records]
    return sacrebleu.corpus_chrf(predictions, [references]).score if predictions else 0.0


def compute_mcd_pair(
    hyp_audio: Path,
    ref_audio: Path,
    sample_rate: int = 16000,
    n_mfcc: int = 25,
    hop_length: int = 256,
) -> float:
    hyp, _ = librosa.load(hyp_audio, sr=sample_rate, mono=True)
    ref, _ = librosa.load(ref_audio, sr=sample_rate, mono=True)
    hyp_mfcc = librosa.feature.mfcc(
        y=hyp,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )[1:].T
    ref_mfcc = librosa.feature.mfcc(
        y=ref,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )[1:].T
    if hyp_mfcc.size == 0 or ref_mfcc.size == 0:
        return float("nan")
    cost = cdist(hyp_mfcc, ref_mfcc, metric="euclidean")
    _, path = librosa.sequence.dtw(C=cost, backtrack=True)
    distances = cost[path[:, 0], path[:, 1]]
    constant = 10.0 / math.log(10.0) * math.sqrt(2.0)
    return float(constant * np.mean(distances))


def compute_mcd(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = []
    missing = 0
    for row in tqdm(records, desc="MCD", leave=False):
        hyp_audio = get_path(row, "hyp_audio", "wav", "output_audio")
        ref_audio = get_path(row, "ref_audio", "reference_audio")
        if not hyp_audio or not ref_audio or not hyp_audio.exists() or not ref_audio.exists():
            missing += 1
            continue
        score = compute_mcd_pair(hyp_audio, ref_audio)
        if not math.isnan(score):
            scores.append(score)
    return {
        "mcd": float(np.mean(scores)) if scores else None,
        "mcd_count": len(scores),
        "mcd_missing": missing,
    }


class SpeechBERTScorer:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def embed(self, audio_path: Path) -> torch.Tensor:
        waveform = load_audio(audio_path, target_rate=16000)
        inputs = self.processor(
            waveform.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        hidden = self.model(**inputs).last_hidden_state.squeeze(0)
        return torch.nn.functional.normalize(hidden.float(), dim=-1).cpu()

    def score_pair(self, hyp_audio: Path, ref_audio: Path) -> dict[str, float]:
        hyp = self.embed(hyp_audio)
        ref = self.embed(ref_audio)
        sim = hyp @ ref.T
        precision = sim.max(dim=1).values.mean().item()
        recall = sim.max(dim=0).values.mean().item()
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        return {
            "speechbertscore_precision": precision,
            "speechbertscore_recall": recall,
            "speechbertscore_f1": f1,
        }


def compute_speechbertscore(
    records: list[dict[str, Any]],
    model_name: str,
    device: torch.device,
) -> dict[str, Any]:
    scorer = SpeechBERTScorer(model_name=model_name, device=device)
    values: dict[str, list[float]] = {
        "speechbertscore_precision": [],
        "speechbertscore_recall": [],
        "speechbertscore_f1": [],
    }
    missing = 0
    for row in tqdm(records, desc="SpeechBERTScore", leave=False):
        hyp_audio = get_path(row, "hyp_audio", "wav", "output_audio")
        ref_audio = get_path(row, "ref_audio", "reference_audio")
        if not hyp_audio or not ref_audio or not hyp_audio.exists() or not ref_audio.exists():
            missing += 1
            continue
        pair = scorer.score_pair(hyp_audio, ref_audio)
        for key, value in pair.items():
            values[key].append(value)
    return {
        key: float(np.mean(score_values)) if score_values else None
        for key, score_values in values.items()
    } | {
        "speechbertscore_count": len(values["speechbertscore_f1"]),
        "speechbertscore_missing": missing,
        "speechbertscore_model": model_name,
    }


def compute_blaser(
    records: list[dict[str, Any]],
    src_lang: str,
    mt_lang: str,
    device: torch.device,
) -> dict[str, Any]:
    try:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.models.blaser.loader import load_blaser_model
    except ImportError as exc:
        raise RuntimeError(
            "BLASER 2.0 requires sonar-space and a matching fairseq2 install. "
            "Install fairseq2 for your torch/CUDA build, then install sonar-space."
        ) from exc

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    embedder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
        dtype=dtype,
    )
    try:
        blaser_ref = load_blaser_model("blaser_2_0_ref", device=device, dtype=dtype).eval()
        blaser_qe = load_blaser_model("blaser_2_0_qe", device=device, dtype=dtype).eval()
    except TypeError:
        blaser_ref = load_blaser_model("blaser_2_0_ref").to(device).to(dtype).eval()
        blaser_qe = load_blaser_model("blaser_2_0_qe").to(device).to(dtype).eval()

    sources = [get_text(row, "source", "src", "text_lug") for row in records]
    references = [get_text(row, "reference", "ref", "ref_text") for row in records]
    predictions = [get_text(row, "prediction", "hyp", "hyp_text") for row in records]
    if not predictions:
        return {"blaser_2_0_ref": None, "blaser_2_0_qe": None}

    src_embs = embedder.predict(sources, source_lang=src_lang).to(device).to(dtype)
    ref_embs = embedder.predict(references, source_lang=mt_lang).to(device).to(dtype)
    mt_embs = embedder.predict(predictions, source_lang=mt_lang).to(device).to(dtype)
    with torch.inference_mode():
        ref_scores = blaser_ref(src=src_embs, ref=ref_embs, mt=mt_embs).detach().float().cpu()
        qe_scores = blaser_qe(src=src_embs, mt=mt_embs).detach().float().cpu()
    return {
        "blaser_2_0_ref": float(ref_scores.mean().item()),
        "blaser_2_0_qe": float(qe_scores.mean().item()),
        "blaser_count": len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        type=system_arg,
        action="append",
        required=True,
        help="System manifest as NAME=path. May be passed multiple times.",
    )
    parser.add_argument("--output", default="outputs/advanced_metrics.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--src-lang", default="lug_Latn")
    parser.add_argument("--mt-lang", default="eng_Latn")
    parser.add_argument("--skip-blaser", action="store_true")
    parser.add_argument("--skip-speechbertscore", action="store_true")
    parser.add_argument("--skip-mcd", action="store_true")
    parser.add_argument("--speech-model", default="microsoft/wavlm-large")
    parser.add_argument(
        "--align-ids",
        action="store_true",
        help="Evaluate only record IDs present in every system manifest.",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    systems = {name: read_records(path) for name, path in args.system}
    aligned_ids = None
    if args.align_ids:
        systems, aligned_ids = filter_to_common_ids(systems)

    results: dict[str, Any] = {}
    for name, records in systems.items():
        metrics: dict[str, Any] = {
            "count": len(records),
            "chrf": compute_chrf(records),
        }
        if aligned_ids is not None:
            metrics["aligned_id_count"] = len(aligned_ids)
        if not args.skip_blaser:
            metrics.update(compute_blaser(records, args.src_lang, args.mt_lang, device))
        if not args.skip_speechbertscore:
            metrics.update(compute_speechbertscore(records, args.speech_model, device))
        if not args.skip_mcd:
            metrics.update(compute_mcd(records))
        results[name] = metrics

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
