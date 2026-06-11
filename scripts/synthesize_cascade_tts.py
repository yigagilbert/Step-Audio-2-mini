from __future__ import annotations

import argparse
import gc
import json
import os
import unicodedata
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch
import yaml
from tqdm import tqdm


ORPHEUS_END_OF_TEXT = 128_009
ORPHEUS_START_OF_SPEECH = 128_257
ORPHEUS_END_OF_SPEECH = 128_258
ORPHEUS_START_OF_HUMAN = 128_259
ORPHEUS_END_OF_HUMAN = 128_260
ORPHEUS_AUDIO_TOKEN_LO = 128_266
ORPHEUS_AUDIO_TOKEN_HI = 128_266 + 7 * 4096
ORPHEUS_SNAC_VOCAB = 4096


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()]
    return ("".join(keep)[:100]).strip("_") or "sample"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    import torchaudio

    wav = torch.from_numpy(waveform.astype(np.float32))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.cpu(), sample_rate)


class OrpheusTTS:
    def __init__(
        self,
        model_id: str,
        codec_id: str,
        sample_rate: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int,
    ) -> None:
        self.model_id = model_id
        self.codec_id = codec_id
        self.sample_rate = sample_rate
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self._llm = None
        self._tokenizer = None
        self._snac = None

    def load(self) -> None:
        from snac import SNAC
        from transformers import AutoTokenizer
        from vllm import LLM

        llm_kwargs = {
            "model": self.model_id,
            "dtype": "bfloat16",
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": False,
            "trust_remote_code": False,
            "hf_overrides": {"vocab_size": 156_939},
        }
        try:
            self._llm = LLM(**llm_kwargs)
        except TypeError:
            llm_kwargs.pop("hf_overrides", None)
            self._llm = LLM(**llm_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        )
        self._snac = SNAC.from_pretrained(self.codec_id).to("cpu")

    def unload(self) -> None:
        try:
            if self._llm is not None and hasattr(self._llm, "llm_engine"):
                engine = self._llm.llm_engine
                if hasattr(engine, "model_executor"):
                    engine.model_executor.shutdown()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
        self._llm = None
        self._tokenizer = None
        self._snac = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def preprocess_text(text: str) -> str:
        allowlist = frozenset("abcdefghijklmnopqrstuvwxyz ',.")
        apostrophes = ("\u2018", "\u2019", "\u02bc", "`", "\u00b4")
        text = text.lower()
        for ch in apostrophes:
            text = text.replace(ch, "'")
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = "".join(ch for ch in text if ch in allowlist)
        return " ".join(text.split())

    def build_prompt_ids(self, text: str, speaker_id: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Orpheus tokenizer is not loaded.")
        tagged = f"{speaker_id}: {text}"
        text_ids = self._tokenizer.encode(tagged, add_special_tokens=True)
        return (
            [ORPHEUS_START_OF_HUMAN]
            + text_ids
            + [ORPHEUS_END_OF_TEXT, ORPHEUS_END_OF_HUMAN]
        )

    def tokens_to_waveform(self, generated_token_ids: list[int]) -> np.ndarray | None:
        if self._snac is None:
            raise RuntimeError("SNAC decoder is not loaded.")
        ids = torch.tensor(generated_token_ids, dtype=torch.int64)
        sos_pos = (ids == ORPHEUS_START_OF_SPEECH).nonzero(as_tuple=True)[0]
        if len(sos_pos) > 0:
            ids = ids[sos_pos[-1].item() + 1 :]

        audio = ids[(ids >= ORPHEUS_AUDIO_TOKEN_LO) & (ids < ORPHEUS_AUDIO_TOKEN_HI)]
        n = (audio.size(0) // 7) * 7
        if n == 0:
            return None

        cl = audio[:n].numpy().astype(np.int64) - ORPHEUS_AUDIO_TOKEN_LO
        cl = cl.reshape(-1, 7)

        layer2 = np.empty(len(cl) * 2, dtype=np.int64)
        layer2[0::2] = cl[:, 1] - 4096
        layer2[1::2] = cl[:, 4] - 4 * 4096

        layer3 = np.empty(len(cl) * 4, dtype=np.int64)
        layer3[0::4] = cl[:, 2] - 2 * 4096
        layer3[1::4] = cl[:, 3] - 3 * 4096
        layer3[2::4] = cl[:, 5] - 5 * 4096
        layer3[3::4] = cl[:, 6] - 6 * 4096

        vocab_max = ORPHEUS_SNAC_VOCAB - 1
        device = next(self._snac.parameters()).device

        def clamp(values: np.ndarray) -> torch.Tensor:
            return (
                torch.tensor(np.clip(values, 0, vocab_max), dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )

        codes = [clamp(cl[:, 0]), clamp(layer2), clamp(layer3)]
        with torch.inference_mode():
            waveform = self._snac.decode(codes)
        return waveform.detach().squeeze().cpu().numpy().astype(np.float32)

    def synthesize_batch(self, texts: list[str], speaker_ids: list[str]) -> list[np.ndarray | None]:
        from vllm import SamplingParams

        if self._llm is None:
            raise RuntimeError("Orpheus vLLM engine is not loaded.")
        if len(texts) != len(speaker_ids):
            raise ValueError("texts and speaker_ids must have the same length.")
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stop_token_ids=[ORPHEUS_END_OF_SPEECH],
            skip_special_tokens=False,
        )
        prompts = [
            {"prompt_token_ids": self.build_prompt_ids(self.preprocess_text(text), speaker)}
            for text, speaker in zip(texts, speaker_ids)
        ]
        outputs = self._llm.generate(prompts, sampling_params)
        return [self.tokens_to_waveform(list(output.outputs[0].token_ids)) for output in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize cascade English predictions with Sunbird Orpheus TTS."
    )
    parser.add_argument("--config", default="configs/h100_nvl_fast_deepspeed.yaml")
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Cascade predictions JSONL. Defaults to "
            "output_dir/eval/cascade_validation_predictions.jsonl."
        ),
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=450)
    parser.add_argument("--tts-model", default="Sunbird/orpheus-3b-tts-multilingual")
    parser.add_argument("--codec-model", default="hubertsiuzdak/snac_24khz")
    parser.add_argument("--speaker", default="salt_eng_0001")
    parser.add_argument(
        "--tts-batch-size",
        type=int,
        default=0,
        help="Number of prompts per vLLM generate call. Use 0 for all rows at once.",
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2625)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["project"]["output_dir"])
    processed_dir = Path(cfg["project"]["processed_dir"])
    predictions_path = Path(
        args.predictions or output_dir / "eval" / f"cascade_{args.split}_predictions.jsonl"
    )
    sample_dir = Path(args.output_dir or output_dir / "eval" / "cascade_audio_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing cascade predictions file: {predictions_path}. "
            "Run eval_cascade.py successfully before synthesizing cascade audio."
        )

    prepared_rows = {row["id"]: row for row in read_jsonl(processed_dir / f"{args.split}.jsonl")}
    predictions = read_jsonl(predictions_path)
    if args.limit is not None:
        predictions = predictions[: args.limit]

    tts = OrpheusTTS(
        model_id=args.tts_model,
        codec_id=args.codec_model,
        sample_rate=args.sample_rate,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    print(f"Loading TTS model: {args.tts_model}")
    tts.load()

    manifest = []
    rows_to_synthesize = []
    for idx, row in enumerate(predictions):
        text = str(row.get("prediction") or row.get("hyp") or "").strip()
        if not text:
            continue
        rows_to_synthesize.append((idx, row, text[: args.max_chars]))

    try:
        if rows_to_synthesize:
            batch_size = (
                args.tts_batch_size
                if args.tts_batch_size and args.tts_batch_size > 0
                else len(rows_to_synthesize)
            )
            for start in tqdm(
                range(0, len(rows_to_synthesize), batch_size),
                desc="Cascade Orpheus TTS",
            ):
                batch = rows_to_synthesize[start : start + batch_size]
                texts = [item[2] for item in batch]
                speakers = [args.speaker] * len(batch)
                waveforms = tts.synthesize_batch(texts, speakers)
                for (idx, row, text), waveform in zip(batch, waveforms):
                    if waveform is None:
                        continue
                    row_id = str(row.get("id", idx))
                    wav_path = sample_dir / f"{idx:04d}_{safe_name(row_id)}.wav"
                    write_wav(wav_path, waveform, args.sample_rate)
                    manifest.append(
                        {
                            "id": row_id,
                            "hyp_audio": str(wav_path),
                            "wav": str(wav_path),
                            "ref_audio": str(
                                processed_dir / args.split / "wav" / f"{row_id}.eng.wav"
                            ),
                            "source": prepared_rows.get(row_id, {}).get(
                                "text_lug", row.get("source", "")
                            ),
                            "asr_text": row.get("asr_text", ""),
                            "reference": row.get("reference") or row.get("ref"),
                            "prediction": row.get("prediction") or row.get("hyp") or text,
                            "tts_model": args.tts_model,
                            "codec_model": args.codec_model,
                            "speaker": args.speaker,
                            "sample_rate": args.sample_rate,
                        }
                    )
    finally:
        tts.unload()

    manifest_path = sample_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_jsonl_path = sample_dir / "manifest.jsonl"
    with manifest_jsonl_path.open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(manifest)} wav files to {sample_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote JSONL manifest to {manifest_jsonl_path}")


if __name__ == "__main__":
    main()
