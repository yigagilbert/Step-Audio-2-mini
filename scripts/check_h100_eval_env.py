from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shlex
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModuleCheck:
    label: str
    module: str
    pip_name: str
    profiles: tuple[str, ...]
    note: str = ""


CHECKS = [
    ModuleCheck("PyYAML", "yaml", "PyYAML>=6.0.2", ("core",)),
    ModuleCheck("NumPy", "numpy", "numpy>=1.26.4", ("core",)),
    ModuleCheck("Torch", "torch", "torch==2.5.1", ("core",)),
    ModuleCheck("TorchAudio", "torchaudio", "torchaudio==2.5.1", ("core",)),
    ModuleCheck("Transformers", "transformers", "transformers==4.49.0", ("core",)),
    ModuleCheck("Datasets", "datasets", "datasets[audio]>=2.21.0", ("core",)),
    ModuleCheck("PEFT", "peft", "peft>=0.12.0", ("core",)),
    ModuleCheck("SacreBLEU", "sacrebleu", "sacrebleu>=2.4.3", ("core",)),
    ModuleCheck("JiWER", "jiwer", "jiwer>=3.0.4", ("core",)),
    ModuleCheck("tqdm", "tqdm", "tqdm>=4.66.5", ("core",)),
    ModuleCheck("Librosa", "librosa", "librosa>=0.10.2.post1", ("core",)),
    ModuleCheck("SoundFile", "soundfile", "soundfile>=0.12.1", ("core",)),
    ModuleCheck("SentencePiece", "sentencepiece", "sentencepiece>=0.2.0", ("core",)),
    ModuleCheck("HF Hub", "huggingface_hub", "huggingface_hub>=0.24.6", ("core",)),
    ModuleCheck("ONNX Runtime", "onnxruntime", "onnxruntime-gpu>=1.18.0", ("core",)),
    ModuleCheck("s3tokenizer", "s3tokenizer", "s3tokenizer", ("stepaudio_tts",)),
    ModuleCheck("Diffusers", "diffusers", "diffusers", ("stepaudio_tts",)),
    ModuleCheck("HyperPyYAML", "hyperpyyaml", "hyperpyyaml", ("stepaudio_tts",)),
    ModuleCheck("COMET", "comet", "unbabel-comet>=2.2.0", ("text_eval",)),
    ModuleCheck("SciPy", "scipy", "scipy>=1.11.0", ("audio_metrics",)),
    ModuleCheck(
        "SONAR",
        "sonar",
        "sonar-space~=0.5.0",
        ("blaser",),
        "Needed only for BLASER 2.0.",
    ),
    ModuleCheck(
        "fairseq2",
        "fairseq2",
        "fairseq2",
        ("blaser",),
        "Must exactly match your PyTorch/CUDA build.",
    ),
    ModuleCheck("SNAC", "snac", "snac>=1.2.1", ("cascade_tts",)),
    ModuleCheck("vLLM", "vllm", "vllm==0.7.3", ("cascade_tts",)),
]


PROFILE_GROUPS = {
    "core": {"core"},
    "text": {"core", "text_eval"},
    "stepaudio-tts": {"core", "stepaudio_tts"},
    "cascade-tts": {"core", "cascade_tts"},
    "audio-metrics": {"core", "audio_metrics"},
    "advanced": {"core", "audio_metrics"},
    "blaser": {"core", "blaser"},
    "eval": {"core", "text_eval", "stepaudio_tts", "cascade_tts", "audio_metrics"},
    "all": {"core", "text_eval", "stepaudio_tts", "cascade_tts", "audio_metrics", "blaser"},
}


def selected_checks(profile: str) -> list[ModuleCheck]:
    groups = PROFILE_GROUPS[profile]
    return [check for check in CHECKS if groups.intersection(check.profiles)]


def check_import(module: str, find_spec_only: bool) -> tuple[bool, str | None]:
    try:
        if find_spec_only:
            spec = importlib.util.find_spec(module)
            if spec is None:
                return False, "module spec not found"
        else:
            importlib.import_module(module)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def torch_summary() -> dict[str, Any]:
    try:
        import torch

        summary: dict[str, Any] = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            summary["cuda_device"] = torch.cuda.get_device_name(0)
            summary["bf16_supported"] = torch.cuda.is_bf16_supported()
        return summary
    except Exception as exc:
        return {"torch_error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check imports needed for the H100 Step-Audio evaluation stack."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_GROUPS),
        default="all",
        help="Dependency group to check.",
    )
    parser.add_argument(
        "--find-spec-only",
        action="store_true",
        help="Only check import specs instead of importing modules.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    results = []
    missing = []
    for check in selected_checks(args.profile):
        ok, error = check_import(check.module, args.find_spec_only)
        result = {
            "label": check.label,
            "module": check.module,
            "ok": ok,
            "pip_name": check.pip_name,
            "error": error,
            "note": check.note,
        }
        results.append(result)
        if not ok:
            missing.append(check)

    payload = {
        "profile": args.profile,
        "torch": torch_summary(),
        "checks": results,
        "missing_pip_names": list(dict.fromkeys(check.pip_name for check in missing)),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("H100 evaluation environment check")
        print(f"profile: {args.profile}")
        print(json.dumps(payload["torch"], indent=2, ensure_ascii=False))
        print()
        for result in results:
            status = "OK" if result["ok"] else "MISSING"
            print(f"{status:8} {result['label']:<16} import {result['module']}")
            if result["error"]:
                print(f"         {result['error']}")
            if result["note"]:
                print(f"         {result['note']}")
        if missing:
            print()
            print("Install missing packages:")
            quoted = [shlex.quote(name) for name in payload["missing_pip_names"]]
            print("python -m pip install " + " ".join(quoted))
            if args.profile == "blaser" and any(check.module == "fairseq2" for check in missing):
                print()
                print("For BLASER, prefer scripts/setup_blaser_env.sh")
                print("so fairseq2 is installed against a matching torch/CUDA build.")
            elif any(check.module == "fairseq2" for check in missing):
                print()
                print("For fairseq2, use a dedicated BLASER environment.")
                print("Run scripts/setup_blaser_env.sh instead of changing the main .venv.")
        else:
            print()
            print("All requested imports are available.")

    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
