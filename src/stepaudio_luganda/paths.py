from __future__ import annotations

from pathlib import Path


def resolve_prompt_wav(
    prompt_wav: str | Path,
    *,
    model_path: str | Path,
    stepaudio2_repo: str | Path | None = None,
    root: str | Path | None = None,
) -> Path:
    raw = Path(prompt_wav).expanduser()
    root_path = Path(root or Path.cwd()).resolve()
    model_path = Path(model_path).expanduser()
    stepaudio2_path = Path(stepaudio2_repo).expanduser() if stepaudio2_repo else None

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                root_path / raw,
                model_path / raw,
            ]
        )
        if stepaudio2_path:
            candidates.append(stepaudio2_path / raw)

    filename = raw.name
    candidates.extend(
        [
            model_path / "assets" / filename,
            model_path / "token2wav" / filename,
        ]
    )
    if stepaudio2_path:
        candidates.extend(
            [
                stepaudio2_path / "assets" / filename,
                stepaudio2_path / "examples" / filename,
                stepaudio2_path / filename,
            ]
        )

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.exists():
            return candidate.resolve()

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Prompt wav not found. Set generation.prompt_wav or pass --prompt-wav to an existing "
        f"speaker prompt wav.\nChecked:\n{checked_text}"
    )
