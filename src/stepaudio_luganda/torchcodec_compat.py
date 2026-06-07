from __future__ import annotations

import io
import tempfile
from pathlib import Path


def patch_torchaudio_bytesio_save() -> None:
    import torchaudio

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
