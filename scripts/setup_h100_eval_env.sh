#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  "torch==${PYTORCH_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_CUDA_INDEX}"
python -m pip install -r requirements.txt

echo "Detecting PyTorch/CUDA build for fairseq2..."
read -r TORCH_MM CUDA_TAG < <(
python - <<'PY'
import re
import torch

version = torch.__version__.split("+")[0]
match = re.match(r"(\d+\.\d+)", version)
torch_mm = match.group(1) if match else version
cuda = torch.version.cuda
cuda_tag = "cpu" if cuda is None else "cu" + cuda.replace(".", "")
print(torch_mm, cuda_tag)
PY
)

if [[ "${CUDA_TAG}" == "cpu" ]]; then
  echo "Installing CPU fairseq2 wheel"
  python -m pip install fairseq2
else
  FAIRSEQ2_INDEX="https://fair.pkg.atmeta.com/fairseq2/whl/pt${TORCH_MM}/${CUDA_TAG}"
  echo "Installing fairseq2 from ${FAIRSEQ2_INDEX}"
  python -m pip install fairseq2 --extra-index-url "${FAIRSEQ2_INDEX}"
fi

python -m pip install -r requirements_h100_eval.txt

echo "Environment ready. Activate it with:"
echo "source ${VENV_DIR}/bin/activate"
