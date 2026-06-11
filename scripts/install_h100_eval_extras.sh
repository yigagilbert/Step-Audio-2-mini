#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "No active virtualenv detected. Activate it first, for example:"
  echo "source .venv/bin/activate"
  exit 1
fi

echo "Using Python: $(command -v python)"
python -m pip install --upgrade pip wheel setuptools

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

echo "Installing H100 evaluation extras..."
python -m pip install -r requirements_h100_eval.txt

echo "Checking the full H100 evaluation environment..."
python scripts/check_h100_eval_env.py --profile all

