#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-blaser}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.6.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
FAIRSEQ2_VERSION="${FAIRSEQ2_VERSION:-0.6.*}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
FAIRSEQ2_INDEX="${FAIRSEQ2_INDEX:-https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124}"
RECREATE="${RECREATE:-0}"

echo "Creating BLASER virtual environment at ${VENV_DIR}"
if [[ -d "${VENV_DIR}" ]]; then
  if [[ "${RECREATE}" == "1" ]]; then
    echo "Removing existing ${VENV_DIR} because RECREATE=1"
    rm -rf "${VENV_DIR}"
  else
    echo "${VENV_DIR} already exists."
    echo "Run with RECREATE=1 to rebuild it from scratch:"
    echo "RECREATE=1 scripts/setup_blaser_env.sh"
    exit 1
  fi
fi
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "Installing PyTorch ${PYTORCH_VERSION} / torchaudio ${TORCHAUDIO_VERSION}"
python -m pip install \
  "torch==${PYTORCH_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_CUDA_INDEX}"

echo "Installing fairseq2 ${FAIRSEQ2_VERSION} from ${FAIRSEQ2_INDEX}"
python -m pip install \
  "fairseq2==${FAIRSEQ2_VERSION}" \
  --extra-index-url "${FAIRSEQ2_INDEX}"

echo "Installing BLASER evaluation requirements"
python -m pip install \
  -r requirements_blaser_eval.txt \
  -c requirements_blaser_constraints.txt

echo "Checking BLASER environment"
python scripts/check_h100_eval_env.py --profile blaser

echo "BLASER environment ready. Activate it with:"
echo "source ${VENV_DIR}/bin/activate"
