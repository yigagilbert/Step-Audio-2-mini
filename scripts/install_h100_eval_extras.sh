#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "No active virtualenv detected. Activate it first, for example:"
  echo "source .venv/bin/activate"
  exit 1
fi

echo "Using Python: $(command -v python)"
python -m pip install --upgrade pip wheel setuptools

echo "Installing H100 evaluation extras..."
python -m pip install -r requirements_h100_eval.txt

echo "Checking the full H100 evaluation environment..."
python scripts/check_h100_eval_env.py --profile eval
