#!/usr/bin/env bash
# One-time environment setup on the AutoDL server (uses the current python; no virtualenv needed).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
echo "[SETUP] python: $($PY -V)"
# Keep the versions that the server already has (numpy 1.26.4, scipy 1.15.2, pandas 2.2.3, sklearn 1.7.2,
# xgboost 3.1.1, joblib 1.5.2) and add only what is missing.
need=()
$PY -c "import lightgbm" 2>/dev/null || need+=("lightgbm==4.6.0")
$PY -c "import catboost" 2>/dev/null || need+=("catboost==1.2.10")
$PY -c "import openpyxl" 2>/dev/null || need+=("openpyxl>=3.1")
$PY -c "import xgboost" 2>/dev/null  || need+=("xgboost==3.1.1")
$PY -c "import scipy.optimize as o; o.milp" 2>/dev/null || need+=("scipy==1.15.2")
if [ ${#need[@]} -gt 0 ]; then
  echo "[SETUP] installing: ${need[*]}"
  $PY -m pip install "${need[@]}"
else
  echo "[SETUP] all packages already installed"
fi
$PY run.py --check-only
