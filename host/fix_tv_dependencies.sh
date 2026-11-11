#!/usr/bin/env bash
# Run after: conda activate tv
# Optional: bash fix_tv_dependencies.sh --dry-run
# Torch/vision pairing: https://pytorch.org/get-started/previous-versions/
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
    echo "Usage: bash fix_tv_dependencies.sh [--dry-run]" >&2
    exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "tv" || -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the target environment first: conda activate tv" >&2
    exit 1
fi

tv_python="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$tv_python" ]]; then
    echo "Python not found: $tv_python" >&2
    exit 1
fi

# Prevent user-site packages from shadowing this environment. These settings
# apply only to this script and its children, not to the calling shell.
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

"$tv_python" -s - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit('This script targets the existing Python 3.10 tv environment.')
print('Target Python:', sys.executable)
PY

packages=(
    'numpy==1.26.4'
    'torch==2.3.0'
    'matplotlib==3.7.5'
    'dex-retargeting==0.4.7'
    'teleimager==1.6.0'
    'televuer==0.0.3'
    # Keep OpenCV on the installed release that accepts NumPy 1.x.
    'opencv-python==4.11.0.86'
)

# Align installed PyTorch companion packages without adding unused ones.
for entry in 'torchvision==0.18.0' 'torchaudio==2.3.0' 'torchtext==0.18.0'; do
    if "$tv_python" -s - "${entry%%==*}" <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    version(sys.argv[1])
except PackageNotFoundError:
    sys.exit(1)
PY
    then
        packages+=("$entry")
    fi
done

pip_options=()
if [[ "${1:-}" == "--dry-run" ]]; then
    pip_options+=(--dry-run)
fi

# Resolve all pins together, including their dependencies. Do not use --no-deps.
# --isolated ignores user pip settings and PIP_* environment overrides.
"$tv_python" -s -m pip --isolated install \
    "${pip_options[@]}" "${packages[@]}"

if [[ "${1:-}" == "--dry-run" ]]; then
    exit 0
fi

"$tv_python" -s -m pip --isolated check
"$tv_python" -s - <<'PY'
import numpy
import matplotlib
import matplotlib._path
import torch
print('NumPy:', numpy.__version__)
print('Matplotlib:', matplotlib.__version__)
print('PyTorch:', torch.__version__)
print('PyTorch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
PY

echo 'Dependency and import checks passed.'
echo 'From the host directory, launch with: PYTHONNOUSERSITE=1 python main.py'
