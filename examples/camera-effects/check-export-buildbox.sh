#!/bin/bash
# SPDX-License-Identifier: MIT
set -euo pipefail
out=$(realpath -m "${BUILD_OUT:?}")
root=$(cd "$(dirname "$0")/../.." && pwd)
source "$root/examples/camera-effects/setup-python.sh"
setup_python "$out"
# Install CPU wheels first so the converter cannot select CUDA dependencies.
pip -q install --only-binary=:all: torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip -q install --only-binary=:all: -r "$root/examples/camera-effects/requirements-export.txt"
python "$root/examples/camera-effects/export.py" "$out/export" > "$out/export.log" 2>&1
