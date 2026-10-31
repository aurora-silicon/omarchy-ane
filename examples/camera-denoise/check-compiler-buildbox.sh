#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Re-export released weights and check compiler interoperability. No retraining.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "${BUILD_OUT:?controller supplies BUILD_OUT}")
cd "$root"
python3 -m venv "$out/bootstrap"
"$out/bootstrap/bin/pip" install --only-binary=:all: uv==0.9.5
export UV_PYTHON_INSTALL_DIR="$out/python" UV_CACHE_DIR="$out/uv-cache"
"$out/bootstrap/bin/uv" venv --python 3.12 --seed "$out/venv"
"$out/venv/bin/pip" install --only-binary=:all: -r examples/camera-denoise/requirements.txt cmake==4.1.2 ninja==1.13.0
export PATH="$out/venv/bin:$PATH"
git clone --quiet https://github.com/aurora-silicon/mil-hwx-compiler.git "$out/compiler"
git -C "$out/compiler" checkout --detach 45558fb76e08f4c12d9b7d3e333dbac7595f2e80
mkdir "$out/model"
cp examples/camera-denoise/models/tiny-luma-v1/{weights.npz,training.json} "$out/model/"
python examples/camera-denoise/export.py "$out/model" > "$out/export.log" 2>&1
# Unchanged CPU graph and weights make the earlier CPU timing/quality receipt
# applicable. Fail rather than silently carrying measurements across a change.
cmp "$out/model/model.onnx" examples/camera-denoise/models/tiny-luma-v1/model.onnx
cmp "$out/model/weights.bin" examples/camera-denoise/models/tiny-luma-v1/weights.bin
export PYTHONPATH="$root/examples/camera-denoise:$root/bindings/python/python"
python examples/camera-denoise/check_mil_reference.py "$out/model" "$out/compiler" --output "$out/mil-reference.json" > "$out/mil-reference.log" 2>&1
export GNUSTEP_PREFIX="$out/gnustep" GNUSTEP_BUILD_ROOT="$out/gnustep-build" JOBS=8
"$out/compiler/scripts/verify-linux-compiler.sh" build-only > "$out/compiler-build.log" 2>&1
export LD_LIBRARY_PATH="$GNUSTEP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python examples/camera-denoise/probe_compiler.py "$out/model" "$out/compiler/build/mil-hwxc" > "$out/compiler-probes.log" 2>&1
printf 'MIL ios18 reference and compiler probe capture: PASS\n'
