#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Run by the controller build slot. No hardware or privileged commands.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$(realpath -m "${BUILD_OUT:?controller supplies BUILD_OUT}")
cd "$root"
sha256sum examples/camera-denoise/*.py examples/camera-denoise/*.c examples/camera-denoise/*.h examples/camera-denoise/aurora_camera_ane/*.py examples/camera-denoise/tests/*.* > "$out/source.log"
python3 -m venv "$out/bootstrap"
"$out/bootstrap/bin/pip" install --only-binary=:all: uv==0.9.5
export UV_PYTHON_INSTALL_DIR="$out/python" UV_CACHE_DIR="$out/uv-cache"
"$out/bootstrap/bin/uv" venv --python 3.12 --seed "$out/venv"
"$out/venv/bin/pip" install --only-binary=:all: -r examples/camera-denoise/requirements.txt
export PATH="$out/venv/bin:$PATH"
python -m pip install --only-binary=:all: cmake==4.1.2 ninja==1.13.0
python -m pip install --only-binary=:all: torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
git clone --quiet https://github.com/aurora-silicon/mil-hwx-compiler.git "$out/compiler"
git -C "$out/compiler" checkout --detach 45558fb76e08f4c12d9b7d3e333dbac7595f2e80
python examples/camera-denoise/train.py "$out/model" --steps 3000 > "$out/train.log" 2>&1
python examples/camera-denoise/export.py "$out/model" > "$out/export.log" 2>&1
cp "$out/model/"{weights.npz,weights.bin,model.mil,model.onnx,model.json,training.json} examples/camera-denoise/models/tiny-luma-v1/
make -C libane -j8 libane
make -C bindings/python/dylib libane_python
export PYTHONPATH="$root/examples/camera-denoise:$root/bindings/python/python"
result=0
python examples/camera-denoise/check_mil_reference.py "$out/model" "$out/compiler" --output "$out/mil-reference.json" > "$out/mil-reference.log" 2>&1 || result=1
python -m pytest -q examples/camera-denoise/tests > "$out/tests.log" 2>&1 || result=1
make -C examples/camera-denoise clean
make -C examples/camera-denoise PYTHON="$(command -v python)" PYTHON_DIR="$root/examples/camera-denoise" check > "$out/bridge.log" 2>&1 || result=1
python examples/camera-denoise/measure.py --output "$out/measurements.json" --frames 20 > "$out/measure.log" 2>&1 || result=1
export GNUSTEP_PREFIX="$out/gnustep" GNUSTEP_BUILD_ROOT="$out/gnustep-build" JOBS=8
if "$out/compiler/scripts/verify-linux-compiler.sh" build-only > "$out/compiler-build.log" 2>&1; then
    export LD_LIBRARY_PATH="$GNUSTEP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python examples/camera-denoise/probe_compiler.py "$out/model" "$out/compiler/build/mil-hwxc" > "$out/compiler-probes.log" 2>&1 || result=1
else
    result=1
fi
printf 'camera model validation rc=%s; individual evidence retained in out/\n' "$result"
exit "$result"
