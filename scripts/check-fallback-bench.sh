#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${BUILD_OUT:-$root/build}")
# Older ONNX wheels need Python <=3.13, even on Python-3.14 build hosts.
python3 -m venv "$out/bootstrap"
"$out/bootstrap/bin/pip" install --only-binary=:all: uv==0.9.5
export UV_PYTHON_INSTALL_DIR="$out/python" UV_CACHE_DIR="$out/uv-cache"
"$out/bootstrap/bin/uv" venv --python 3.12 --seed "$out/venv"
"$out/venv/bin/pip" install --only-binary=:all: -r "$root/tools/requirements-bench.txt"
export PATH="$out/venv/bin:$PATH"
make -C "$root/libane" -j8 libane
make -C "$root/bindings/python/dylib" libane_python
make -C "$root/tools" check-fallback test_libane_ioctl
"$root/tools/test_libane_ioctl" "$root/fixtures/h14-anec"
python3 -m unittest discover -s "$root/tools" -p 'test_fallback.py' -v
python3 -m unittest discover -s "$root/tools" -p 'test_mobilenet_baseline.py' -v
for batch in 1 8; do
    compiler_options=()
    if [[ -n "${MIL_HWXC:-}" ]]; then compiler_options=(--compiler "$MIL_HWXC"); fi
    python3 "$root/tools/mobilenet_baseline.py" "$out/mnv1-b$batch" --batch "$batch" "${compiler_options[@]}"
    "$root/tools/ane-bench" "$out/mnv1-b$batch" .2 > "$out/bench-b$batch.json"
    cat "$out/bench-b$batch.json"
done
