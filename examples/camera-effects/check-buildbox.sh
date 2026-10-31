#!/bin/bash
# SPDX-License-Identifier: MIT
set -euo pipefail
out=$(realpath -m "${BUILD_OUT:?}")
root=$(cd "$(dirname "$0")/../.." && pwd)
source "$root/examples/camera-effects/setup-python.sh"
setup_python "$out"
(cd "$root" && sha256sum examples/camera-effects/*.py examples/camera-effects/*.c examples/camera-effects/*.h examples/camera-effects/aurora_camera_effects/*.py examples/camera-effects/tests/*.[cp]* examples/camera-effects/models/*/model.json) > "$out/source.sha256"
pip -q install --only-binary=:all: -r "$root/examples/camera-effects/requirements-test.txt"
result=0
if make -C "$root/libane" -j8 libane > "$out/libane.log" 2>&1 &&
   make -C "$root/bindings/python/dylib" libane_python >> "$out/libane.log" 2>&1; then
    printf 'Shared runtime build passed\n'
else
    result=1;tail -15 "$out/libane.log"
fi
export PYTHONPATH="$root/examples/camera-effects:$root/bindings/python/python"
timeout 300 python -m pytest -q "$root/examples/camera-effects/tests" > "$out/tests.log" 2>&1 || result=1
tail -15 "$out/tests.log"
if make -C "$root/examples/camera-effects" PYTHON="$(command -v python)" PYTHON_DIR="$root/examples/camera-effects" > "$out/bridge.log" 2>&1 &&
   cc -Wall -Wextra -Werror -O2 "$root/examples/camera-effects/tests/test_bridge.c" -ldl -o "$out/test-bridge" >> "$out/bridge.log" 2>&1; then
    if timeout 180 "$out/test-bridge" "$root/examples/camera-effects/libaurora-camera-effects.so" >> "$out/bridge.log" 2>&1; then
        printf 'RTLD_LOCAL dynamic bridge passed\n'
    else
        result=1;tail -15 "$out/bridge.log"
    fi
else
    result=1;tail -15 "$out/bridge.log"
fi
timeout 900 python "$root/examples/camera-effects/measure.py" --output "$out/measurements.json" --work "$out/dataset" --frames 30 > "$out/measure.log" 2>&1 || result=1
timeout 180 python "$root/examples/camera-effects/check_source.py" --output "$out/source-reference.json" --images "$out/dataset" > "$out/source-reference.log" 2>&1 || result=1
if test ! -f "$out/measurements.json"; then tail -15 "$out/measure.log"; fi
if test ! -f "$out/source-reference.json"; then tail -15 "$out/source-reference.log"; fi
python "$root/examples/camera-effects/summarize.py" "$out"
printf 'Runtime validation exit=%s\n' "$result"
exit "$result"
