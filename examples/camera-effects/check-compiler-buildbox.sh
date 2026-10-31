#!/bin/bash
# SPDX-License-Identifier: MIT
set -euo pipefail
out=$(realpath -m "${BUILD_OUT:?}")
root=$(cd "$(dirname "$0")/../.." && pwd)
source "$root/examples/camera-effects/setup-python.sh"
setup_python "$out"
pip -q install cmake==4.1.2 ninja==1.13.0
git clone --quiet https://github.com/aurora-silicon/mil-hwx-compiler.git "$out/compiler"
git -C "$out/compiler" checkout --detach 8f8fb25f58e05a27db4c503e8eea7d4f13be4219
export GNUSTEP_PREFIX="$out/gnustep" GNUSTEP_BUILD_ROOT="$out/gnustep-build" JOBS=8
"$out/compiler/scripts/verify-linux-compiler.sh" build-only > "$out/compiler-build.log" 2>&1
export LD_LIBRARY_PATH="$GNUSTEP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python "$root/examples/camera-effects/probe_compiler.py" "$root/examples/camera-effects/models" "$out/compiler/build/mil-hwxc" --output "$out/compiler-gaps.json" > "$out/compiler-probes.log" 2>&1
