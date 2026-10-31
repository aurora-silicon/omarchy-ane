#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out=$(realpath -m "${BUILD_OUT:-$root/build}")
python3 -m venv "$out/compiler-venv"
"$out/compiler-venv/bin/pip" install --only-binary=:all: cmake==4.1.2 ninja==1.13.0
export PATH="$out/compiler-venv/bin:$PATH"
export GNUSTEP_PREFIX="$out/gnustep" GNUSTEP_BUILD_ROOT="$out/gnustep-build" JOBS=8
git clone --quiet https://github.com/joshuaswarren/mil-hwx-compiler.git "$out/compiler"
git -C "$out/compiler" checkout --quiet --detach 927013f5480768b07c8e62a866e1514c41aba778
"$out/compiler/scripts/verify-linux-compiler.sh" build-only
export MIL_HWXC="$out/compiler/build/mil-hwxc"
export LD_LIBRARY_PATH="$GNUSTEP_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$root/scripts/check-fallback-bench.sh"
