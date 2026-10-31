#!/bin/bash
# SPDX-License-Identifier: MIT
# Install into a user-owned checkout; never changes a camera or system setting.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv-h13
else
    # Fedora's default Python can be newer than the pinned ORT/NumPy wheels.
    # A managed interpreter stays inside this user-owned checkout's tooling.
    python3 -m venv .venv-h13-bootstrap
    .venv-h13-bootstrap/bin/python -m pip install --only-binary=:all: uv==0.9.5
    .venv-h13-bootstrap/bin/uv venv --python 3.12 --seed .venv-h13
fi
.venv-h13/bin/python -m pip install --only-binary=:all: -r requirements-h13.txt
.venv-h13/bin/python fetch-models.py
if [ "$#" -gt 0 ]; then
    compiler=$(realpath "$1")
    for model in selfie yunet; do
        dest="models/$model/compiled/H13/hybrid"
        if [ -e "$dest" ]; then
            printf 'Preserving existing package: %s (move it aside to rebuild)\n' "$dest"
        else
            .venv-h13/bin/python h13/build.py "models/$model" "$compiler" --output "$dest"
        fi
    done
fi
printf 'Ready: .venv-h13/bin/python try-camera-h13 preview --input portrait.jpg --engine auto --framing\n'
