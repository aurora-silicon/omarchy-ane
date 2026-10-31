#!/bin/bash
# SPDX-License-Identifier: MIT
# Sourced only by controller build jobs; environment/cache belongs to output.
setup_python() {
    local output=$1
    python3 -m venv "$output/bootstrap"
    "$output/bootstrap/bin/pip" -q install --only-binary=:all: uv==0.9.5
    export UV_PYTHON_INSTALL_DIR="$output/python" UV_CACHE_DIR="$output/cache"
    "$output/bootstrap/bin/uv" venv --python 3.12 --seed "$output/venv"
    source "$output/venv/bin/activate"
}
