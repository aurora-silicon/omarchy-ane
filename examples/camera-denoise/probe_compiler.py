#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe an already-exported model without blocking independent CPU checks."""
import argparse
from pathlib import Path
from aurora_camera_ane.backend import verify
from export import compile_and_record

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("model", type=Path)
p.add_argument("compiler", type=Path)
a = p.parse_args()
root = a.model.resolve()
compile_and_record(root, a.compiler.resolve(), verify(root))
