#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent compiler fp16 MIL reference versus ONNX tile outputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from aurora_camera_ane.backend import FrameCPU, verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("compiler", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    verify(a.model)
    sys.path.insert(0, str(a.compiler/"tools"))
    from h13_reference import evaluate
    cpu = FrameCPU(a.model)
    rng = np.random.default_rng(3652)
    inputs = [rng.uniform(0, 1, (1, 1, 64, 64)).astype(np.float16),
              np.linspace(0, 1, 64*64, dtype=np.float16).reshape(1,1,64,64)]
    results = []
    for value in inputs:
        expected = evaluate((a.model/"model.mil").read_text(), a.model, {"x": value.astype("<f2").tobytes()})
        expected = np.frombuffer(expected["denoised"], "<f2").reshape(value.shape)
        actual = cpu.tensor(value)
        error = float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32))))
        np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=.002, rtol=.002)
        results.append({"max_abs_error": error})
    report = {"reference": "compiler h13_reference.evaluate", "cases": results,
              "atol": .002, "rtol": .002}
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
