#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren

"""Time the Q6_K head matvec on the real tied head of a GGUF. CPU and memory only: no ANE, no GPU, no device node.

usage: q6k_head_bench.py MODEL.gguf [--gguf-py DIR] [--reps 20] [--threads 1,2,4,8]

It pins itself to the performance cores (tools/pcore_mask.py, OMARCHY_ANE_NO_PIN=1 turns that off), reads
token_embd.weight (Q6_K) as bytes, and for each thread count times `logits(hidden)` over --reps runs with a random fp32
hidden vector. It prints one JSON object: per thread count the median, best and worst time in ms, the bytes read, the
GB/s that implies, and the fp32-head baseline the Qwen decode pays today (21.65 ms) only as a reference value. It also
checks the first 256 logits against a float64 dot of the dequantized rows, so a fast wrong kernel cannot pass.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcore_mask  # noqa: E402

PINNED = pcore_mask.pin_self()

import numpy as np  # noqa: E402
from q6k_head import Q6KHead  # noqa: E402

BASELINE_FP32_HEAD_MS = 21.65     # the sealed M2 decode head, 2.03 GB read per step (reference value, not measured here)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--gguf-py", default="/var/tmp/qwen-decode/gguf-py")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--threads", default="1,2,4,8")
    args = parser.parse_args()
    sys.path.insert(0, args.gguf_py)
    from gguf import GGMLQuantizationType, GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(args.model, "r")
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    if tensor.tensor_type != GGMLQuantizationType.Q6_K:
        print(f"token_embd.weight is {tensor.tensor_type.name}, not Q6_K", file=sys.stderr)
        return 2
    k, rows = int(tensor.shape[0]), int(tensor.shape[1])
    blocks = np.asarray(tensor.data).reshape(rows, -1)
    hidden = (np.random.default_rng(1).standard_normal(k) * 2.0).astype(np.float32)
    check = dequantize(np.ascontiguousarray(blocks[:256]), GGMLQuantizationType.Q6_K).astype(np.float64) @ hidden.astype(np.float64)
    result = {"model_bytes": int(blocks.nbytes), "rows": rows, "k": k, "pinned": PINNED, "reps": args.reps,
              "baseline_fp32_head_ms": BASELINE_FP32_HEAD_MS, "threads": {}}
    for threads in (int(n) for n in args.threads.split(",")):
        head = Q6KHead(blocks, rows, k, threads=threads)
        head.logits(hidden)                                   # warm: page the tensor in
        times = []
        for _ in range(args.reps):
            start = time.perf_counter()
            logits = head.logits(hidden)
            times.append((time.perf_counter() - start) * 1e3)
        error = float(np.abs(logits[:256].astype(np.float64) - check).max())
        median = statistics.median(times)
        result["threads"][str(threads)] = {"median_ms": round(median, 3), "best_ms": round(min(times), 3), "worst_ms": round(max(times), 3),
                                           "gb_per_s": round(blocks.nbytes / (median / 1e3) / 1e9, 1), "max_abs_err_first256": error}
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
