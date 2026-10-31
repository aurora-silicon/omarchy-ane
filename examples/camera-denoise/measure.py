#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Quality on the held-out open AWGN set and CPU wall time at camera sizes."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from aurora_camera_ane.backend import Backend, MODEL, LIBANE
from aurora_camera_ane.data import TEST, image, identity, noisy


def summary(samples):
    return {"frames": len(samples), "p50_ms": float(np.median(samples)),
            "p90_ms": float(np.percentile(samples, 90)), "min_ms": min(samples),
            "samples_ms": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--libane", type=Path, default=LIBANE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=20)
    a = p.parse_args()
    if a.frames < 1:
        p.error("frames must be positive")
    report = {"model": "tiny-luma-unet-v1", "backend": "cpu", "machine": platform.machine(),
              "kernel": platform.release(), "python": platform.python_version(),
              "cpu_info": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                if line.startswith("model name")), platform.processor()),
              "warmup": 5, "threads": 8, "quality": [], "timings": [],
              "logical_cpus": os.cpu_count(), "load_average_at_start": list(os.getloadavg()),
              "metrics": {"data_range": 1, "ssim": "skimage default 7x7 window, sample covariance"},
              "dataset": "open-luma-awgn-v1", "split": [identity(n) for n in TEST],
              "noise": "clipped AWGN followed by uint8 quantization; seeds 365200+index*100+sigma",
              "scope": "luma only, full-range BT.709; synthetic noise, no real-sensor or temporal claim"}
    for index, name in enumerate(TEST):
        clean = image(name)
        backend = Backend(clean.shape[1], clean.shape[0], model_root=a.model, library=a.libane,
                          native_factory=lambda: None)
        try:
            output = np.empty(clean.shape, np.uint8)
            for sigma in (15, 25, 40):
                source = np.rint(noisy(clean, sigma, 365200+index*100+sigma)*255).astype(np.uint8)
                backend.process(source, None, 1., output)
                before, after = source.astype(np.float32)/255, output.astype(np.float32)/255
                report["quality"].append({"image": name, "sigma": sigma,
                    "noisy_psnr_db": float(peak_signal_noise_ratio(clean, before, data_range=1)),
                    "model_psnr_db": float(peak_signal_noise_ratio(clean, after, data_range=1)),
                    "noisy_ssim": float(structural_similarity(clean, before, data_range=1)),
                    "model_ssim": float(structural_similarity(clean, after, data_range=1))})
        finally:
            backend.fini()
    for width, height in ((1280, 720), (1920, 1080)):
        clean = resize(image("chelsea"), (height, width), anti_aliasing=True).astype(np.float32)
        source = np.rint(noisy(clean, 25, 3652)*255).astype(np.uint8)
        output = np.empty_like(source)
        backend = Backend(width, height, model_root=a.model, library=a.libane, native_factory=lambda: None)
        model_samples, total_samples = [], []
        try:
            for iteration in range(5+a.frames):
                start = time.perf_counter_ns()
                call = backend.process(source, None, 1., output)
                elapsed = (time.perf_counter_ns()-start)/1e6
                if iteration >= 5:
                    assert call.backend == 0  # ANE_INFER_CPU
                    model_samples.append(call.latency_ns/1e6)
                    total_samples.append(elapsed)
            report["timings"].append({"width": width, "height": height,
                "model": summary(model_samples), "plane_process": summary(total_samples),
                "model_scope": "libane fallback callback, edge pad + full-frame ORT + output crop",
                "plane_process_scope": "uint8 normalize + model + strength blend + quantize; Python API",
                "weights_sha256": backend.metadata["artifacts"]["weights.npz"]})
        finally:
            backend.fini()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
