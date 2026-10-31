#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Train only on the audited train split, never on quality-test images."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import torch
from aurora_camera_ane.network import TinyLuma
from aurora_camera_ane.data import TRAIN, image, identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--steps", type=int, default=3000)
    a = p.parse_args()
    if a.steps < 1 or a.output.exists():
        p.error("steps must be positive and output must be absent")
    torch.set_num_threads(8)
    torch.manual_seed(3652)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(3652)
    images = [image(n) for n in TRAIN]
    model = TinyLuma().train()
    optim = torch.optim.Adam(model.parameters(), lr=.001)
    losses = []
    started = time.monotonic()
    for step in range(a.steps):
        patches = []
        for _ in range(16):
            source = images[rng.integers(len(images))]
            y, x = [rng.integers(0, n - 64 + 1) for n in source.shape]
            patch = source[y:y+64, x:x+64]
            patch = np.rot90(patch, rng.integers(4))
            if rng.integers(2):
                patch = patch[:, ::-1]
            patches.append(patch.copy())
        clean = torch.from_numpy(np.stack(patches)[:, None])
        sigma = torch.from_numpy(rng.uniform(5, 40, (16, 1, 1, 1)).astype(np.float32)/255)
        noisy = torch.clamp(clean + torch.randn_like(clean)*sigma, 0, 1)
        prediction = model(noisy)
        # The halo is discarded at inference as well. Train the useful center.
        loss = (prediction[..., 12:-12, 12:-12] - clean[..., 12:-12, 12:-12]).square().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step in (1500, 2500):
            for group in optim.param_groups:
                group["lr"] *= .3
        if step % 100 == 0 or step == a.steps-1:
            record = {"step": step+1, "mse": float(loss.detach())}
            losses.append(record)
            print(json.dumps(record), flush=True)
    a.output.mkdir(parents=True)
    # The one released weight source. Both exporters consume these fp16 arrays.
    weights = {n: v.detach().numpy().astype(np.float16) for n, v in model.state_dict().items()}
    np.savez(a.output/"weights.npz", **weights)
    receipt = {"model": "tiny-luma-unet-v1", "license": "MIT", "seed": 3652,
               "steps": a.steps, "batch": 16, "crop": 64, "loss_crop": 40,
               "noise": "clipped AWGN, sigma uniform [5,40]/255",
               "train": [identity(n) for n in TRAIN], "test_used_in_training": False,
               "torch": torch.__version__, "numpy": np.__version__,
               "threads": 8, "machine": platform.machine(),
               "implementation_sha256": {
                   "train.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "network.py": hashlib.sha256((Path(__file__).parent/"aurora_camera_ane/network.py").read_bytes()).hexdigest()},
               "training_seconds": time.monotonic()-started, "losses": losses,
               "parameters": sum(v.size for v in weights.values()),
               "weight_sha256": hashlib.sha256((a.output/"weights.npz").read_bytes()).hexdigest()}
    (a.output/"training.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
