# SPDX-License-Identifier: MIT
"""Small, explicitly licensed open corpus and deterministic AWGN test set."""
import hashlib
import numpy as np
from skimage import data

TRAIN = ("astronaut", "coffee")
TEST = ("chelsea", "hubble_deep_field")
LICENSES = {"astronaut": "public-domain", "coffee": "CC0-1.0",
            "chelsea": "CC0-1.0", "hubble_deep_field": "public-domain"}


def image(name):
    if name not in TRAIN + TEST:
        raise ValueError("image has not been licence audited")
    rgb = getattr(data, name)()
    # Full-range BT.709 luma, not skimage's slightly different RGB luminance.
    return (rgb.astype(np.float32) @ np.array([.2126, .7152, .0722], np.float32)) / 255


def identity(name):
    rgb = getattr(data, name)()
    return {"name": name, "license": LICENSES[name], "shape": list(rgb.shape),
            "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}


def noisy(clean, sigma, seed):
    return np.clip(clean + np.random.default_rng(seed).normal(
        0, sigma / 255, clean.shape).astype(np.float32), 0, 1)
