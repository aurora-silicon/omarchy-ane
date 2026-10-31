# SPDX-License-Identifier: MIT
"""Original, additive-skip, one-level UNet. No third-party model code/weights."""
import torch
from torch import nn
from torch.nn import functional as F


class TinyLuma(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(1, 8, 3, padding=1)
        self.down = nn.Conv2d(8, 8, 3, stride=2, padding=0)
        self.mid = nn.Conv2d(8, 8, 3, padding=1)
        self.dec = nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skip = F.relu(self.enc(x))
        # SAME_UPPER on even images matches MIL/ONNX. Keep leading pad zero
        # for stride 2, rather than PyTorch's symmetric padding convention.
        low = F.relu(self.mid(F.relu(self.down(F.pad(skip, (0, 1, 0, 1))))))
        up = F.interpolate(low, scale_factor=2, mode="nearest")
        return x - self.head(F.relu(self.dec(up + skip)))
