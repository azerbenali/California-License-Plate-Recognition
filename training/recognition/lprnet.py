# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""LPRNet-style recognizer (after Zherzdev & Gruzdev, 2018, "LPRNet: License
Plate Recognition via Deep Neural Networks").

Input  : (B, 3, 24, 94) BGR, normalized to about [-1, 1]
Output : (B, num_classes, 18) raw logits, one column per CTC time step

Differences from the common PyTorch port: channel reduction uses 1x1
convolutions instead of MaxPool3d, so every layer maps cleanly to ncnn via
pnnx; and the global-context pooling sizes are fixed for the 24x94 input.

Shape walk-through (H x W):
  stem conv3x3           24x94 -> 22x92      (context 1, 64 ch)
  maxpool 3, s1          22x92 -> 20x90
  block 64->128          20x90               (context 2, 128 ch)
  maxpool 3, s(1,2)      20x90 -> 18x44
  block 128->256, 256    18x44               (context 3, 256 ch)
  maxpool 3, s(1,2)      18x44 -> 16x21
  conv 1x1 256->64       16x21
  conv 1x4 64->256       16x21 -> 16x18
  conv 13x1 256->C       16x18 -> 4x18       (context 4, C ch)
  global context concat  4x18, 448+C ch -> 1x1 conv -> C
  mean over H            -> C x 18
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(cin: int, cout: int, k, s=1, p=0) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallBasicBlock(nn.Module):
    """Bottleneck with separable 3x1 / 1x3 convolutions (LPRNet paper, Table 1)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        mid = cout // 4
        self.body = nn.Sequential(
            nn.Conv2d(cin, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, (3, 1), padding=(1, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, (1, 3), padding=(0, 1)), nn.ReLU(inplace=True),
            nn.Conv2d(mid, cout, 1),
        )
        self.post = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.post(self.body(x))


class LPRNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.stem = conv_bn_relu(3, 64, 3)
        self.pool1 = nn.MaxPool2d(3, stride=1)
        self.block1 = SmallBasicBlock(64, 128)
        self.pool2 = nn.MaxPool2d(3, stride=(1, 2))
        self.block2 = nn.Sequential(SmallBasicBlock(128, 256), SmallBasicBlock(256, 256))
        self.pool3 = nn.MaxPool2d(3, stride=(1, 2))
        self.head = nn.Sequential(
            conv_bn_relu(256, 64, 1),
            nn.Dropout(dropout),
            conv_bn_relu(64, 256, (1, 4)),
            nn.Dropout(dropout),
            conv_bn_relu(256, num_classes, (13, 1)),
        )
        # Pool every context map down to the 4x18 grid of the head output.
        self.ctx_pool1 = nn.AvgPool2d(5, stride=5)                       # 22x92 -> 4x18
        self.ctx_pool2 = nn.AvgPool2d(5, stride=5)                       # 20x90 -> 4x18
        self.ctx_pool3 = nn.AvgPool2d((4, 10), stride=(4, 2))            # 18x44 -> 4x18
        self.container = nn.Conv2d(64 + 128 + 256 + num_classes, num_classes, 1)

    @staticmethod
    def _norm(f: torch.Tensor) -> torch.Tensor:
        # Scale each context map (per sample) to a comparable magnitude before
        # concatenation.
        return f / (f.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.stem(x)
        c2 = self.block1(self.pool1(c1))
        c3 = self.block2(self.pool2(c2))
        c4 = self.head(self.pool3(c3))
        ctx = torch.cat([self._norm(self.ctx_pool1(c1)), self._norm(self.ctx_pool2(c2)),
                         self._norm(self.ctx_pool3(c3)), self._norm(c4)], dim=1)
        logits = self.container(ctx)       # (B, C, 4, 18)
        return logits.mean(dim=2)           # (B, C, 18)


def ctc_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """(B, C, T) logits -> (T, B, C) log-probabilities for nn.CTCLoss."""
    return F.log_softmax(logits.permute(2, 0, 1), dim=2)
