# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Plate-crop dataset for the recognizer.

A dataset folder holds upright, tightly cropped plate images, ideally made by
training/tools/make_rec_crops.py (same rectification as deployment). Labels
come from, in order of preference:

  1. labels.csv in the folder, with a header row:  filename,text
  2. the file name: the part before the first "_" is the plate text,
     e.g. 8KMZ245.jpg or 8KMZ245_0003.png
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.preprocess import encode, normalize_text, rec_input, rectify

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_samples(root: str | Path) -> list[tuple[Path, str]]:
    root = Path(root)
    samples: list[tuple[Path, str]] = []
    csv_path = root / "labels.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = normalize_text(row.get("text", "") or "")
                if text:
                    samples.append((root / row["filename"], text))
    else:
        for p in sorted(root.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                samples.append((p, normalize_text(p.stem.split("_")[0])))

    good, skipped = [], 0
    for path, text in samples:
        try:
            encode(text)
        except ValueError:
            skipped += 1
            continue
        if 1 <= len(text) <= 8 and path.exists():
            good.append((path, text))
        else:
            skipped += 1
    if skipped:
        print(f"[dataset] {root}: skipped {skipped} sample(s) with missing files or bad labels")
    return good


class Augment:
    """Photometric + small geometric jitter applied to the whole plate image.

    The geometric part re-rectifies the plate from slightly perturbed corners,
    which imitates imperfect corner keypoints from the detector.
    """

    def __init__(self, corner_jitter: float = 0.04):
        self.corner_jitter = corner_jitter

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if random.random() < 0.8:
            jx, jy = self.corner_jitter * w, self.corner_jitter * h
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
            corners += np.random.uniform([-jx, -jy], [jx, jy], size=(4, 2)).astype(np.float32)
            img = rectify(img, corners, w, h)
        if random.random() < 0.1:  # light-on-dark <-> dark-on-light plate designs
            img = 255 - img
        if random.random() < 0.2:  # night / IR cameras give gray images
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif random.random() < 0.3:  # white balance / tinted light
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.8, 1.2, 3), 0, 255).astype(np.uint8)
        if random.random() < 0.8:  # brightness / contrast
            alpha = random.uniform(0.6, 1.4)
            beta = random.uniform(-40, 40)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if random.random() < 0.3:  # blur (motion or defocus)
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.3:  # sensor noise
            noise = np.random.normal(0, random.uniform(3, 12), img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.3:  # low resolution, as from a distant plate
            s = random.uniform(0.3, 0.7)
            small = cv2.resize(img, (max(8, int(w * s)), max(4, int(h * s))))
            img = cv2.resize(small, (w, h))
        if random.random() < 0.3:  # JPEG artifacts
            q = random.randint(25, 80)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img


class PlateDataset(Dataset):
    def __init__(self, root: str | Path, augment: bool = False, use_band: bool = True):
        self.samples = load_samples(root)
        if not self.samples:
            raise RuntimeError(f"No labeled plate images found in {root}")
        self.augment = Augment() if augment else None
        self.use_band = use_band

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, text = self.samples[i]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR, like the C++ side
        if img is None:
            raise RuntimeError(f"Cannot read {path}")
        if self.augment is not None:
            img = self.augment(img)
        x = torch.from_numpy(rec_input(img, self.use_band))
        return x, torch.tensor(encode(text), dtype=torch.long), text


def collate(batch):
    images, labels, texts = zip(*batch)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return torch.stack(images), torch.cat(labels), lengths, list(texts)
