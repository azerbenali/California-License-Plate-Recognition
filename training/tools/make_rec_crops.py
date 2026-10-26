# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Cut rectified plate crops out of a detector dataset, for recognizer training.

The crops go through the same perspective rectification as the C++ pipeline,
so the recognizer trains on exactly what it will see in deployment.

    python -m training.tools.make_rec_crops \
        --images data/det/images/train --labels data/det/labels/train \
        --out data/rec/train [--texts plate_texts.csv]

--texts (optional) is a CSV with header  image,index,text  where `image` is the
image file name, `index` the 0-based line number of the plate in its label
file, and `text` the plate number. Without it, labels.csv is written with an
empty text column for you to fill in.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from ..common import config as C
from ..common.preprocess import normalize_text, order_corners, rectify

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_texts(path: str | None) -> dict[tuple[str, int], str]:
    texts: dict[tuple[str, int], str] = {}
    if not path:
        return texts
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts[(row["image"], int(row["index"]))] = normalize_text(row["text"])
    return texts


def parse_label_line(line: str, w: int, h: int, k: int = C.DET_NUM_KEYPOINTS):
    """YOLO pose line -> (corners in pixels, all corners labeled?)."""
    v = [float(t) for t in line.split()]
    if len(v) < 5 + 3 * k:
        return None, False
    kp = np.array(v[5:5 + 3 * k], dtype=np.float32).reshape(k, 3)
    corners = kp[:, :2] * np.array([w, h], dtype=np.float32)
    labeled = bool(np.all(kp[:, 2] > 0))
    if not labeled:  # fall back to the box
        cx, cy, bw, bh = v[1] * w, v[2] * h, v[3] * w, v[4] * h
        corners = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                            [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]], np.float32)
    return corners, labeled


def make_crops(images: str | Path, labels: str | Path, out: str | Path,
               texts: dict[tuple[str, int], str] | None = None, min_width: int = 40,
               only: set[str] | None = None) -> dict:
    """Write rectified crops + labels.csv for every labeled plate (of the image
    files named in `only`, if given). Returns counts."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    texts = texts or {}
    rows, n_box_fallback, n_small = [], 0, 0

    for img_path in sorted(Path(images).iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS or (only is not None and img_path.name not in only):
            continue
        label_path = Path(labels) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = [l for l in label_path.read_text().splitlines() if l.strip()]
        for idx, line in enumerate(lines):
            corners, labeled = parse_label_line(line, w, h)
            if corners is None:
                continue
            corners = order_corners(corners)
            if np.linalg.norm(corners[1] - corners[0]) < min_width:
                n_small += 1
                continue
            n_box_fallback += int(not labeled)
            crop = rectify(img, corners)
            name = f"{img_path.stem}_{idx}.png"
            cv2.imwrite(str(out / name), crop)
            rows.append((name, texts.get((img_path.name, idx), "")))

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename", "text"])
        wr.writerows(rows)
    return {"crops": len(rows), "box_fallback": n_box_fallback, "too_small": n_small,
            "missing_text": sum(1 for _, t in rows if not t)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--texts", default=None)
    ap.add_argument("--min-width", type=int, default=40,
                    help="skip plates narrower than this many pixels (unreadable)")
    args = ap.parse_args()

    n = make_crops(args.images, args.labels, args.out, read_texts(args.texts), args.min_width)
    print(f"wrote {n['crops']} crops to {args.out} "
          f"({n['box_fallback']} from boxes without corners, {n['too_small']} too small skipped)")
    if n["missing_text"]:
        print(f"{n['missing_text']} crop(s) have no text yet: fill them in "
              f"{Path(args.out) / 'labels.csv'}")


if __name__ == "__main__":
    main()
