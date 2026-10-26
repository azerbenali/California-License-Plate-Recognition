# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Image preprocessing shared by training and the dataset tools.

These functions mirror the C++ library (include/calpr/image.hpp,
src/recognizer.cpp) so the model sees the same pixels in training and in
deployment.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points as TL, TR, BR, BL (same rule as calpr::order_corners)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 1] - pts[:, 0]
    idx = [int(np.argmin(s)), int(np.argmin(d)), int(np.argmax(s)), int(np.argmax(d))]
    if len(set(idx)) != 4:
        return pts
    return pts[idx]


def rectify(img_bgr: np.ndarray, corners: np.ndarray,
            out_w: int = C.PLATE_WARP_W, out_h: int = C.PLATE_WARP_H) -> np.ndarray:
    """Warp the plate quadrilateral (TL, TR, BR, BL) to an upright out_w x out_h image.

    Same mapping as calpr::warp_perspective.
    """
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_band(plate_bgr: np.ndarray, top: float = C.BAND_TOP,
              bottom: float = C.BAND_BOTTOM) -> np.ndarray:
    """Rows [top*h, bottom*h) -- same integer rounding as calpr::crop_rows."""
    h = plate_bgr.shape[0]
    y0 = min(max(int(h * top), 0), h - 1)
    y1 = max(y0 + 1, min(int(h * bottom), h))
    return plate_bgr[y0:y1]


def rec_input(plate_bgr: np.ndarray, use_band: bool = True) -> np.ndarray:
    """Plate image (BGR uint8) -> recognizer input, float32 CHW, about [-1, 1]."""
    img = crop_band(plate_bgr) if use_band else plate_bgr
    img = cv2.resize(img, (C.REC_INPUT_W, C.REC_INPUT_H), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) - C.REC_MEAN) * C.REC_SCALE
    return np.ascontiguousarray(img.transpose(2, 0, 1))


CHAR_TO_INDEX = {c: i for i, c in enumerate(C.CHARSET[:-1])}  # blank excluded


def normalize_text(text: str) -> str:
    """Uppercase and drop spaces/dashes, which are not part of the plate number."""
    return "".join(ch for ch in text.upper() if ch not in " -_")


def encode(text: str) -> list[int]:
    """Plate text -> class indices. Raises ValueError on characters outside the charset."""
    text = normalize_text(text)
    bad = [ch for ch in text if ch not in CHAR_TO_INDEX]
    if bad:
        raise ValueError(f"unsupported characters {bad!r} in label {text!r}")
    return [CHAR_TO_INDEX[ch] for ch in text]


def greedy_decode(logits: np.ndarray) -> str:
    """logits: (num_classes, T). Standard CTC greedy decoding."""
    best = np.argmax(logits, axis=0)
    out, prev = [], C.BLANK_INDEX
    for k in best:
        k = int(k)
        if k != C.BLANK_INDEX and k != prev:
            out.append(C.CHARSET[k])
        prev = k
    return "".join(out)
