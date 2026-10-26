# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Python preprocessing and tools (no PyTorch needed).

Run from the repository root:  python -m pytest tests   (or: python tests/test_preprocess.py)
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.common import config as C  # noqa: E402
from training.common.preprocess import (  # noqa: E402
    crop_band, encode, greedy_decode, normalize_text, order_corners, rec_input, rectify,
)
from training.common.plate_format import interpret  # noqa: E402
from training.tools.make_rec_crops import parse_label_line  # noqa: E402
from training.tools.synth_plates import SCHEMES, random_plate_text, render_plate  # noqa: E402

# Same layouts as calpr::ca_formats() (R = letter except I, O, Q).
CA_PATTERNS = [
    r"^\d{3}[A-HJ-NPR-Z][A-Z][A-HJ-NPR-Z]\d$",   # 123ABC1
    r"^[1-9][A-HJ-NPR-Z][A-Z][A-HJ-NPR-Z]\d{3}$",  # 1ABC234
    r"^\d[A-HJ-NPR-Z]\d{5}$",                      # 1A23456
    r"^\d[A-HJ-NPR-Z]{2}\d{4}$",                   # 1AB1234
]


def test_band_matches_cpp_rounding():
    plate = np.zeros((C.PLATE_WARP_H, C.PLATE_WARP_W, 3), np.uint8)
    plate[np.arange(C.PLATE_WARP_H), :, 0] = np.arange(C.PLATE_WARP_H)[:, None]
    band = crop_band(plate)
    assert band.shape[0] == 105 - 26 and band[0, 0, 0] == 26  # same as tests/test_geometry.cpp


def test_rec_input_shape_and_range():
    x = rec_input(np.full((120, 240, 3), 255, np.uint8))
    assert x.shape == (3, C.REC_INPUT_H, C.REC_INPUT_W) and x.dtype == np.float32
    assert abs(float(x.max()) - (255 - 127.5) / 128) < 1e-6


def test_encode_decode_roundtrip():
    assert normalize_text("7abc-123") == "7ABC123"
    idx = encode("7ABC123")
    T = 18
    logits = np.zeros((C.NUM_CLASSES, T), np.float32)
    logits[C.BLANK_INDEX, :] = 1.0
    for t, k in zip(range(1, T, 2), idx):
        logits[k, t] = 5.0
    assert greedy_decode(logits) == "7ABC123"
    try:
        encode("ABC#1")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_order_corners_matches_cpp():
    o = order_corners([[90, 52], [10, 10], [12, 50], [88, 12]])
    assert [int(p[0]) for p in o] == [10, 88, 90, 12]  # same as tests/test_geometry.cpp


def test_rectify_identity_and_mirror():
    img = (np.arange(16 * 8 * 3) * 37 % 251).astype(np.uint8).reshape(8, 16, 3)
    same = rectify(img, [[0, 0], [15, 0], [15, 7], [0, 7]], 16, 8)
    assert np.array_equal(same, img)
    mirror = rectify(img, [[15, 0], [0, 0], [0, 7], [15, 7]], 16, 8)
    assert np.array_equal(mirror, img[:, ::-1])


def test_label_line_parsing():
    line = "0 0.5 0.5 0.2 0.1  0.41 0.46 2  0.59 0.45 2  0.60 0.55 2  0.40 0.54 2"
    corners, labeled = parse_label_line(line, 1000, 500)
    assert labeled and np.allclose(corners[0], [410, 230]) and np.allclose(corners[2], [600, 275])
    _, labeled = parse_label_line("0 0.5 0.5 0.2 0.1 0 0 0 0 0 0 0 0 0 0 0 0", 1000, 500)
    assert not labeled


def test_synthetic_texts_follow_ca_layouts():
    rng = random.Random(0)
    texts = [random_plate_text(rng) for _ in range(2000)]
    for t in texts:
        encode(t)  # only charset characters
        assert 2 <= len(t) <= 7
    # 92% are generated from the sequential layouts; the rest are vanity plates.
    matched = sum(any(re.match(p, t) for p in CA_PATTERNS) for t in texts)
    assert matched / len(texts) > 0.9, matched


def test_synthetic_plate_render():
    img = render_plate("8KMZ245", random.Random(0), scheme="standard")
    assert img.shape == (C.PLATE_WARP_H, C.PLATE_WARP_W, 3) and img.dtype == np.uint8
    band = crop_band(img)
    # The characters are dark and sit inside the band.
    assert band.min() < 120 and img[:5].mean() > 150


def test_every_plate_design_has_readable_characters():
    for name in SCHEMES:
        img = render_plate("7ABC123", random.Random(1), scheme=name)
        gray = cv2.cvtColor(crop_band(img), cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Characters and background differ clearly inside the band.
        assert np.percentile(gray, 95) - np.percentile(gray, 5) > 80, name


def test_plate_format_matches_cpp_decoder():
    # Same cases as tests/test_decoder.cpp.
    cases = [
        ("8KMZ245", "8KMZ245", True, 0, "1980"),
        ("123ABC1", "123ABC1", True, 0, "2026+"),
        ("7ABC123", "7ABC123", True, 0, "1980"),
        ("7A8C123", "7ABC123", True, 1, "1980"),
        ("7ABC1Z3", "7ABC123", True, 1, "1980"),
        ("123A0C1", "123AOC1", True, 1, "2026+"),
        ("7OBC123", "7DBC123", True, 1, "1980"),
        ("4X12345", "4X12345", True, 0, "Commercial"),
        ("HELLO", "HELLO", False, 0, "personalized"),
        ("KKKKKKK", "KKKKKKK", False, 0, "personalized"),
        ("201AAA1", "201AAA1", True, 0, "2026+"),
        ("PERC50", "PERC50", False, 0, "personalized"),
    ]
    for raw, text, match, edits, fmt in cases:
        r = interpret(raw, 0.9)
        assert (r.text, r.format_match, r.corrections) == (text, match, edits), (raw, r)
        assert fmt in r.format, (raw, r.format)
    assert interpret("", 0.9).confidence == 0.0


def test_sample_labels_are_valid():
    samples = ROOT / "samples"
    texts = {}
    for line in (samples / "plates.csv").read_text().splitlines()[1:]:
        image, index, text = line.split(",")
        texts[(image, int(index))] = text
    for img_path in sorted((samples / "images").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lines = (samples / "labels" / f"{img_path.stem}.txt").read_text().split("\n")
        lines = [l for l in lines if l.strip()]
        assert lines, img_path.name
        for idx, line in enumerate(lines):
            corners, labeled = parse_label_line(line, w, h)
            assert labeled, img_path.name
            o = order_corners(corners)
            assert np.allclose(o, corners), f"{img_path.name}: corners not in TL, TR, BR, BL order"
            assert normalize_text(texts[(img_path.name, idx)]) == texts[(img_path.name, idx)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok    {name}")
