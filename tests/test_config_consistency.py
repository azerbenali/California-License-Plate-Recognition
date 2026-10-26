# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""C++ and Python must preprocess identically: compare config.hpp with config.py.

Run from the repository root:  python -m pytest tests   (or: python tests/test_config_consistency.py)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.common import config as C  # noqa: E402

PAIRS = {
    "kCharset": "CHARSET",
    "kNumClasses": "NUM_CLASSES",
    "kRecInputW": "REC_INPUT_W",
    "kRecInputH": "REC_INPUT_H",
    "kRecMean": "REC_MEAN",
    "kRecScale": "REC_SCALE",
    "kBandTop": "BAND_TOP",
    "kBandBottom": "BAND_BOTTOM",
    "kPlateWarpW": "PLATE_WARP_W",
    "kPlateWarpH": "PLATE_WARP_H",
    "kDetInputSize": "DET_INPUT_SIZE",
    "kDetNumKeypoints": "DET_NUM_KEYPOINTS",
    "kDetPadValue": "DET_PAD_VALUE",
}


def read_cpp_constants() -> dict:
    text = (ROOT / "include" / "calpr" / "config.hpp").read_text()
    values = {}
    for m in re.finditer(r"inline constexpr [\w ]+?\s+(k\w+)(?:\[\])?\s*=\s*([^;]+);", text):
        name, raw = m.group(1), m.group(2).strip()
        if raw.startswith('"'):
            values[name] = raw.strip('"')
        else:
            try:
                values[name] = float(raw.rstrip("f"))
            except ValueError:
                pass  # derived expressions such as kNumClasses - 1
    return values


def test_constants_match():
    cpp = read_cpp_constants()
    for cpp_name, py_name in PAIRS.items():
        assert cpp_name in cpp, f"{cpp_name} not found in config.hpp"
        py_val = getattr(C, py_name)
        if isinstance(py_val, str):
            assert cpp[cpp_name] == py_val, f"{cpp_name} != {py_name}"
        else:
            assert abs(cpp[cpp_name] - float(py_val)) < 1e-9, f"{cpp_name}={cpp[cpp_name]} but {py_name}={py_val}"


if __name__ == "__main__":
    test_constants_match()
    print("ok    config.hpp and config.py agree")
