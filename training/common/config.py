# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Shared constants.

Every value here has a twin in include/calpr/config.hpp, and
tests/test_config_consistency.py fails if the two drift apart.
"""

# Recognizer (LPRNet). CTC blank is the LAST class.
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
NUM_CLASSES = 37
BLANK_INDEX = NUM_CLASSES - 1

REC_INPUT_W = 94
REC_INPUT_H = 24
REC_MEAN = 127.5
REC_SCALE = 0.0078125  # 1/128

# Character band of a rectified California plate (fractions of its height).
BAND_TOP = 0.22
BAND_BOTTOM = 0.88

# Rectified plate size (2:1, like the 12" x 6" plate).
PLATE_WARP_W = 240
PLATE_WARP_H = 120

# Detector (YOLO pose, 4 keypoints = TL, TR, BR, BL plate corners).
DET_INPUT_SIZE = 640
DET_NUM_KEYPOINTS = 4
DET_PAD_VALUE = 114.0

assert len(CHARSET) == NUM_CLASSES
