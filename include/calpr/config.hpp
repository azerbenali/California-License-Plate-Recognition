// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Shared constants. Every value here has a twin in training/common/config.py,
// and tests/test_config_consistency.py fails if the two drift apart: the model
// only works if inference preprocesses images exactly like training did.
#pragma once

namespace calpr {

// --- Recognizer (LPRNet) ------------------------------------------------------
// Class order of the recognizer output. CTC blank is the LAST class.
inline constexpr const char kCharset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-";
inline constexpr int kNumClasses = 37;
inline constexpr int kBlankIndex = kNumClasses - 1;
static_assert(sizeof(kCharset) - 1 == kNumClasses, "charset size mismatch");

inline constexpr int kRecInputW = 94;
inline constexpr int kRecInputH = 24;
inline constexpr float kRecMean = 127.5f;
inline constexpr float kRecScale = 0.0078125f;  // 1/128 -> pixels in about [-1, 1]

// California plates are 12" x 6". The characters sit in a horizontal band;
// the "California" script is above it and the frame/slogan below it. The
// recognizer only sees this band (fractions of the rectified plate height).
inline constexpr float kBandTop = 0.22f;
inline constexpr float kBandBottom = 0.88f;

// Size of the rectified (perspective-corrected) plate image, 2:1 like the plate.
inline constexpr int kPlateWarpW = 240;
inline constexpr int kPlateWarpH = 120;

// --- Detector (YOLO pose, 4 keypoints = plate corners) -------------------------
inline constexpr int kDetInputSize = 640;
inline constexpr int kDetNumKeypoints = 4;  // TL, TR, BR, BL
inline constexpr float kDetPadValue = 114.f;

}  // namespace calpr
