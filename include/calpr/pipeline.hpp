// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Full pipeline: detect plates -> rectify each one -> recognize characters.
#pragma once

#include "calpr/detector.hpp"
#include "calpr/recognizer.hpp"

#include <string>
#include <vector>

namespace calpr {

struct PlateRead {
    PlateDetection detection;
    Quad corners_used{};       // corners actually used for rectification
    bool used_keypoints = false;
    Image rectified;           // kPlateWarpW x kPlateWarpH upright plate
    PlateResult result;
};

struct PipelineOptions {
    // Keypoints below this visibility fall back to the axis-aligned box.
    float min_corner_conf = 0.5f;
};

class PlatePipeline {
public:
    using Options = PipelineOptions;

    PlatePipeline(const PlateDetector& detector, const PlateRecognizer& recognizer,
                  Options options = Options())
        : det_(detector), rec_(recognizer), opt_(options) {}

    std::vector<PlateRead> run(const ImageView& bgr, std::string* error = nullptr) const;

    // Rectify one detection (exposed for tests and for saving training crops).
    PlateRead rectify(const ImageView& bgr, const PlateDetection& det) const;

private:
    const PlateDetector& det_;
    const PlateRecognizer& rec_;
    Options opt_;
};

}  // namespace calpr
