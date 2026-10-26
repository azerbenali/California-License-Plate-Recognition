// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Plate detector: YOLO pose model (1 class, 4 corner keypoints) run with ncnn.
#pragma once

#include "calpr/config.hpp"
#include "calpr/image.hpp"
#include "calpr/yolo_pose.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ncnn { class Net; }

namespace calpr {

class PlateDetector {
public:
    struct Options {
        std::string param_path;
        std::string bin_path;
        std::string input_blob = "in0";   // YOLO ncnn export names
        std::string output_blob = "out0";
        int input_size = kDetInputSize;
        float conf_threshold = 0.4f;
        float iou_threshold = 0.5f;
        int num_threads = 2;
    };

    PlateDetector();
    ~PlateDetector();
    PlateDetector(const PlateDetector&) = delete;
    PlateDetector& operator=(const PlateDetector&) = delete;

    bool load(const Options& options, std::string* error = nullptr);
    bool loaded() const { return loaded_; }

    std::vector<PlateDetection> detect(const ImageView& bgr, std::string* error = nullptr) const;

private:
    std::unique_ptr<ncnn::Net> net_;
    Options opt_;
    bool loaded_ = false;
};

}  // namespace calpr
