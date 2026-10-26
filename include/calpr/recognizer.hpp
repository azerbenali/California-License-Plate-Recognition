// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Plate recognizer: LPRNet-style CTC model run with ncnn.
#pragma once

#include "calpr/image.hpp"
#include "calpr/plate_decoder.hpp"

#include <memory>
#include <string>

namespace ncnn { class Net; }

namespace calpr {

class PlateRecognizer {
public:
    struct Options {
        std::string param_path;
        std::string bin_path;
        std::string input_blob = "in0";   // pnnx export names
        std::string output_blob = "out0";
        bool crop_band = true;            // must match training (see config.hpp)
        int num_threads = 2;
    };

    PlateRecognizer();
    ~PlateRecognizer();
    PlateRecognizer(const PlateRecognizer&) = delete;
    PlateRecognizer& operator=(const PlateRecognizer&) = delete;

    bool load(const Options& options, std::string* error = nullptr);
    bool loaded() const { return loaded_; }

    // `plate_bgr` is a tightly cropped, upright plate (any size; 2:1 expected).
    // Returns false (and fills `error`) on inference failure.
    bool recognize(const ImageView& plate_bgr, PlateResult& result,
                   std::string* error = nullptr) const;

private:
    std::unique_ptr<ncnn::Net> net_;
    Options opt_;
    bool loaded_ = false;
};

}  // namespace calpr
