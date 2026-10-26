// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// This file was modified by Azer Ben Ali in 2026: turned an ncnn demo into a
// reusable class; 37-class California charset; character-band cropping; error
// checks on model loading, blob names and output shape; support for both
// output orientations; decoding moved to plate_decoder.hpp.

#include "calpr/recognizer.hpp"

#include "net.h"  // ncnn

namespace calpr {

namespace {

void set_error(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

}  // namespace

PlateRecognizer::PlateRecognizer() : net_(new ncnn::Net) {}
PlateRecognizer::~PlateRecognizer() = default;

bool PlateRecognizer::load(const Options& options, std::string* error) {
    opt_ = options;
    loaded_ = false;
    net_->clear();
    net_->opt.use_vulkan_compute = false;
    net_->opt.num_threads = options.num_threads;
    if (net_->load_param(options.param_path.c_str()) != 0) {
        set_error(error, "Cannot load recognizer param file: " + options.param_path);
        return false;
    }
    if (net_->load_model(options.bin_path.c_str()) != 0) {
        set_error(error, "Cannot load recognizer weights file: " + options.bin_path);
        return false;
    }
    loaded_ = true;
    return true;
}

bool PlateRecognizer::recognize(const ImageView& plate_bgr, PlateResult& result,
                                std::string* error) const {
    if (!loaded_) { set_error(error, "Recognizer is not loaded"); return false; }
    if (plate_bgr.empty()) { set_error(error, "Empty plate image"); return false; }

    // BGR, character band only, 94x24, (x - 127.5) / 128 -- exactly as in training.
    const ImageView band = opt_.crop_band ? crop_rows(plate_bgr, kBandTop, kBandBottom)
                                          : plate_bgr;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(band.data, ncnn::Mat::PIXEL_BGR,
                                                 band.width, band.height, band.stride,
                                                 kRecInputW, kRecInputH);
    const float mean[3] = {kRecMean, kRecMean, kRecMean};
    const float norm[3] = {kRecScale, kRecScale, kRecScale};
    in.substract_mean_normalize(mean, norm);

    ncnn::Extractor ex = net_->create_extractor();
    ex.set_light_mode(true);
    if (ex.input(opt_.input_blob.c_str(), in) != 0) {
        set_error(error, "Recognizer input blob not found: " + opt_.input_blob);
        return false;
    }
    ncnn::Mat out;
    if (ex.extract(opt_.output_blob.c_str(), out) != 0) {
        set_error(error, "Recognizer output blob not found: " + opt_.output_blob);
        return false;
    }

    const float* data = nullptr;
    if (out.dims == 2) data = static_cast<const float*>(out.data);
    else if (out.dims == 3 && out.c == 1) data = out.channel(0);
    if (!data) {
        set_error(error, "Unsupported recognizer output (dims=" + std::to_string(out.dims) +
                             ", c=" + std::to_string(out.c) + ")");
        return false;
    }

    // LPRNet gives classes x time (37 x 18); some exports give time x classes.
    const int rows = out.h, cols = out.w;
    bool classes_on_rows;
    int T;
    if (rows == kNumClasses) { classes_on_rows = true; T = cols; }
    else if (cols == kNumClasses) { classes_on_rows = false; T = rows; }
    else {
        set_error(error,
                  "Recognizer output is " + std::to_string(rows) + "x" + std::to_string(cols) +
                      " but the California charset has " + std::to_string(kNumClasses) +
                      " classes. The model was trained with a different charset; "
                      "retrain it with training/recognition/train.py.");
        return false;
    }

    auto score = [&](int t, int k) -> float {
        return classes_on_rows ? data[k * cols + t] : data[t * cols + k];
    };
    result = interpret(ctc_greedy_decode(T, kNumClasses, score));
    return true;
}

}  // namespace calpr
