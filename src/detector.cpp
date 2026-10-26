// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0

#include "calpr/detector.hpp"

#include "net.h"  // ncnn

namespace calpr {

namespace {

void set_error(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

// Extract a contiguous 2D float matrix from an ncnn output blob.
bool as_matrix(const ncnn::Mat& m, const float*& data, int& rows, int& cols) {
    if (m.dims == 2) {
        data = static_cast<const float*>(m.data);
    } else if (m.dims == 3 && m.c == 1) {
        data = m.channel(0);
    } else {
        return false;
    }
    rows = m.h;
    cols = m.w;
    return data != nullptr;
}

}  // namespace

PlateDetector::PlateDetector() : net_(new ncnn::Net) {}
PlateDetector::~PlateDetector() = default;

bool PlateDetector::load(const Options& options, std::string* error) {
    opt_ = options;
    loaded_ = false;
    net_->clear();
    net_->opt.use_vulkan_compute = false;
    net_->opt.num_threads = options.num_threads;
    if (net_->load_param(options.param_path.c_str()) != 0) {
        set_error(error, "Cannot load detector param file: " + options.param_path);
        return false;
    }
    if (net_->load_model(options.bin_path.c_str()) != 0) {
        set_error(error, "Cannot load detector weights file: " + options.bin_path);
        return false;
    }
    loaded_ = true;
    return true;
}

std::vector<PlateDetection> PlateDetector::detect(const ImageView& bgr, std::string* error) const {
    if (!loaded_) { set_error(error, "Detector is not loaded"); return {}; }
    if (bgr.empty()) { set_error(error, "Empty input image"); return {}; }

    // Letterbox to a square input, RGB, 0..1 (YOLO preprocessing).
    const Letterbox lb = Letterbox::compute(bgr.width, bgr.height, opt_.input_size);
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,
                                                      bgr.width, bgr.height, bgr.stride,
                                                      lb.new_w, lb.new_h);
    ncnn::Mat in;
    ncnn::copy_make_border(resized, in, lb.pad_top, lb.pad_bottom, lb.pad_left, lb.pad_right,
                           ncnn::BORDER_CONSTANT, kDetPadValue);
    const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in.substract_mean_normalize(nullptr, norm);

    ncnn::Extractor ex = net_->create_extractor();
    ex.set_light_mode(true);
    if (ex.input(opt_.input_blob.c_str(), in) != 0) {
        set_error(error, "Detector input blob not found: " + opt_.input_blob);
        return {};
    }
    ncnn::Mat out;
    if (ex.extract(opt_.output_blob.c_str(), out) != 0) {
        set_error(error, "Detector output blob not found: " + opt_.output_blob);
        return {};
    }

    const float* data = nullptr;
    int rows = 0, cols = 0;
    if (!as_matrix(out, data, rows, cols)) {
        set_error(error, "Unsupported detector output (dims=" + std::to_string(out.dims) +
                             ", c=" + std::to_string(out.c) + ")");
        return {};
    }

    YoloDecodeParams p;
    p.num_keypoints = kDetNumKeypoints;
    p.conf_threshold = opt_.conf_threshold;
    p.iou_threshold = opt_.iou_threshold;
    return decode_yolo_pose(data, rows, cols, lb, bgr.width, bgr.height, p, error);
}

}  // namespace calpr
