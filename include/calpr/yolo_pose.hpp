// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Decoding of YOLO pose-model output (plate box + 4 corner keypoints).
// Pure C++ (no ncnn) so it can be tested with synthetic tensors.
//
// Two export layouts are supported, detected from the tensor shape:
//
//  1. Classic one-to-many head (YOLOv8 / YOLO11 pose):
//       attributes x anchors, attributes = 4 + nc + 3K
//       [cx, cy, w, h, class scores..., k0x, k0y, k0v, ...]
//     Boxes and keypoints are in model-input pixels, scores already sigmoid.
//     Needs NMS.
//
//  2. End-to-end head (YOLO26 default, NMS-free):
//       detections x attributes, attributes = 6 + 3K
//       [x1, y1, x2, y2, score, class, k0x, k0y, k0v, ...]
//
// Either may arrive transposed; the attribute axis is the short one.
#pragma once

#include "calpr/geometry.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace calpr {

struct PlateDetection {
    Box box;                   // original-image pixels
    float score = 0.f;
    Quad corners{};            // TL, TR, BR, BL in original-image pixels
    float corner_conf = 0.f;   // lowest keypoint visibility score
};

enum class YoloLayout { Classic, EndToEnd, Unknown };

struct YoloDecodeParams {
    int num_keypoints = 4;
    float conf_threshold = 0.4f;
    float iou_threshold = 0.5f;
};

// Classic heads have thousands of anchors; end-to-end heads output at most
// max_det rows (300 by default). This splits the two when the attribute
// counts collide (e.g. 18 = 6 + 3*4 = 4 + 2 + 3*4).
inline constexpr int kMaxEndToEndRows = 1000;

inline YoloLayout detect_layout(int rows, int cols, int num_kpts, int* num_classes) {
    const int small = std::min(rows, cols), large = std::max(rows, cols);
    if (small == 6 + 3 * num_kpts && large <= kMaxEndToEndRows) {
        if (num_classes) *num_classes = 1;
        return YoloLayout::EndToEnd;
    }
    const int nc = small - 4 - 3 * num_kpts;
    if (nc >= 1 && large > kMaxEndToEndRows) {
        if (num_classes) *num_classes = nc;
        return YoloLayout::Classic;
    }
    return YoloLayout::Unknown;
}

// data : row-major rows x cols float matrix (the raw model output)
// lb   : letterbox used to build the model input
// img_w, img_h : original image size, used for clamping
inline std::vector<PlateDetection> decode_yolo_pose(const float* data, int rows, int cols,
                                                    const Letterbox& lb, int img_w, int img_h,
                                                    const YoloDecodeParams& p,
                                                    std::string* error = nullptr) {
    std::vector<PlateDetection> dets;
    int nc = 1;
    const YoloLayout layout = detect_layout(rows, cols, p.num_keypoints, &nc);
    if (layout == YoloLayout::Unknown) {
        if (error)
            *error = "Unexpected detector output shape " + std::to_string(rows) + "x" +
                     std::to_string(cols) + " for a " + std::to_string(p.num_keypoints) +
                     "-keypoint pose model";
        return dets;
    }

    const bool attr_major = rows <= cols;  // attributes on the short axis
    const int n = std::max(rows, cols);
    auto at = [&](int attr, int i) -> float {
        return attr_major ? data[size_t(attr) * size_t(cols) + size_t(i)]
                          : data[size_t(i) * size_t(cols) + size_t(attr)];
    };
    auto clamp_pt = [&](Point q) {
        q.x = std::min(std::max(q.x, 0.f), float(img_w - 1));
        q.y = std::min(std::max(q.y, 0.f), float(img_h - 1));
        return q;
    };

    const int kpt_base = (layout == YoloLayout::Classic) ? 4 + nc : 6;
    for (int i = 0; i < n; ++i) {
        float score = 0.f;
        Point a, b;  // box corners in model-input pixels
        if (layout == YoloLayout::Classic) {
            for (int c = 0; c < nc; ++c) score = std::max(score, at(4 + c, i));
            if (score < p.conf_threshold) continue;
            const float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
            a = {cx - w * 0.5f, cy - h * 0.5f};
            b = {cx + w * 0.5f, cy + h * 0.5f};
        } else {
            score = at(4, i);
            if (score < p.conf_threshold) continue;
            a = {at(0, i), at(1, i)};
            b = {at(2, i), at(3, i)};
        }

        PlateDetection d;
        d.score = score;
        const Point s0 = clamp_pt(lb.to_source(a.x, a.y));
        const Point s1 = clamp_pt(lb.to_source(b.x, b.y));
        d.box = {s0.x, s0.y, s1.x, s1.y};
        d.corner_conf = 1.f;
        for (int k = 0; k < p.num_keypoints && k < 4; ++k) {
            const int base = kpt_base + 3 * k;
            d.corners[k] = clamp_pt(lb.to_source(at(base, i), at(base + 1, i)));
            d.corner_conf = std::min(d.corner_conf, at(base + 2, i));
        }
        dets.push_back(d);
    }

    if (layout == YoloLayout::Classic) {
        std::vector<Box> boxes;
        std::vector<float> scores;
        for (const auto& d : dets) { boxes.push_back(d.box); scores.push_back(d.score); }
        std::vector<PlateDetection> kept;
        for (int idx : nms(boxes, scores, p.iou_threshold)) kept.push_back(dets[idx]);
        return kept;
    }
    std::sort(dets.begin(), dets.end(),
              [](const PlateDetection& x, const PlateDetection& y) { return x.score > y.score; });
    return dets;
}

}  // namespace calpr
