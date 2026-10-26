// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Tests for YOLO pose output decoding with synthetic tensors.

#include "calpr/yolo_pose.hpp"
#include "test_util.hpp"

#include <string>
#include <vector>

using namespace calpr;

namespace {

constexpr int K = 4;

struct FakeDet {
    float x1, y1, x2, y2, score;
};

// Plate corners in model-input pixels for a box (slightly skewed quad).
void corners_for(const FakeDet& d, float* kp) {
    const float s = 3.f;
    const float pts[8] = {d.x1 + s, d.y1, d.x2, d.y1 + s, d.x2 - s, d.y2, d.x1, d.y2 - s};
    for (int k = 0; k < 4; ++k) {
        kp[3 * k] = pts[2 * k];
        kp[3 * k + 1] = pts[2 * k + 1];
        kp[3 * k + 2] = 0.95f;
    }
}

// Classic layout: attributes (4 + 1 + 3K = 17) x anchors.
std::vector<float> classic(const std::vector<FakeDet>& dets, int anchors, bool transposed) {
    const int A = 4 + 1 + 3 * K;
    std::vector<float> m(size_t(A) * anchors, 0.f);
    auto set = [&](int a, int i, float v) {
        if (transposed) m[size_t(i) * A + a] = v; else m[size_t(a) * anchors + i] = v;
    };
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        set(0, int(i), (d.x1 + d.x2) / 2); set(1, int(i), (d.y1 + d.y2) / 2);
        set(2, int(i), d.x2 - d.x1);       set(3, int(i), d.y2 - d.y1);
        set(4, int(i), d.score);
        float kp[12];
        corners_for(d, kp);
        for (int j = 0; j < 12; ++j) set(5 + j, int(i), kp[j]);
    }
    return m;
}

// End-to-end layout: 300 detections x (6 + 3K = 18).
std::vector<float> end2end(const std::vector<FakeDet>& dets) {
    const int A = 6 + 3 * K, N = 300;
    std::vector<float> m(size_t(A) * N, 0.f);
    for (size_t i = 0; i < dets.size(); ++i) {
        float* r = &m[i * A];
        const auto& d = dets[i];
        r[0] = d.x1; r[1] = d.y1; r[2] = d.x2; r[3] = d.y2; r[4] = d.score; r[5] = 0;
        corners_for(d, r + 6);
    }
    return m;
}

}  // namespace

int main() {
    // A 1280x720 frame letterboxed to 640: scale 0.5, 140px bars top/bottom.
    const Letterbox lb = Letterbox::compute(1280, 720, 640);
    YoloDecodeParams p;
    p.num_keypoints = K;
    p.conf_threshold = 0.4f;
    p.iou_threshold = 0.5f;

    const std::vector<FakeDet> dets = {
        {100, 200, 200, 250, 0.90f},   // plate A
        {102, 201, 201, 251, 0.85f},   // duplicate of A (should be suppressed)
        {400, 300, 480, 340, 0.70f},   // plate B
        {10, 150, 60, 180, 0.20f},     // below threshold
    };

    int nc = 0;
    CHECK(detect_layout(17, 8400, K, &nc) == YoloLayout::Classic && nc == 1, "17x8400 -> classic, 1 class");
    CHECK(detect_layout(300, 18, K, &nc) == YoloLayout::EndToEnd, "300x18 -> end-to-end");
    CHECK(detect_layout(18, 8400, K, &nc) == YoloLayout::Classic && nc == 2, "18x8400 -> classic, 2 classes");
    CHECK(detect_layout(20, 20, K, &nc) == YoloLayout::Unknown, "small square tensor is rejected");
    CHECK(detect_layout(17, 500, K, &nc) == YoloLayout::Unknown, "too few anchors for a classic head is rejected");

    {
        const auto m = classic(dets, 2100, false);
        const auto out = decode_yolo_pose(m.data(), 17, 2100, lb, 1280, 720, p);
        CHECK(out.size() == 2, "classic: NMS + threshold leave 2 plates");
        if (out.size() == 2) {
            CHECK(near(out[0].score, 0.90), "classic: best plate first");
            // (100, 200) in model input -> ((100-0)/0.5, (200-140)/0.5) = (200, 120)
            CHECK(near(out[0].box.x0, 200) && near(out[0].box.y0, 120) &&
                  near(out[0].box.x1, 400) && near(out[0].box.y1, 220), "classic: box mapped to source pixels");
            CHECK(near(out[0].corners[0].x, 206) && near(out[0].corners[0].y, 120), "classic: TL corner mapped");
            CHECK(near(out[0].corner_conf, 0.95), "classic: corner confidence");
        }
    }
    {
        const auto m = classic(dets, 2100, true);
        const auto out = decode_yolo_pose(m.data(), 2100, 17, lb, 1280, 720, p);
        CHECK(out.size() == 2 && near(out[0].box.x0, 200), "classic transposed: same result");
    }
    {
        // NMS-free head: duplicates are not expected, so feed A, B and the low one.
        const std::vector<FakeDet> e2e_dets = {dets[0], dets[2], dets[3]};
        const auto m = end2end(e2e_dets);
        const auto out = decode_yolo_pose(m.data(), 300, 18, lb, 1280, 720, p);
        CHECK(out.size() == 2 && near(out[1].box.x0, 800) && near(out[1].box.y1, 400),
              "end-to-end: 2 plates, boxes mapped");
    }
    {
        std::string err;
        std::vector<float> junk(10 * 10, 0.f);
        const auto out = decode_yolo_pose(junk.data(), 10, 10, lb, 1280, 720, p, &err);
        CHECK(out.empty() && err.find("Unexpected detector output shape") != std::string::npos,
              "unknown shape reports an error");
    }
    {
        // Boxes running off the image are clamped.
        const std::vector<FakeDet> edge = {{-50, 100, 100, 200, 0.9f}};
        const auto m = classic(edge, 2100, false);
        const auto out = decode_yolo_pose(m.data(), 17, 2100, lb, 1280, 720, p);
        CHECK(out.size() == 1 && out[0].box.x0 == 0.f, "boxes are clamped to the image");
    }

    return test_summary();
}
