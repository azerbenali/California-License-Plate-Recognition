// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Small geometry toolkit: boxes, IoU/NMS, corner ordering, letterbox mapping
// and homographies. Header-only, no ncnn/OpenCV, so it is unit-testable and
// portable to mobile.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace calpr {

struct Point {
    float x = 0.f, y = 0.f;
};

// Axis-aligned box, corner form.
struct Box {
    float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
    float width() const { return std::max(0.f, x1 - x0); }
    float height() const { return std::max(0.f, y1 - y0); }
    float area() const { return width() * height(); }
};

// Plate corners in the order top-left, top-right, bottom-right, bottom-left.
using Quad = std::array<Point, 4>;

inline Quad box_to_quad(const Box& b) {
    return {Point{b.x0, b.y0}, Point{b.x1, b.y0}, Point{b.x1, b.y1}, Point{b.x0, b.y1}};
}

inline float iou(const Box& a, const Box& b) {
    const Box inter{std::max(a.x0, b.x0), std::max(a.y0, b.y0),
                    std::min(a.x1, b.x1), std::min(a.y1, b.y1)};
    const float i = inter.area();
    const float u = a.area() + b.area() - i;
    return u > 0.f ? i / u : 0.f;
}

// Greedy non-maximum suppression. Returns indices of kept boxes, best first.
inline std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores,
                            float iou_threshold) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<char> removed(boxes.size(), 0);
    for (size_t i = 0; i < order.size(); ++i) {
        const int a = order[i];
        if (removed[a]) continue;
        keep.push_back(a);
        for (size_t j = i + 1; j < order.size(); ++j) {
            const int b = order[j];
            if (!removed[b] && iou(boxes[a], boxes[b]) > iou_threshold) removed[b] = 1;
        }
    }
    return keep;
}

// Put 4 arbitrary corners into TL, TR, BR, BL order. Uses x+y (TL smallest,
// BR largest) and y-x (TR smallest, BL largest); valid for plates rotated by
// less than 45 degrees, which covers real traffic footage. Falls back to the
// input order if the heuristic picks the same point twice.
inline Quad order_corners(const Quad& q) {
    int tl = 0, br = 0, tr = 0, bl = 0;
    for (int i = 1; i < 4; ++i) {
        const float s = q[i].x + q[i].y, d = q[i].y - q[i].x;
        if (s < q[tl].x + q[tl].y) tl = i;
        if (s > q[br].x + q[br].y) br = i;
        if (d < q[tr].y - q[tr].x) tr = i;
        if (d > q[bl].y - q[bl].x) bl = i;
    }
    const int mask = (1 << tl) | (1 << tr) | (1 << br) | (1 << bl);
    if (mask != 0xF) return q;
    return {q[tl], q[tr], q[br], q[bl]};
}

inline Box quad_bounds(const Quad& q) {
    Box b{q[0].x, q[0].y, q[0].x, q[0].y};
    for (const auto& p : q) {
        b.x0 = std::min(b.x0, p.x); b.y0 = std::min(b.y0, p.y);
        b.x1 = std::max(b.x1, p.x); b.y1 = std::max(b.y1, p.y);
    }
    return b;
}

// Letterbox: resize keeping aspect ratio, then pad to a square of `size`,
// centered (same as the YOLO export-time preprocessing).
struct Letterbox {
    float scale = 1.f;
    int new_w = 0, new_h = 0;
    int pad_left = 0, pad_top = 0, pad_right = 0, pad_bottom = 0;

    static Letterbox compute(int src_w, int src_h, int size) {
        Letterbox lb;
        lb.scale = std::min(float(size) / float(src_w), float(size) / float(src_h));
        lb.new_w = std::max(1, int(std::lround(float(src_w) * lb.scale)));
        lb.new_h = std::max(1, int(std::lround(float(src_h) * lb.scale)));
        const int pw = size - lb.new_w, ph = size - lb.new_h;
        lb.pad_left = pw / 2;  lb.pad_right = pw - lb.pad_left;
        lb.pad_top = ph / 2;   lb.pad_bottom = ph - lb.pad_top;
        return lb;
    }
    // Model-input coordinates -> original image coordinates.
    Point to_source(float x, float y) const {
        return {(x - float(pad_left)) / scale, (y - float(pad_top)) / scale};
    }
};

// 3x3 homography, row-major, h[8] is 1.
using Homography = std::array<double, 9>;

inline Point apply(const Homography& h, double x, double y) {
    const double w = h[6] * x + h[7] * y + h[8];
    return {float((h[0] * x + h[1] * y + h[2]) / w), float((h[3] * x + h[4] * y + h[5]) / w)};
}

// Homography mapping src[i] -> dst[i] (same convention as
// cv::getPerspectiveTransform). Returns false for degenerate input.
inline bool find_homography(const Quad& src, const Quad& dst, Homography& out) {
    // Unknowns h0..h7 (h8 = 1). Two equations per correspondence.
    double a[8][9] = {};
    for (int i = 0; i < 4; ++i) {
        const double x = src[i].x, y = src[i].y, u = dst[i].x, v = dst[i].y;
        double* r0 = a[2 * i];
        double* r1 = a[2 * i + 1];
        r0[0] = x; r0[1] = y; r0[2] = 1; r0[6] = -u * x; r0[7] = -u * y; r0[8] = u;
        r1[3] = x; r1[4] = y; r1[5] = 1; r1[6] = -v * x; r1[7] = -v * y; r1[8] = v;
    }
    // Gaussian elimination with partial pivoting on the 8x9 augmented matrix.
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        for (int r = col + 1; r < 8; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
        if (std::fabs(a[piv][col]) < 1e-12) return false;
        if (piv != col)
            for (int k = 0; k < 9; ++k) std::swap(a[col][k], a[piv][k]);
        for (int r = 0; r < 8; ++r) {
            if (r == col) continue;
            const double f = a[r][col] / a[col][col];
            if (f == 0.0) continue;
            for (int k = col; k < 9; ++k) a[r][k] -= f * a[col][k];
        }
    }
    for (int i = 0; i < 8; ++i) out[i] = a[i][8] / a[i][i];
    out[8] = 1.0;
    return true;
}

}  // namespace calpr
