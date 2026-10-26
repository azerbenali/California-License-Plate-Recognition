// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Minimal 8-bit BGR image types plus the two image operations the pipeline
// needs (row crop and perspective warp). No OpenCV, so the core library stays
// small enough for mobile and can be tested anywhere.
#pragma once

#include "calpr/geometry.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace calpr {

// Non-owning view of an interleaved 3-channel BGR image.
struct ImageView {
    const uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;  // bytes per row (>= width * 3)

    bool empty() const { return data == nullptr || width <= 0 || height <= 0; }
};

// Owning, tightly packed BGR image.
struct Image {
    std::vector<uint8_t> data;
    int width = 0;
    int height = 0;

    Image() = default;
    Image(int w, int h) : data(size_t(w) * size_t(h) * 3, 0), width(w), height(h) {}
    ImageView view() const { return {data.data(), width, height, width * 3}; }
};

// Rows [top_frac * h, bottom_frac * h) of `src`, as a view (no copy).
inline ImageView crop_rows(const ImageView& src, float top_frac, float bottom_frac) {
    int y0 = int(float(src.height) * top_frac);
    int y1 = int(float(src.height) * bottom_frac);
    y0 = std::max(0, std::min(y0, src.height - 1));
    y1 = std::max(y0 + 1, std::min(y1, src.height));
    return {src.data + size_t(y0) * size_t(src.stride), src.width, y1 - y0, src.stride};
}

// Warp the quadrilateral `corners` (TL, TR, BR, BL, in src pixels) onto an
// out_w x out_h image. Same mapping as
//   cv::warpPerspective(src, dst, cv::getPerspectiveTransform(corners, rect), size)
// with rect = (0,0), (w-1,0), (w-1,h-1), (0,h-1), bilinear sampling and a
// black constant border. Returns an empty image if the quad is degenerate.
inline Image warp_perspective(const ImageView& src, const Quad& corners, int out_w, int out_h) {
    const Quad rect = {Point{0.f, 0.f}, Point{float(out_w - 1), 0.f},
                       Point{float(out_w - 1), float(out_h - 1)}, Point{0.f, float(out_h - 1)}};
    Homography inv;  // output pixel -> source pixel
    if (src.empty() || out_w <= 0 || out_h <= 0 || !find_homography(rect, corners, inv))
        return Image();

    Image out(out_w, out_h);
    auto px = [&](int x, int y, int ch) -> float {
        if (x < 0 || y < 0 || x >= src.width || y >= src.height) return 0.f;
        return float(src.data[size_t(y) * size_t(src.stride) + size_t(x) * 3 + size_t(ch)]);
    };
    for (int v = 0; v < out_h; ++v) {
        uint8_t* row = out.data.data() + size_t(v) * size_t(out_w) * 3;
        for (int u = 0; u < out_w; ++u) {
            const Point s = apply(inv, u, v);
            // Far outside (or NaN): border value. Also keeps the int casts safe.
            if (!(s.x > -2.f && s.y > -2.f && s.x < float(src.width) + 1.f &&
                  s.y < float(src.height) + 1.f))
                continue;  // already zero
            const float fx = std::floor(s.x), fy = std::floor(s.y);
            const int x0 = int(fx), y0 = int(fy);
            const float ax = s.x - fx, ay = s.y - fy;
            for (int ch = 0; ch < 3; ++ch) {
                const float top = px(x0, y0, ch) * (1.f - ax) + px(x0 + 1, y0, ch) * ax;
                const float bot = px(x0, y0 + 1, ch) * (1.f - ax) + px(x0 + 1, y0 + 1, ch) * ax;
                const float val = top * (1.f - ay) + bot * ay;
                row[u * 3 + ch] = uint8_t(std::min(255.f, std::max(0.f, std::round(val))));
            }
        }
    }
    return out;
}

}  // namespace calpr
