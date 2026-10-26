// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Tests for geometry.hpp and image.hpp.

#include "calpr/config.hpp"
#include "calpr/geometry.hpp"
#include "calpr/image.hpp"
#include "test_util.hpp"

using namespace calpr;

int main() {
    // IoU
    CHECK(near(iou({0, 0, 10, 10}, {0, 0, 10, 10}), 1.0), "IoU of identical boxes is 1");
    CHECK(near(iou({0, 0, 10, 10}, {20, 20, 30, 30}), 0.0), "IoU of disjoint boxes is 0");
    CHECK(near(iou({0, 0, 10, 10}, {5, 0, 15, 10}), 50.0 / 150.0), "IoU of half-overlapping boxes");

    // NMS
    {
        std::vector<Box> b = {{0, 0, 10, 10}, {1, 1, 11, 11}, {50, 50, 60, 60}};
        std::vector<float> s = {0.8f, 0.9f, 0.7f};
        auto keep = nms(b, s, 0.5f);
        CHECK(keep.size() == 2 && keep[0] == 1 && keep[1] == 2, "NMS keeps best of overlapping pair + separate box");
    }

    // Corner ordering
    {
        const Quad shuffled = {Point{90, 52}, Point{10, 10}, Point{12, 50}, Point{88, 12}};
        const Quad o = order_corners(shuffled);
        CHECK(o[0].x == 10 && o[1].x == 88 && o[2].x == 90 && o[3].x == 12, "order_corners -> TL, TR, BR, BL");
    }

    // Letterbox
    {
        const Letterbox lb = Letterbox::compute(1280, 720, 640);
        CHECK(lb.new_w == 640 && lb.new_h == 360 && lb.pad_top == 140 && lb.pad_bottom == 140,
              "letterbox 1280x720 -> 640x360 + 140px bars");
        const Point p = lb.to_source(320.f, 320.f);
        CHECK(near(p.x, 640.0) && near(p.y, 360.0), "letterbox maps model center to image center");
        const Letterbox odd = Letterbox::compute(333, 101, 640);
        CHECK(odd.new_w + odd.pad_left + odd.pad_right == 640 &&
              odd.new_h + odd.pad_top + odd.pad_bottom == 640, "letterbox padding fills the square");
    }

    // Homography
    {
        const Quad src = {Point{10, 20}, Point{200, 35}, Point{190, 120}, Point{5, 100}};
        const Quad dst = {Point{0, 0}, Point{239, 0}, Point{239, 119}, Point{0, 119}};
        Homography h;
        const bool ok = find_homography(src, dst, h);
        bool maps = ok;
        for (int i = 0; ok && i < 4; ++i) {
            const Point p = apply(h, src[i].x, src[i].y);
            maps = maps && near(p.x, dst[i].x, 1e-3) && near(p.y, dst[i].y, 1e-3);
        }
        CHECK(maps, "homography maps all four correspondences");
        const Quad degenerate = {Point{0, 0}, Point{0, 0}, Point{0, 0}, Point{0, 0}};
        CHECK(!find_homography(degenerate, dst, h), "degenerate quad is rejected");
    }

    // Warp: identity mapping reproduces the image exactly
    {
        Image img(16, 8);
        for (size_t i = 0; i < img.data.size(); ++i) img.data[i] = uint8_t((i * 37) % 251);
        const Quad full = {Point{0, 0}, Point{15, 0}, Point{15, 7}, Point{0, 7}};
        const Image w = warp_perspective(img.view(), full, 16, 8);
        CHECK(w.width == 16 && w.height == 8 && w.data == img.data, "identity warp is an exact copy");

        const Quad flipped = {Point{15, 0}, Point{0, 0}, Point{0, 7}, Point{15, 7}};
        const Image f = warp_perspective(img.view(), flipped, 16, 8);
        bool mirrored = true;
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 16; ++x)
                for (int c = 0; c < 3; ++c)
                    mirrored = mirrored && f.data[(y * 16 + x) * 3 + c] == img.data[(y * 16 + (15 - x)) * 3 + c];
        CHECK(mirrored, "swapped corners give a mirror image");

        const Quad outside = {Point{100, 100}, Point{120, 100}, Point{120, 110}, Point{100, 110}};
        const Image o = warp_perspective(img.view(), outside, 8, 4);
        bool black = true;
        for (auto v : o.data) black = black && v == 0;
        CHECK(black, "sampling outside the image gives the black border");
    }

    // Row crop (character band)
    {
        Image img(4, kPlateWarpH);
        const ImageView band = crop_rows(img.view(), kBandTop, kBandBottom);
        CHECK(band.height == 105 - 26 && band.data == img.data.data() + 26 * 12,
              "character band of a 120px plate is rows 26..104");
    }

    return test_summary();
}
