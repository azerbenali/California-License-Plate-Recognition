// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0

#include "calpr/pipeline.hpp"

#include <utility>

namespace calpr {

PlateRead PlatePipeline::rectify(const ImageView& bgr, const PlateDetection& det) const {
    PlateRead r;
    r.detection = det;
    r.used_keypoints = det.corner_conf >= opt_.min_corner_conf;
    r.corners_used = r.used_keypoints ? order_corners(det.corners) : box_to_quad(det.box);
    r.rectified = warp_perspective(bgr, r.corners_used, kPlateWarpW, kPlateWarpH);

    if (r.rectified.data.empty() && r.used_keypoints) {
        // Degenerate keypoints (e.g. collapsed corners): use the box instead.
        r.used_keypoints = false;
        r.corners_used = box_to_quad(det.box);
        r.rectified = warp_perspective(bgr, r.corners_used, kPlateWarpW, kPlateWarpH);
    }
    return r;
}

std::vector<PlateRead> PlatePipeline::run(const ImageView& bgr, std::string* error) const {
    std::vector<PlateRead> reads;
    std::string err;
    const std::vector<PlateDetection> dets = det_.detect(bgr, &err);
    if (!err.empty()) {
        if (error) *error = err;
        return reads;
    }
    for (const auto& d : dets) {
        PlateRead r = rectify(bgr, d);
        if (r.rectified.data.empty()) continue;  // zero-area detection
        if (!rec_.recognize(r.rectified.view(), r.result, &err)) {
            if (error) *error = err;
            continue;
        }
        reads.push_back(std::move(r));
    }
    return reads;
}

}  // namespace calpr
