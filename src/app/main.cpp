// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Command-line app: image, video or camera in; plate text out, plus an
// optional annotated image/video and rectified plate crops.

#include "calpr/pipeline.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::string image, video;
    int camera = -1;
    std::string det_param, det_bin, rec_param, rec_bin;
    std::string det_in = "in0", det_out = "out0", rec_in = "in0", rec_out = "out0";
    int det_size = calpr::kDetInputSize;
    float conf = 0.4f, iou = 0.5f, min_rec_conf = 0.5f;
    bool crop_band = true;
    int threads = 2;
    std::string out, save_crops;
    bool show = false, json = false;
};

void usage() {
    std::fprintf(stderr,
        "California License Plate Recognition - Azer Ben Ali, 2026\n\n"
        "Input (one of):\n"
        "  --image PATH        still image\n"
        "  --video PATH        video file\n"
        "  --camera INDEX      webcam\n"
        "Models:\n"
        "  --rec-param P --rec-bin B   recognizer (required)\n"
        "  --det-param P --det-bin B   detector; omit to treat the image as a\n"
        "                              tightly cropped plate (recognition only)\n"
        "  --rec-in/--rec-out/--det-in/--det-out NAME   blob names (default in0/out0)\n"
        "Tuning:\n"
        "  --det-size N        detector input size (default 640)\n"
        "  --conf F            detection confidence (default 0.4)\n"
        "  --iou F             NMS IoU (default 0.5)\n"
        "  --min-rec-conf F    hide reads below this confidence (default 0.5)\n"
        "  --no-crop           feed the whole plate, not the character band\n"
        "  --threads N         CPU threads (default 2)\n"
        "Output:\n"
        "  --out PATH          save annotated image or video\n"
        "  --save-crops DIR    save rectified plates (to build training data)\n"
        "  --json              one JSON line per frame instead of text\n"
        "  --show              display a window\n");
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        auto str = [&](std::string& dst) { const char* v = val(); if (!v) return false; dst = v; return true; };
        auto num = [&](float& dst) { const char* v = val(); if (!v) return false; dst = std::strtof(v, nullptr); return true; };
        auto inum = [&](int& dst) { const char* v = val(); if (!v) return false; dst = std::atoi(v); return true; };
        bool ok = true;
        if (k == "--image") ok = str(a.image);
        else if (k == "--video") ok = str(a.video);
        else if (k == "--camera") ok = inum(a.camera);
        else if (k == "--det-param") ok = str(a.det_param);
        else if (k == "--det-bin") ok = str(a.det_bin);
        else if (k == "--rec-param") ok = str(a.rec_param);
        else if (k == "--rec-bin") ok = str(a.rec_bin);
        else if (k == "--det-in") ok = str(a.det_in);
        else if (k == "--det-out") ok = str(a.det_out);
        else if (k == "--rec-in") ok = str(a.rec_in);
        else if (k == "--rec-out") ok = str(a.rec_out);
        else if (k == "--det-size") ok = inum(a.det_size);
        else if (k == "--conf") ok = num(a.conf);
        else if (k == "--iou") ok = num(a.iou);
        else if (k == "--min-rec-conf") ok = num(a.min_rec_conf);
        else if (k == "--threads") ok = inum(a.threads);
        else if (k == "--out") ok = str(a.out);
        else if (k == "--save-crops") ok = str(a.save_crops);
        else if (k == "--no-crop") a.crop_band = false;
        else if (k == "--show") a.show = true;
        else if (k == "--json") a.json = true;
        else if (k == "-h" || k == "--help") return false;
        else { std::fprintf(stderr, "Unknown option: %s\n", k.c_str()); return false; }
        if (!ok) { std::fprintf(stderr, "Missing value for %s\n", k.c_str()); return false; }
    }
    const int inputs = int(!a.image.empty()) + int(!a.video.empty()) + int(a.camera >= 0);
    if (inputs != 1) { std::fprintf(stderr, "Give exactly one of --image, --video, --camera\n"); return false; }
    if (a.rec_param.empty() || a.rec_bin.empty()) { std::fprintf(stderr, "--rec-param and --rec-bin are required\n"); return false; }
    if (a.det_param.empty() != a.det_bin.empty()) { std::fprintf(stderr, "Give both --det-param and --det-bin\n"); return false; }
    return true;
}

calpr::ImageView view_of(const cv::Mat& bgr) {
    return {bgr.data, bgr.cols, bgr.rows, int(bgr.step)};
}

cv::Mat to_bgr(const cv::Mat& m) {
    if (m.type() == CV_8UC3) return m;
    cv::Mat out;
    if (m.channels() == 1) cv::cvtColor(m, out, cv::COLOR_GRAY2BGR);
    else if (m.channels() == 4) cv::cvtColor(m, out, cv::COLOR_BGRA2BGR);
    else m.convertTo(out, CV_8UC3);
    return out;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
    return o;
}

struct App {
    Args args;
    calpr::PlateDetector detector;
    calpr::PlateRecognizer recognizer;
    bool use_detector = false;
    long crop_counter = 0;

    bool init() {
        std::string err;
        calpr::PlateRecognizer::Options ro;
        ro.param_path = args.rec_param; ro.bin_path = args.rec_bin;
        ro.input_blob = args.rec_in; ro.output_blob = args.rec_out;
        ro.crop_band = args.crop_band; ro.num_threads = args.threads;
        if (!recognizer.load(ro, &err)) { std::cerr << err << "\n"; return false; }

        use_detector = !args.det_param.empty();
        if (use_detector) {
            calpr::PlateDetector::Options d;
            d.param_path = args.det_param; d.bin_path = args.det_bin;
            d.input_blob = args.det_in; d.output_blob = args.det_out;
            d.input_size = args.det_size; d.conf_threshold = args.conf;
            d.iou_threshold = args.iou; d.num_threads = args.threads;
            if (!detector.load(d, &err)) { std::cerr << err << "\n"; return false; }
        }
        return true;
    }

    // Returns the reads that pass --min-rec-conf.
    std::vector<calpr::PlateRead> process(const cv::Mat& bgr) {
        std::vector<calpr::PlateRead> reads;
        std::string err;
        const calpr::ImageView v = view_of(bgr);
        if (use_detector) {
            calpr::PlatePipeline pipe(detector, recognizer);
            reads = pipe.run(v, &err);
        } else {
            // Recognition only: the whole image is the plate.
            calpr::PlateRead r;
            r.corners_used = calpr::box_to_quad({0.f, 0.f, float(bgr.cols - 1), float(bgr.rows - 1)});
            r.detection.box = {0.f, 0.f, float(bgr.cols - 1), float(bgr.rows - 1)};
            r.detection.score = 1.f;
            if (recognizer.recognize(v, r.result, &err)) reads.push_back(std::move(r));
        }
        if (!err.empty()) std::cerr << "warning: " << err << "\n";

        std::vector<calpr::PlateRead> kept;
        for (auto& r : reads)
            if (r.result.confidence >= args.min_rec_conf && !r.result.text.empty())
                kept.push_back(std::move(r));
        return kept;
    }

    void report(long frame, const std::vector<calpr::PlateRead>& reads) const {
        if (args.json) {
            std::cout << "{\"frame\":" << frame << ",\"plates\":[";
            for (size_t i = 0; i < reads.size(); ++i) {
                const auto& r = reads[i];
                const auto& b = r.detection.box;
                std::cout << (i ? "," : "") << "{\"text\":\"" << json_escape(r.result.text)
                          << "\",\"raw\":\"" << json_escape(r.result.raw)
                          << "\",\"format\":\"" << json_escape(r.result.format)
                          << "\",\"format_match\":" << (r.result.format_match ? "true" : "false")
                          << ",\"rec_conf\":" << r.result.confidence
                          << ",\"det_conf\":" << r.detection.score
                          << ",\"box\":[" << b.x0 << "," << b.y0 << "," << b.x1 << "," << b.y1 << "]}";
            }
            std::cout << "]}\n";
            return;
        }
        for (const auto& r : reads) {
            if (frame >= 0) std::cout << "[frame " << frame << "] ";
            std::cout << r.result.text << "  (" << r.result.format
                      << ", rec " << r.result.confidence
                      << ", det " << r.detection.score;
            if (r.result.corrections) std::cout << ", raw " << r.result.raw;
            std::cout << ")\n";
        }
    }

    void draw(cv::Mat& img, const std::vector<calpr::PlateRead>& reads) const {
        for (const auto& r : reads) {
            const cv::Scalar color = r.result.format_match ? cv::Scalar(60, 200, 60)
                                                           : cv::Scalar(0, 165, 255);
            std::vector<cv::Point> poly;
            for (const auto& p : r.corners_used) poly.emplace_back(int(p.x), int(p.y));
            cv::polylines(img, poly, true, color, 2, cv::LINE_AA);

            const std::string label = r.result.text;
            int base = 0;
            const double scale = std::max(0.6, img.cols / 1400.0);
            const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, 2, &base);
            const int x = std::max(0, int(r.detection.box.x0));
            const int y = std::max(ts.height + 6, int(r.detection.box.y0) - 6);
            cv::rectangle(img, {x, y - ts.height - 6}, {x + ts.width + 8, y + 4}, color, cv::FILLED);
            cv::putText(img, label, {x + 4, y - 2}, cv::FONT_HERSHEY_SIMPLEX, scale,
                        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
    }

    void save_crops(const std::vector<calpr::PlateRead>& reads) {
        if (args.save_crops.empty()) return;
        for (const auto& r : reads) {
            if (r.rectified.data.empty()) continue;
            cv::Mat m(r.rectified.height, r.rectified.width, CV_8UC3,
                      const_cast<uint8_t*>(r.rectified.data.data()));
            char name[64];
            std::snprintf(name, sizeof(name), "/plate_%06ld.png", crop_counter++);
            cv::imwrite(args.save_crops + name, m);
        }
    }
};

int run_image(App& app) {
    cv::Mat img = cv::imread(app.args.image, cv::IMREAD_COLOR);
    if (img.empty()) { std::cerr << "Cannot read image: " << app.args.image << "\n"; return 1; }
    img = to_bgr(img);
    const auto reads = app.process(img);
    app.report(-1, reads);
    app.save_crops(reads);
    if (!app.args.out.empty() || app.args.show) {
        cv::Mat vis = img.clone();
        app.draw(vis, reads);
        if (!app.args.out.empty() && !cv::imwrite(app.args.out, vis))
            std::cerr << "Cannot write " << app.args.out << "\n";
        if (app.args.show) { cv::imshow("calpr", vis); cv::waitKey(0); }
    }
    if (reads.empty() && !app.args.json) std::cout << "No plate read.\n";
    return reads.empty() ? 3 : 0;
}

int run_stream(App& app) {
    cv::VideoCapture cap;
    if (!app.args.video.empty()) cap.open(app.args.video);
    else cap.open(app.args.camera);
    if (!cap.isOpened()) { std::cerr << "Cannot open video source\n"; return 1; }

    cv::VideoWriter writer;
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0 || fps > 240) fps = 25.0;

    cv::Mat frame;
    for (long idx = 0; cap.read(frame); ++idx) {
        frame = to_bgr(frame);
        const auto reads = app.process(frame);
        if (!reads.empty() || app.args.json) app.report(idx, reads);
        app.save_crops(reads);

        if (!app.args.out.empty() || app.args.show) {
            app.draw(frame, reads);
            if (!app.args.out.empty()) {
                if (!writer.isOpened()) {
                    writer.open(app.args.out, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                                frame.size());
                    if (!writer.isOpened()) { std::cerr << "Cannot write " << app.args.out << "\n"; return 1; }
                }
                writer.write(frame);
            }
            if (app.args.show) {
                cv::imshow("calpr", frame);
                if ((cv::waitKey(1) & 0xFF) == 27) break;  // Esc
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    App app;
    if (!parse(argc, argv, app.args)) { usage(); return 1; }
    if (!app.init()) return 1;
    return app.args.image.empty() ? run_stream(app) : run_image(app);
}
