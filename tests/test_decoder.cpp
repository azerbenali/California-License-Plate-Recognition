// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Tests for CTC decoding and California format correction.

#include "calpr/plate_decoder.hpp"
#include "test_util.hpp"

#include <string>
#include <vector>

using namespace calpr;

static int class_of(char c) {
    for (int i = 0; i < kNumClasses; ++i)
        if (kCharset[i] == c) return i;
    return -1;
}

// Build a [classes x time] logit matrix whose argmax spells `frames` ('-' = blank).
static std::vector<float> make_logits(const std::string& frames) {
    const int T = int(frames.size());
    std::vector<float> m(size_t(kNumClasses) * T, 0.f);
    for (int t = 0; t < T; ++t) m[size_t(class_of(frames[t])) * T + t] = 10.f;
    return m;
}

static PlateResult run(const std::string& frames) {
    const auto m = make_logits(frames);
    const int T = int(frames.size());
    return interpret(ctc_greedy_decode(T, kNumClasses,
                                       [&](int t, int k) { return m[size_t(k) * T + t]; }));
}

int main() {
    auto r = run("-88-8KK-L-111-2-3--");
    CHECK(r.raw == "88KL123", "CTC collapse: repeats merge, blank separates real doubles");

    r = run("--8KMZ-2-4-5------");
    CHECK(r.text == "8KMZ245" && r.format_match && r.corrections == 0, "1980-2026 plate, no corrections");

    r = run("-1-2-3-A-B-C-1----");
    CHECK(r.text == "123ABC1" && r.format.find("2026+") != std::string::npos, "2026+ plate 123ABC1");

    r = run("-7-A-B-C-1-2-3----");
    CHECK(r.text == "7ABC123" && r.format.find("1980") != std::string::npos, "legacy plate picks the 1980 layout");

    // The plates in samples/ (photos ca_plate_01..03).
    r = run("-9-E-P-W-7-4-6----");
    CHECK(r.text == "9EPW746" && r.corrections == 0 && r.format.find("1980") != std::string::npos,
          "sample 01: 9EPW746, 1980-2026 layout");
    r = run("-2-0-1-A-A-A-1----");
    CHECK(r.text == "201AAA1" && r.corrections == 0 && r.format.find("2026+") != std::string::npos,
          "sample 02: 201AAA1, 2026+ layout");
    r = run("-P-E-R-C-5-0------");
    CHECK(r.text == "PERC50" && !r.format_match, "sample 03: personalized PERC50 left unchanged");

    r = run("-7-A-8-C-1-2-3----");
    CHECK(r.text == "7ABC123" && r.corrections == 1, "8 -> B in a letter slot");

    r = run("-7-A-B-C-1-Z-3----");
    CHECK(r.text == "7ABC123" && r.corrections == 1, "Z -> 2 in a digit slot");

    r = run("-1-2-3-A-0-C-1----");
    CHECK(r.text == "123AOC1", "0 -> O in the middle letter slot (O allowed there)");

    r = run("-7-O-B-C-1-2-3----");
    CHECK(r.text == "7DBC123", "O in a restricted letter slot -> D");

    r = run("-4-X-1-2-3-4-5----");
    CHECK(r.text == "4X12345" && r.format.find("Commercial") != std::string::npos, "commercial 1A23456");

    r = run("-H-E-L-L-O--------");
    CHECK(r.raw == "HELLO" && r.text == "HELLO" && !r.format_match, "vanity plate left unchanged");

    r = run("-K-K-K-K-K-K-K----");
    CHECK(!r.format_match && r.text == r.raw, "no forced fit when too many corrections are needed");

    r = run("------------------");
    CHECK(r.raw.empty() && !r.format_match && r.confidence == 0.f, "all-blank output is empty");

    // Confidence is the weakest character's softmax probability.
    {
        const int T = 4;
        std::vector<float> m(size_t(kNumClasses) * T, 0.f);
        m[size_t(class_of('A')) * T + 0] = 10.f;   // very confident
        m[size_t(kBlankIndex) * T + 1] = 10.f;     // blank
        m[size_t(class_of('B')) * T + 2] = 2.f;    // weak
        m[size_t(kBlankIndex) * T + 3] = 10.f;     // blank
        auto chars = ctc_greedy_decode(T, kNumClasses, [&](int t, int k) { return m[size_t(k) * T + t]; });
        auto res = interpret(chars);
        const double pB = std::exp(2.0) / (std::exp(2.0) + (kNumClasses - 1));
        CHECK(res.raw == "AB" && near(res.confidence, pB, 1e-4), "confidence = min per-char probability");
    }

    return test_summary();
}
