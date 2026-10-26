// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// This file was modified by Azer Ben Ali in 2026: CTC greedy decoding
// rewritten for a 37-class California charset, variable sequence length,
// softmax confidence, and California format checks.
//
// Header-only and free of ncnn/OpenCV so it can be unit-tested on its own.
#pragma once

#include "calpr/config.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace calpr {

struct DecodedChar {
    char c;
    float prob;  // softmax probability of the chosen class
};

// Greedy CTC decoding.
//   T     : number of time steps (18 for LPRNet)
//   K     : number of classes (kNumClasses)
//   score : callable score(t, k) returning the raw logit of class k at step t
template <typename ScoreFn>
std::vector<DecodedChar> ctc_greedy_decode(int T, int K, ScoreFn score) {
    std::vector<DecodedChar> out;
    int prev = kBlankIndex;
    for (int t = 0; t < T; ++t) {
        int best = 0;
        float best_v = score(t, 0);
        for (int k = 1; k < K; ++k) {
            const float v = score(t, k);
            if (v > best_v) { best_v = v; best = k; }
        }
        double denom = 0.0;
        for (int k = 0; k < K; ++k) denom += std::exp(double(score(t, k) - best_v));
        const float p = float(1.0 / denom);

        if (best != kBlankIndex) {
            if (best != prev) {
                out.push_back({kCharset[best], p});
            } else if (!out.empty()) {
                out.back().prob = std::max(out.back().prob, p);  // repeated frame
            }
        }
        prev = best;
    }
    return out;
}

// -----------------------------------------------------------------------------
// California plate layouts.
//   D = digit,  L = any letter,
//   R = letter except I, O, Q (California skips these as the 1st and 3rd
//       letter of the 3-letter block on standard passenger plates)
// -----------------------------------------------------------------------------
struct PlateFormat {
    const char* name;
    const char* pattern;
};

// Order = tie-break priority (most common first).
inline const std::vector<PlateFormat>& ca_formats() {
    static const std::vector<PlateFormat> f = {
        {"Passenger (2026+, 123ABC1)",     "DDDRLRD"},
        {"Passenger (1980-2026, 1ABC234)", "DRLRDDD"},
        {"Commercial (1A23456)",           "DLDDDDD"},
        {"Trailer (1AB1234)",              "DLLDDDD"},
    };
    return f;
}

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
inline bool is_letter(char c) { return c >= 'A' && c <= 'Z'; }
inline bool is_ioq(char c) { return c == 'I' || c == 'O' || c == 'Q'; }

// Common OCR confusions, letter -> digit. Returns 0 if there is no sensible one.
inline char letter_to_digit(char c) {
    switch (c) {
        case 'O': case 'Q': case 'D': case 'U': return '0';
        case 'I': case 'L': case 'J':           return '1';
        case 'Z':                               return '2';
        case 'A':                               return '4';
        case 'S':                               return '5';
        case 'G':                               return '6';
        case 'T':                               return '7';
        case 'B':                               return '8';
        default:                                return 0;
    }
}

// Common OCR confusions, digit -> letter. Returns 0 if there is no sensible one.
inline char digit_to_letter(char c, bool restricted) {
    switch (c) {
        case '0': return restricted ? 'D' : 'O';  // O not allowed there -> D
        case '1': return restricted ? 0 : 'I';    // I not allowed there
        case '2': return 'Z';
        case '4': return 'A';
        case '5': return 'S';
        case '6': return 'G';
        case '7': return 'T';
        case '8': return 'B';
        default:  return 0;
    }
}

// Force `s` into `pattern`. Returns the number of corrections, or -1 if it
// cannot fit. `fixed` receives the corrected string.
inline int fit_pattern(const std::string& s, const char* pattern, std::string& fixed) {
    const std::string p(pattern);
    if (s.size() != p.size()) return -1;
    fixed = s;
    int edits = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const char c = s[i];
        switch (p[i]) {
            case 'D':
                if (!is_digit(c)) {
                    const char d = letter_to_digit(c);
                    if (!d) return -1;
                    fixed[i] = d; ++edits;
                }
                break;
            case 'L':
                if (!is_letter(c)) {
                    const char l = digit_to_letter(c, false);
                    if (!l) return -1;
                    fixed[i] = l; ++edits;
                }
                break;
            case 'R':
                if (is_letter(c)) {
                    if (is_ioq(c)) {
                        if (c == 'I') return -1;  // no safe substitute
                        fixed[i] = 'D'; ++edits;  // O/Q -> D
                    }
                } else {
                    const char l = digit_to_letter(c, true);
                    if (!l) return -1;
                    fixed[i] = l; ++edits;
                }
                break;
            default:
                return -1;
        }
    }
    return edits;
}

struct PlateResult {
    std::string raw;          // straight CTC output
    std::string text;         // after format-aware correction
    std::string format;       // matched layout name
    int corrections = 0;      // characters changed by correction
    float confidence = 0.f;   // lowest per-character probability
    bool format_match = false;
};

// Beyond this many changes a "correction" is more likely to invent a plate
// than to fix one.
inline constexpr int kMaxCorrections = 2;

inline PlateResult interpret(const std::vector<DecodedChar>& chars) {
    PlateResult r;
    r.confidence = chars.empty() ? 0.f : 1.f;
    for (const auto& dc : chars) {
        r.raw += dc.c;
        r.confidence = std::min(r.confidence, dc.prob);
    }
    r.text = r.raw;

    int best_edits = kMaxCorrections + 1;
    for (const auto& f : ca_formats()) {
        std::string fixed;
        const int e = fit_pattern(r.raw, f.pattern, fixed);
        if (e >= 0 && e < best_edits) {  // strict '<' keeps priority order on ties
            best_edits = e;
            r.text = fixed;
            r.format = f.name;
            r.corrections = e;
            r.format_match = true;
        }
    }

    if (!r.format_match) {
        // Personalized ("vanity") plates are 2-7 characters with no fixed
        // layout, so they are reported unchanged.
        if (r.raw.size() >= 2 && r.raw.size() <= 7)
            r.format = "Unrecognized layout (possibly personalized)";
        else
            r.format = "Invalid length for a California plate";
    }
    return r;
}

}  // namespace calpr
