// California License Plate Recognition
// Copyright 2026 Azer Ben Ali
// SPDX-License-Identifier: Apache-2.0
//
// Tiny test helper: CHECK(cond, msg) and a summary exit code.
#pragma once

#include <cmath>
#include <cstdio>

inline int g_failures = 0;
inline int g_checks = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) { std::printf("FAIL  %s  (%s:%d)\n", msg, __FILE__, __LINE__); ++g_failures; } \
        else         { std::printf("ok    %s\n", msg); }                   \
    } while (0)

inline bool near(double a, double b, double tol = 1e-3) { return std::fabs(a - b) <= tol; }

inline int test_summary() {
    std::printf("\n%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
