# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""California layout check and correction -- Python twin of the format part of
include/calpr/plate_decoder.hpp (same layouts, same substitutions, same rules),
used by training/tools/run_ncnn.py. tests/test_preprocess.py runs the same
cases as tests/test_decoder.cpp against it.

  D = digit, L = any letter,
  R = letter except I, O, Q (not used as the 1st and 3rd letter of the
      3-letter block on standard passenger plates)
"""
from __future__ import annotations

from dataclasses import dataclass

# Order = tie-break priority (most common first).
CA_FORMATS = [
    ("Passenger (2026+, 123ABC1)", "DDDRLRD"),
    ("Passenger (1980-2026, 1ABC234)", "DRLRDDD"),
    ("Commercial (1A23456)", "DLDDDDD"),
    ("Trailer (1AB1234)", "DLLDDDD"),
]

MAX_CORRECTIONS = 2

_LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
                    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}


def _digit_to_letter(c: str, restricted: bool) -> str | None:
    if restricted and c == "0":
        return "D"  # O is not allowed there
    if restricted and c == "1":
        return None  # I is not allowed there
    return _DIGIT_TO_LETTER.get(c)


def fit_pattern(s: str, pattern: str) -> tuple[int, str]:
    """Force `s` into `pattern`. Returns (corrections, fixed) or (-1, s)."""
    if len(s) != len(pattern):
        return -1, s
    fixed, edits = list(s), 0
    for i, (c, p) in enumerate(zip(s, pattern)):
        if p == "D":
            if not c.isdigit():
                d = _LETTER_TO_DIGIT.get(c)
                if d is None:
                    return -1, s
                fixed[i], edits = d, edits + 1
        elif p == "L":
            if not c.isalpha():
                l = _digit_to_letter(c, False)
                if l is None:
                    return -1, s
                fixed[i], edits = l, edits + 1
        elif p == "R":
            if c.isalpha():
                if c in "IOQ":
                    if c == "I":
                        return -1, s
                    fixed[i], edits = "D", edits + 1
            else:
                l = _digit_to_letter(c, True)
                if l is None:
                    return -1, s
                fixed[i], edits = l, edits + 1
        else:
            return -1, s
    return edits, "".join(fixed)


@dataclass
class PlateResult:
    raw: str
    text: str
    format: str
    corrections: int = 0
    confidence: float = 0.0
    format_match: bool = False


def interpret(raw: str, confidence: float) -> PlateResult:
    r = PlateResult(raw=raw, text=raw, format="", confidence=confidence if raw else 0.0)
    best = MAX_CORRECTIONS + 1
    for name, pattern in CA_FORMATS:
        e, fixed = fit_pattern(raw, pattern)
        if 0 <= e < best:  # strict '<' keeps priority order on ties
            best, r.text, r.format, r.corrections, r.format_match = e, fixed, name, e, True
    if not r.format_match:
        r.format = ("Unrecognized layout (possibly personalized)" if 2 <= len(raw) <= 7
                    else "Invalid length for a California plate")
    return r
