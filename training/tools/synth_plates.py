# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Render synthetic California-style plates for pretraining and smoke tests.

    python -m training.tools.synth_plates --out data/synth/train --count 20000
    python -m training.tools.synth_plates --out data/synth/val --count 2000 --seed 1

Images are 240x120 rectified plates named <TEXT>_<n>.png, so PlateDataset
reads the label from the file name. Besides the standard white plate they
include the dark designs seen on real cars (black with white or yellow
characters, the blue and yellow legacy plates), month/year stickers and
dealer frames. They use generic fonts, not the real plate typeface, so treat
them as a warm-up: fine-tune on real plate crops before deploying.
"""
from __future__ import annotations

import argparse
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..common import config as C

LETTERS = string.ascii_uppercase
NO_IOQ = [c for c in LETTERS if c not in "IOQ"]
DIGITS = string.digits

# Bold sans fonts, condensed ones first (closest to the plate typeface).
# Whichever of these exist are used, picked at random per plate.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/ARIALNB.TTF",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]

# Plate designs (RGB): background, characters, top line, top line style, weight.
SCHEMES = {
    "standard":      ((242, 242, 242), (25, 45, 110), (190, 30, 40), "script", 0.55),
    "black_white":   ((24, 24, 27), (236, 236, 236), (236, 236, 236), "script", 0.15),
    "legacy_black":  ((22, 22, 24), (232, 200, 85), (232, 200, 85), "caps", 0.12),
    "legacy_blue":   ((28, 44, 112), (236, 200, 72), (236, 200, 72), "caps", 0.09),
    "legacy_yellow": ((236, 196, 64), (22, 22, 22), (22, 22, 22), "caps", 0.09),
}


def random_plate_text(rng: random.Random) -> str:
    """Plate number following the California layouts (plus some vanity plates)."""
    r = rng.random()
    d = lambda: rng.choice(DIGITS)
    l = lambda: rng.choice(LETTERS)
    rl = lambda: rng.choice(NO_IOQ)
    if r < 0.35:   # 2026+ passenger: 123ABC1
        return d() + d() + d() + rl() + l() + rl() + d()
    if r < 0.75:   # 1980-2026 passenger: 1ABC234
        return rng.choice("123456789") + rl() + l() + rl() + d() + d() + d()
    if r < 0.85:   # commercial: 1A23456
        return d() + rl() + d() + d() + d() + d() + d()
    if r < 0.92:   # trailer: 1AB1234
        return d() + rl() + rl() + d() + d() + d() + d()
    n = rng.randint(2, 7)  # personalized
    return "".join(rng.choice(LETTERS + DIGITS) for _ in range(n))


def available_fonts(extra: list[str] | None = None) -> list[str]:
    fonts = [p for p in (extra or []) if Path(p).exists()]
    return fonts or [p for p in FONT_CANDIDATES if Path(p).exists()]


def load_font(path: str | None, size: int):
    if path and Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def jitter(color: tuple[int, int, int], rng: random.Random, amount: int = 14) -> tuple[int, ...]:
    return tuple(int(np.clip(c + rng.uniform(-amount, amount), 0, 255)) for c in color)


def pick_scheme(rng: random.Random) -> str:
    names = list(SCHEMES)
    return rng.choices(names, weights=[SCHEMES[n][4] for n in names])[0]


def draw_text_fit(img: Image.Image, text: str, font_path: str | None, box, color) -> None:
    """Draw `text` stretched to fill `box` (x0, y0, x1, y1), like the tall,
    narrow characters of a real plate, whatever font is used."""
    x0, y0, x1, y1 = (int(v) for v in box)
    font = load_font(font_path, 96)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bx0, by0, bx1, by1 = probe.textbbox((0, 0), text, font=font)
    mask = Image.new("L", (bx1 - bx0 + 4, by1 - by0 + 4), 0)
    ImageDraw.Draw(mask).text((2 - bx0, 2 - by0), text, font=font, fill=255)
    mask = mask.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.LANCZOS)
    img.paste(Image.new("RGB", mask.size, color), (x0, y0), mask)


def render_plate(text: str, rng: random.Random, font_path: str | None = None,
                 w: int = 480, h: int = 240, scheme: str | None = None,
                 fonts: list[str] | None = None) -> np.ndarray:
    """Render one plate at 2x and downscale to the rectified size (BGR uint8).

    `scheme` is a key of SCHEMES (random if None). `font_path` forces a font;
    otherwise one of `fonts` (default: available_fonts()) is picked at random.
    """
    scheme = scheme or pick_scheme(rng)
    bg_c, char_c, top_c, top_style, _ = SCHEMES[scheme]
    if font_path is None:
        fonts = fonts if fonts is not None else available_fonts()
        font_path = rng.choice(fonts) if fonts else None

    img = Image.new("RGB", (w, h), jitter(bg_c, rng, 10))
    draw = ImageDraw.Draw(img)

    # Embossed border
    m = int(h * 0.04)
    edge = tuple(int(c * 0.6) for c in char_c) if scheme == "standard" else jitter(char_c, rng)
    draw.rounded_rectangle([m, m, w - m, h - m], radius=int(h * 0.08),
                           outline=edge, width=max(2, h // 60))

    # State name above the character band
    top = "California" if top_style == "script" else "CALIFORNIA"
    tw = w * (rng.uniform(0.40, 0.52) if top_style == "script" else rng.uniform(0.34, 0.46))
    th = h * (0.14 if top_style == "script" else 0.08)
    ty = h * (0.06 if top_style == "script" else 0.08)
    draw_text_fit(img, top, font_path, ((w - tw) / 2, ty, (w + tw) / 2, ty + th), jitter(top_c, rng))

    # Registration stickers in the top corners
    if rng.random() < 0.6:
        for x in (m * 3, w - m * 3 - w * 0.13):
            col = rng.choice([(40, 70, 170), (230, 200, 40), (200, 40, 50), (240, 240, 240),
                              (60, 150, 70), (230, 120, 40)])
            draw.rectangle([x, h * 0.08, x + w * 0.13, h * 0.19], fill=col)

    # Main characters, filling most of the character band
    band_top, band_bot = h * C.BAND_TOP, h * C.BAND_BOTTOM
    band_h = band_bot - band_top
    ch = band_h * rng.uniform(0.62, 0.80)
    per_char = rng.uniform(0.105, 0.125) * w
    cw = min(w * 0.88, per_char * len(text) * rng.uniform(0.95, 1.05))
    cx = w / 2 + rng.uniform(-0.03, 0.03) * w
    cy = band_top + band_h * rng.uniform(0.44, 0.52)
    draw_text_fit(img, text, font_path,
                  (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2), jitter(char_c, rng))

    # Dealer frame (covers the bottom edge and a bit of the top), or a footer
    if rng.random() < 0.3:
        frame_c = jitter(rng.choice([(20, 20, 20), (200, 200, 200), (120, 30, 30)]), rng)
        draw.rectangle([0, h * 0.89, w, h], fill=frame_c)
        draw.rectangle([0, 0, w, h * 0.05], fill=frame_c)
        draw.rectangle([0, 0, w * 0.035, h], fill=frame_c)
        draw.rectangle([w * 0.965, 0, w, h], fill=frame_c)
    elif scheme == "standard":
        fw = w * 0.22
        draw_text_fit(img, "dmv.ca.gov", font_path,
                      ((w - fw) / 2, h * 0.88, (w + fw) / 2, h * 0.93), edge)

    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return cv2.resize(arr, (C.PLATE_WARP_W, C.PLATE_WARP_H), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--font", action="append", default=None,
                    help="TTF font file (bold sans works best); repeat to use several")
    ap.add_argument("--scheme", choices=list(SCHEMES), default=None,
                    help="render only this plate design (default: mix of all)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    fonts = available_fonts(args.font)
    print(f"fonts: {', '.join(Path(f).name for f in fonts) or 'PIL default'}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        text = random_plate_text(rng)
        img = render_plate(text, rng, scheme=args.scheme, fonts=fonts)
        cv2.imwrite(str(out / f"{text}_{i:06d}.png"), img)
    print(f"wrote {args.count} synthetic plates to {out}")


if __name__ == "__main__":
    main()
