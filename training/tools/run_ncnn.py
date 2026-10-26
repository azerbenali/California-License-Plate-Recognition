# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Run the exported ncnn models on photos from Python -- a quick check of
freshly trained models without building the C++ app.

    pip install ncnn
    python -m training.tools.run_ncnn samples/images \\
        --det models/model.ncnn.param --rec models/ca_lprnet.ncnn.param --out runs/demo

Same steps as the C++ pipeline (letterbox, YOLO pose decoding with NMS, corner
rectification, character band, LPRNet, California layout check), so the
results should match calpr_app up to small resize differences. `--det` /
`--rec` take the .param path; the .bin next to it is loaded too. Without
--det every image is treated as an already cropped plate.

Writes <out>/<name>.jpg (annotated), <out>/<name>_plate<i>.png (rectified
plate) and prints one line per plate; --json writes <out>/results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..common import config as C
from ..common.plate_format import interpret
from ..common.preprocess import order_corners, rec_input, rectify

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_net(param: str, threads: int):
    import ncnn  # pip install ncnn

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads
    bin_path = str(Path(param).with_suffix(".bin"))
    if net.load_param(param) != 0 or net.load_model(bin_path) != 0:
        raise SystemExit(f"Cannot load ncnn model {param} / {bin_path}")
    return net


def run_net(net, x: np.ndarray, inp: str = "in0", out: str = "out0") -> np.ndarray:
    import ncnn

    # ncnn.Mat(array) shares the array's memory, so `arr` must stay alive until
    # extract() has run; clone() gives the Mat its own copy to be safe.
    arr = np.ascontiguousarray(x, dtype=np.float32)
    mat = ncnn.Mat(arr).clone()
    ex = net.create_extractor()
    ex.input(inp, mat)
    ret, y = ex.extract(out)
    if ret != 0:
        raise RuntimeError(f"ncnn extract failed for blob {out}")
    return np.array(y).copy()


def letterbox(img: np.ndarray, size: int):
    """Same as calpr::Letterbox: keep aspect, pad centered with 114. Returns CHW RGB 0..1."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), C.DET_PAD_VALUE, np.float32)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0
    return x, scale, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order, keep = list(np.argsort(-scores)), []
    while order:
        i = order.pop(0)
        keep.append(i)
        rest = []
        for j in order:
            xx0, yy0 = max(boxes[i, 0], boxes[j, 0]), max(boxes[i, 1], boxes[j, 1])
            xx1, yy1 = min(boxes[i, 2], boxes[j, 2]), min(boxes[i, 3], boxes[j, 3])
            inter = max(0.0, xx1 - xx0) * max(0.0, yy1 - yy0)
            area = lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area(boxes[i]) + area(boxes[j]) - inter
            if union <= 0 or inter / union <= iou_thr:
                rest.append(j)
        order = rest
    return keep


def decode_pose(out: np.ndarray, scale: float, left: int, top: int, img_w: int, img_h: int,
                conf: float, iou: float, k: int = C.DET_NUM_KEYPOINTS) -> list[dict]:
    """Mirror of calpr::decode_yolo_pose (classic and end-to-end heads)."""
    m = out.reshape(out.shape[-2], out.shape[-1]) if out.ndim > 2 else out
    small, large = min(m.shape), max(m.shape)
    if m.shape[0] != small:
        m = m.T  # attributes x candidates
    if small == 6 + 3 * k and large <= 1000:
        end_to_end, nc = True, 1
    elif small - 4 - 3 * k >= 1 and large > 1000:
        end_to_end, nc = False, small - 4 - 3 * k
    else:
        raise SystemExit(f"Unexpected detector output shape {out.shape}")

    def src(x, y):
        return (np.clip((x - left) / scale, 0, img_w - 1), np.clip((y - top) / scale, 0, img_h - 1))

    dets = []
    base = 6 if end_to_end else 4 + nc
    for i in range(m.shape[1]):
        a = m[:, i]
        if end_to_end:
            score, (x0, y0, x1, y1) = float(a[4]), a[:4]
        else:
            score = float(a[4:4 + nc].max())
            cx, cy, bw, bh = a[:4]
            x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        if score < conf:
            continue
        corners = np.array([src(a[base + 3 * j], a[base + 3 * j + 1]) for j in range(k)], np.float32)
        dets.append({"score": score, "box": np.array([*src(x0, y0), *src(x1, y1)], np.float32),
                     "corners": corners, "corner_conf": float(min(a[base + 3 * j + 2] for j in range(k)))})
    if end_to_end:
        return sorted(dets, key=lambda d: -d["score"])
    if not dets:
        return []
    keep = nms(np.stack([d["box"] for d in dets]), np.array([d["score"] for d in dets]), iou)
    return [dets[i] for i in keep]


def recognize(net, plate: np.ndarray, use_band: bool = True):
    logits = run_net(net, rec_input(plate, use_band))
    logits = logits.reshape(logits.shape[-2], logits.shape[-1])
    if logits.shape[0] != C.NUM_CLASSES:
        logits = logits.T
    if logits.shape[0] != C.NUM_CLASSES:
        raise SystemExit(f"Recognizer output {logits.shape} does not have {C.NUM_CLASSES} classes")
    # Greedy CTC with the weakest character's softmax probability as confidence.
    p = np.exp(logits - logits.max(axis=0, keepdims=True))
    p /= p.sum(axis=0, keepdims=True)
    best = logits.argmax(axis=0)
    text, probs, prev = [], [], C.BLANK_INDEX
    for t, k in enumerate(best):
        if k != C.BLANK_INDEX:
            if k != prev:
                text.append(C.CHARSET[k])
                probs.append(float(p[k, t]))
            elif probs:
                probs[-1] = max(probs[-1], float(p[k, t]))
        prev = k
    return interpret("".join(text), min(probs) if probs else 0.0)


def draw(img: np.ndarray, quad: np.ndarray, box: np.ndarray, label: str, ok: bool) -> None:
    color = (60, 200, 60) if ok else (0, 165, 255)
    cv2.polylines(img, [quad.astype(np.int32)], True, color, max(2, img.shape[1] // 500), cv2.LINE_AA)
    scale = max(0.6, img.shape[1] / 1400.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = max(0, int(box[0])), max(th + 6, int(box[1]) - 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 4), color, cv2.FILLED)
    cv2.putText(img, label, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--rec", required=True, help="recognizer .param (e.g. models/ca_lprnet.ncnn.param)")
    ap.add_argument("--det", default="", help="detector .param (e.g. models/model.ncnn.param)")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-corner-conf", type=float, default=0.5)
    ap.add_argument("--no-crop", action="store_true", help="feed the whole plate, not the band")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rec = load_net(args.rec, args.threads)
    det = load_net(args.det, args.threads) if args.det else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for p in map(Path, args.inputs):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]

    results = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{f}: cannot read")
            continue
        h, w = img.shape[:2]
        if det is not None:
            x, scale, left, top = letterbox(img, C.DET_INPUT_SIZE)
            dets = decode_pose(run_net(det, x), scale, left, top, w, h, args.conf, args.iou)
        else:
            full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
            dets = [{"score": 1.0, "box": np.array([0, 0, w - 1, h - 1], np.float32),
                     "corners": full, "corner_conf": 1.0}]

        vis, plates = img.copy(), []
        for i, d in enumerate(dets):
            use_kp = d["corner_conf"] >= args.min_corner_conf
            b = d["box"]
            quad = order_corners(d["corners"]) if use_kp else \
                np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
            plate = rectify(img, quad)
            r = recognize(rec, plate, not args.no_crop)
            cv2.imwrite(str(out / f"{f.stem}_plate{i}.png"), plate)
            draw(vis, quad, b, r.text or "?", r.format_match)
            plates.append({"text": r.text, "raw": r.raw, "format": r.format,
                           "format_match": r.format_match, "rec_conf": round(r.confidence, 4),
                           "det_conf": round(d["score"], 4),
                           "corner_conf": round(d["corner_conf"], 4), "used_corners": bool(use_kp),
                           "corners": np.round(quad.astype(np.float64), 1).tolist()})
            print(f"{f.name}: {r.text}  ({r.format}, rec {r.confidence:.2f}, det {d['score']:.2f}, "
                  + ("corners" if use_kp else f"box, corner conf {d['corner_conf']:.2f}")
                  + (f", raw {r.raw}" if r.corrections else "") + ")")
        if not plates:
            print(f"{f.name}: no plate found")
        cv2.imwrite(str(out / f"{f.stem}.jpg"), vis)
        results.append({"image": f.name, "plates": plates})

    if args.json:
        (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"annotated images and plate crops in {out}")


if __name__ == "__main__":
    main()
