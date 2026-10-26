# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Export a trained recognizer checkpoint to ncnn (and optionally ONNX).

    python -m training.recognition.export_ncnn --weights runs/recognizer/best.pt --out models

Produces models/ca_lprnet.ncnn.param and models/ca_lprnet.ncnn.bin with blob
names in0 / out0, which are the C++ app's defaults.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from ..common import config as C
from .lprnet import LPRNet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="checkpoint from train.py")
    ap.add_argument("--out", default="models", help="output folder")
    ap.add_argument("--name", default="ca_lprnet")
    ap.add_argument("--onnx", action="store_true", help="also write an ONNX file")
    ap.add_argument("--fp32", action="store_true", help="store ncnn weights as fp32 (default fp16)")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    if ckpt.get("charset") != C.CHARSET:
        raise SystemExit("Checkpoint charset differs from training/common/config.py")
    model = LPRNet(C.NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, 3, C.REC_INPUT_H, C.REC_INPUT_W)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, C.NUM_CLASSES, 18), f"unexpected output shape {tuple(y.shape)}"

    import pnnx  # pip install pnnx

    # pnnx also writes TorchScript, pnnx and Python files next to its output;
    # keep those in a temporary folder and copy only the two ncnn files.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pnnx.export(model, str(tmp / f"{args.name}.pt"), x,
                    ncnnparam=str(tmp / f"{args.name}.ncnn.param"),
                    ncnnbin=str(tmp / f"{args.name}.ncnn.bin"),
                    fp16=not args.fp32)
        for ext in (".ncnn.param", ".ncnn.bin"):
            shutil.copy2(tmp / f"{args.name}{ext}", out / f"{args.name}{ext}")
    print(f"ncnn model: {out / (args.name + '.ncnn.param')} / .bin  (blobs in0 -> out0)")

    if args.onnx:
        onnx_path = out / f"{args.name}.onnx"
        torch.onnx.export(model, x, str(onnx_path), input_names=["input"],
                          output_names=["logits"], opset_version=17)
        print(f"onnx model: {onnx_path}")


if __name__ == "__main__":
    main()
