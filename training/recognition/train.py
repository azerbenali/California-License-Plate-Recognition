# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Train the California LPRNet recognizer with CTC loss.

Run from the repository root:

    python -m training.recognition.train --train data/rec/train --val data/rec/val

Several training folders can be mixed, each repeated N times per epoch, e.g.
a handful of real crops together with synthetic plates:

    python -m training.recognition.train --train data/rec/train data/synth/train \
        --repeat 200 1 --val data/rec/val --init runs/synth/best.pt

Works on CUDA (NVIDIA), Apple Silicon (MPS) and CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from ..common import config as C
from ..common.preprocess import greedy_decode
from .dataset import PlateDataset, collate
from .lprnet import LPRNet, ctc_log_probs


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    exact, total, char_err, char_total = 0, 0, 0, 0
    for images, _, _, texts in loader:
        logits = model(images.to(device)).float().cpu().numpy()
        for lg, truth in zip(logits, texts):
            pred = greedy_decode(lg)
            exact += int(pred == truth)
            total += 1
            char_err += edit_distance(pred, truth)
            char_total += len(truth)
    return {"accuracy": exact / max(1, total), "cer": char_err / max(1, char_total)}


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict) -> None:
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "charset": C.CHARSET,
        "input_size": [C.REC_INPUT_H, C.REC_INPUT_W],
        "band": [C.BAND_TOP, C.BAND_BOTTOM],
    }, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, nargs="+",
                    help="one or more folders of labeled training crops")
    ap.add_argument("--repeat", type=int, nargs="+", default=None,
                    help="times each --train folder is used per epoch (default 1 each); "
                         "use it to oversample a small set of real plates")
    ap.add_argument("--val", required=True, help="folder of labeled validation crops")
    ap.add_argument("--out", default="runs/recognizer", help="output folder")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    ap.add_argument("--no-band", action="store_true",
                    help="train on the whole plate instead of the character band "
                         "(then run the C++ app with --no-crop)")
    ap.add_argument("--init", default="",
                    help="start from these weights (e.g. synthetic warm-up) with a fresh schedule")
    ap.add_argument("--resume", default="",
                    help="continue an interrupted run (same data) from this checkpoint")
    args = ap.parse_args()
    if args.init and args.resume:
        raise SystemExit("Use either --init or --resume, not both")

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    use_band = not args.no_band
    repeat = args.repeat or [1] * len(args.train)
    if len(repeat) != len(args.train) or min(repeat) < 1:
        raise SystemExit("--repeat needs one positive number per --train folder")
    parts = []
    for folder, n in zip(args.train, repeat):
        ds = PlateDataset(folder, augment=True, use_band=use_band)
        print(f"train: {folder}: {len(ds)} plates x {n}")
        parts += [ds] * n
    train_ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    val_ds = PlateDataset(args.val, augment=False, use_band=use_band)
    print(f"train: {len(train_ds)} samples per epoch, val: {len(val_ds)} plates")
    pin = device.type == "cuda"
    # drop_last keeps batch statistics stable, but would skip a set smaller than a batch.
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=len(train_ds) > args.batch,
                          num_workers=args.workers, collate_fn=collate, pin_memory=pin,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=pin)

    model = LPRNet(C.NUM_CLASSES).to(device)
    start_epoch, best_acc = 0, -1.0
    if args.init or args.resume:
        ckpt = torch.load(args.init or args.resume, map_location="cpu")
        if ckpt.get("charset") != C.CHARSET:
            raise SystemExit("Checkpoint charset differs from training/common/config.py")
        model.load_state_dict(ckpt["model"])
        if args.resume:
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_acc = float(ckpt.get("metrics", {}).get("accuracy", -1.0))
            print(f"resumed from {args.resume} at epoch {start_epoch}")
        else:
            print(f"initialized weights from {args.init}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for group in optimizer.param_groups:  # needed by LambdaLR when resuming
        group.setdefault("initial_lr", args.lr)
    steps_per_epoch = max(1, len(train_dl))
    total_steps = args.epochs * steps_per_epoch
    warmup = min(1000, total_steps // 20)

    def lr_lambda(step: int) -> float:  # linear warmup, then cosine decay
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda,
                                                  last_epoch=start_epoch * steps_per_epoch - 1)
    ctc = nn.CTCLoss(blank=C.BLANK_INDEX, reduction="mean", zero_infinity=True)
    # CTC loss is not implemented on MPS; compute it on the CPU in that case.
    loss_device = torch.device("cpu") if device.type == "mps" else device
    history = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for images, targets, lengths, _ in train_dl:
            images = images.to(device, non_blocking=True)
            log_probs = ctc_log_probs(model(images)).to(loss_device)
            T, B = log_probs.shape[:2]
            input_lengths = torch.full((B,), T, dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        metrics = evaluate(model, val_dl, device)
        metrics["loss"] = running / steps_per_epoch
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, **metrics})
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {metrics['loss']:.4f}  "
              f"val acc {metrics['accuracy']:.4f}  cer {metrics['cer']:.4f}  "
              f"lr {metrics['lr']:.2e}  {time.time() - t0:.0f}s")

        save_checkpoint(out / "last.pt", model, epoch, metrics)
        if metrics["accuracy"] >= best_acc:  # ties: keep the later, better-trained epoch
            best_acc = metrics["accuracy"]
            save_checkpoint(out / "best.pt", model, epoch, metrics)
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"best val accuracy: {best_acc:.4f}  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
