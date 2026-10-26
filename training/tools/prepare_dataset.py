# California License Plate Recognition
# Copyright 2026 Azer Ben Ali
# SPDX-License-Identifier: Apache-2.0
"""Turn a folder of labeled photos into the detector and recognizer datasets.

    python -m training.tools.prepare_dataset --src samples --out data

Source layout (the `samples/` folder of this repository is an example):

    <src>/images/*.jpg        photos
    <src>/labels/*.txt        YOLO pose labels, one line per plate:
                              0 cx cy w h  x_tl y_tl 2  x_tr y_tr 2  x_br y_br 2  x_bl y_bl 2
    <src>/plates.csv          image,index,text   (index = line number in the label file)

Output:

    <out>/det/images/{train,val}   <out>/det/labels/{train,val}   -> detector training
    <out>/rec/{train,val}/*.png + labels.csv                      -> recognizer training

Images are split by photo (never the same photo in train and val). With fewer
than --min-split photos there is nothing to hold out, so validation reuses the
training photos: fine for checking that the pipeline runs, meaningless as an
accuracy figure.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from .make_rec_crops import IMAGE_EXTS, make_crops, read_texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with images/, labels/ and plates.csv")
    ap.add_argument("--out", default="data", help="output root (default: data)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--min-split", type=int, default=10,
                    help="below this many photos, validation reuses the training photos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="copy each training photo N times into det/images/train, so a "
                         "tiny smoke-test set gives the detector more steps per epoch")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    photos = sorted(p for p in (src / "images").iterdir()
                    if p.suffix.lower() in IMAGE_EXTS and (src / "labels" / f"{p.stem}.txt").exists())
    if not photos:
        raise SystemExit(f"No labeled photos in {src}/images (each needs {src}/labels/<name>.txt)")

    if len(photos) < args.min_split:
        split = {"train": photos, "val": photos}
        print(f"warning: only {len(photos)} labeled photo(s); validation reuses the training "
              f"photos (smoke test only, not an accuracy measure)")
    else:
        shuffled = photos[:]
        random.Random(args.seed).shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * args.val_fraction))
        split = {"train": sorted(shuffled[n_val:]), "val": sorted(shuffled[:n_val])}

    texts = read_texts(str(src / "plates.csv")) if (src / "plates.csv").exists() else {}
    for part, items in split.items():
        img_dir, lbl_dir = out / "det" / "images" / part, out / "det" / "labels" / part
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for p in items:
            copies = args.repeat if part == "train" else 1
            for k in range(copies):
                stem = p.stem if k == 0 else f"{p.stem}_r{k}"
                shutil.copy2(p, img_dir / f"{stem}{p.suffix}")
                shutil.copy2(src / "labels" / f"{p.stem}.txt", lbl_dir / f"{stem}.txt")
        # Recognizer crops: one per plate, from the original photos.
        n = make_crops(src / "images", src / "labels", out / "rec" / part, texts,
                       only={p.name for p in items})
        print(f"{part}: {len(items)} photo(s) -> {out / 'det'}, {n['crops']} plate crop(s) -> "
              f"{out / 'rec' / part}" + (f"  ({n['missing_text']} without text)" if n["missing_text"] else ""))


if __name__ == "__main__":
    main()
