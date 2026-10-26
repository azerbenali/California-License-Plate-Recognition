# Models

## Your trained models

Put the exported ncnn models here (they are git-ignored):

| File | Made by | App option |
|---|---|---|
| `ca_lprnet.ncnn.param`, `ca_lprnet.ncnn.bin` | `python -m training.recognition.export_ncnn` | `--rec-param`, `--rec-bin` |
| `model.ncnn.param`, `model.ncnn.bin` (from `best_ncnn_model/`) | `python -m training.detection.train_detector --export ...` | `--det-param`, `--det-bin` |

Both exports use the blob names `in0` / `out0`, the app's defaults.

A recognizer trained with a different character set does **not** work here:
the app checks that the output has the 37 classes of the California charset
and refuses any other model.

## `demo/` — smoke-test recognizer

`demo/ca_lprnet.ncnn.param/.bin` is an LPRNet recognizer produced by running
this repository's training pipeline once: one epoch on 20,000 synthetic
plates, then three epochs mixing them with the 3 real plates in `samples/`.
It reads 99.2 % of 1,000 held-out synthetic plates and the 3 sample plates.

It proves that training, export and inference work end to end; it is **not**
a production model. Train on your own labeled photos (hundreds at least)
before relying on any result. No demo detector is included: train one with
`training/detection/train_detector.py` (see the main README).
