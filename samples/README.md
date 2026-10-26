# Sample photos

Three labeled photos of California plates, used to check the whole system
end to end (labeling format → dataset tools → training → ncnn export →
detection and reading). They are also the start of a real dataset: add your
own photos here in the same format.

| File | Plate | Layout | Design |
|---|---|---|---|
| `images/ca_plate_01.jpg` | `9EPW746` | Passenger 1980–2026 (`1ABC234`) | black plate, white characters |
| `images/ca_plate_02.jpg` | `201AAA1` | Passenger 2026+ (`123ABC1`) | standard white plate, at night |
| `images/ca_plate_03.jpg` | `PERC50` | Personalized | black legacy plate, yellow characters, seen at an angle |

## Format

- `images/` — photos (`.jpg`, `.png`, `.webp`, …).
- `labels/<photo name>.txt` — one line per plate, YOLO pose format,
  all values normalized to 0..1 by the image width/height:

  ```
  0 cx cy w h  x_tl y_tl 2  x_tr y_tr 2  x_br y_br 2  x_bl y_bl 2
  ```

  `0` is the class (plate), `cx cy w h` the box, then the four plate corners
  in the order top-left, top-right, bottom-right, bottom-left, each followed
  by its visibility (`2` = visible). Label the corners of the plate itself;
  when a frame hides the edge, follow the frame's inner edge.
- `plates.csv` — the plate text, `image,index,text`, where `index` is the
  line number (from 0) of the plate in the label file.

Any annotation tool with keypoint support can produce the labels, e.g. CVAT,
Label Studio or Roboflow (export as "YOLO pose" with 4 keypoints).

## Use

```bash
python -m training.tools.prepare_dataset --src samples --out data
```

builds `data/det/` (detector) and `data/rec/` (recognizer crops). With fewer
than 10 photos there is nothing to hold out, so validation reuses the
training photos: enough to check that everything runs, not to measure
accuracy.
