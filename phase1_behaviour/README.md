# Phase 1 — Behaviour / Gaze Detector

YOLOv11s bounding-box binary classifier: **True-Gaze** vs **False-Gaze**.

## Dataset

Custom dataset, both authors as driver models (Mercedes ML250, iPad M1), across varied lighting
(Fair/Mid/Dark) and facial conditions (Clean/Bearded/Spectacles/Sunglasses).

The dataset went through several revisions during the project:

| Revision | Size | Notes |
|---|---|---|
| Initial | 192 images | Imbalanced, spectacles-dominant bias |
| Improved | 1,565 images | 1:1 balanced, 3× augmentation |
| **Final (Revised)** | **3,254 images** | ~540/driver, 1:1 balanced across 6 sub-scenarios |

Final class balance (Revised):

```
False-Gaze-without-specs = 67   (×driver set)
True-Gaze-with-specs     = 135
False-Gaze-with-specs    = 67
Looking-down-with-specs  = 67
Looking-down-without-specs = 67
True-Gaze-without-specs  = 135
```

> Note: slides variously state the final total as 3,254 and 3,300 — this hasn't been
> reconciled. Use whichever the actual Roboflow export confirms.

## Augmentation (final/revised run)

- Resize: stretch to 512×512, 3 outputs per training example
- Flip: horizontal
- Rotation: ±5°
- Hue: ±6° · Saturation: ±10% · Brightness: ±20% · Exposure: ±10%
- Blur: up to 0.8px · Noise: up to 0.02% of pixels

## Training command

```
yolo detect train ^
model=yolo11s.pt ^
data="<path>\Phase 1\Dataset\Dataset 4\data.yaml" ^
imgsz=640 ^
epochs=180 ^
batch=8 ^
optimizer=AdamW ^
lr0=0.0018 ^
lrf=0.01 ^
momentum=0.937 ^
weight_decay=0.0007 ^
warmup_epochs=5 ^
degrees=0 ^
translate=0.04 ^
scale=0.15 ^
fliplr=0 ^
hsv_h=0 ^
hsv_s=0 ^
hsv_v=0 ^
mosaic=0.12 ^
mixup=0 ^
close_mosaic=20 ^
patience=30 ^
device=0
```

> **Correction from an earlier session summary:** this exact command (imgsz=640, epochs=180,
> lr0=0.0018, mosaic=0.12) was previously mis-attributed to the Phase 3 drowsiness detector.
> The slide it actually appears on has `data=".../Phase 1/Dataset/Dataset 4/data.yaml"` — it's
> the Phase 1 command. Flagging this so it doesn't propagate further.

## Results (canonical — matches the paper)

Checkpoint: `Phase 1\Dataset\runs\detect\final-train\weights\best.pt`

| Metric | All | False-Gaze | True-Gaze |
|---|---|---|---|
| Precision | 0.9255 | 0.8977 | 0.9533 |
| Recall | 0.9463 | 0.8927 | 1.0000 |
| mAP@0.50 | **0.9557** | 0.9237 | 0.9877 |
| mAP@0.50:0.95 | 0.8059 | 0.7105 | 0.9013 |

Params: 9,428,566 · Size: 18.29 MB · Inference: 1.94 ms/image (RTX 5070 Ti)

**This is the number the paper reports.** Two other mAP@0.50 figures show up on earlier slides
(0.987 for the 1,565-image dataset, 0.976 for a training-time validation pass on the 3,254-image
set) — those are from earlier dataset iterations / training-time snapshots, not this final
benchmarked checkpoint. Use 0.9557 as canonical.
