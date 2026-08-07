# Phase 2 — Prohibited Object Detector

YOLO bounding-box single-class detector: **True-Object** (phone-in-hand).

## Dataset

Custom dataset, both authors as driver models (Chevrolet Malibu, iPad M1). Captured as 4
sub-scenarios, merged into one class for training:

```
Black Phone — Right Hand : 175
Black Phone — Left Hand  : 175
White Phone — Right Hand : 175
White Phone — Left Hand  : 175
Total = 700 images
```

## Two trained variants

Two backbone versions were trained on (nominally) the same task — checkpoints and reported
numbers differ, and the exact command that produced each is not fully documented:

| Checkpoint | Path | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|---|
| YOLOv8s (custom) | `Phase 2\Dataset V8\...\best.pt` | 0.9649 | 1.0000 | 0.9940 | 0.7949 |
| YOLOv11s (custom) | `Phase 2\Dataset\...\best.pt` | 0.9654 | 0.9633 | 0.9758 | 0.7926 |

> A separate comparison slide reports *different* numbers for what's labeled the same v8/v11
> comparison (mAP@0.50: v11=0.993, v8=0.991; mAP@0.50:0.95: v11=0.840, v8=0.883) — this doesn't
> match the table above. Likely a different benchmark pass or dataset split. **Needs
> reconciling** — both can't be "the" v8 vs v11 result.

## Training command (documented — variant unclear)

```
model=yolo11s.pt
imgsz=512
epochs=100
batch=16
augment=False
mosaic=0.5
patience=15
optimizer=AdamW
lr_find=True
```

It's not documented whether this specific command produced the V8 checkpoint, the V11
checkpoint, or a third run entirely.

## Which checkpoint the demo scripts actually use

- `pipeline/pipeline.py` loads the **custom V11** checkpoint above.
- `pipeline/pipeline_object.py` loads **generic pretrained `yolov8s.pt`** (stock COCO weights,
  using COCO's built-in "cell phone" class) — **not** either custom checkpoint trained here.

Worth deciding which pipeline script represents the "real" demo before publishing.
