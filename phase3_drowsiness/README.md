# Phase 3 — Drowsiness Detector

YOLOv11s bounding-box single-class detector: **Drowsy-True**.

## Dataset

Two sources feed this phase:

1. **Custom greyscale dataset** — 18 drivers across different nationalities/genders, ~400
   images/driver, 7,900 images total. Ratio 2:3 (Drowsy-True : null). Collected with 2
   sub-behaviours (`0: Yawning`, `1: Dropping Head`) but merged into a single `Drowsy-True`
   class for training (see `data.yaml`).
2. **NTHU CV Lab licensed dataset** (Taiwan) — 640×480, 30fps driver video across 5 scenarios
   (Bare Face, Glasses, Sunglasses, Night–Bare Face, Night–Glasses), each frame labeled
   drowsy/non-drowsy. Supplied as AVI video; frames were extracted with `extract_frames.py`
   in this folder.

## ⚠️ Open question: two different training runs exist

Two separate, fully-documented training runs for this phase were found across the project
materials, with different hyperparameters and different results. **It's not confirmed which
one's weights were actually loaded as `base_weights` into `UnifiedInCabinNet` for the
validated Hook-Tap paper result.**

### Run A — GPU, fast, output folder `yolo11s_640_final`

```
88 epochs completed in 0.886 hours
Ultralytics 8.4.14 · CUDA:0 (NVIDIA GeForce RTX 5070 Ti)
YOLO11s: 101 layers, 9,413,187 params, 21.3 GFLOPs
Class   Images  Instances  P      R      mAP50  mAP50-95
all     480     186        0.938  0.978  0.97   0.858
```

The screenshot of this run was filed as `Drowsiness_model_result_individual_used_in_Hook-tap.jpeg`
— the filename implies this is the run used in Hook-Tap, but that hasn't been confirmed.

### Run B — CPU, slow, full 120/120 epochs

```
120 epochs completed in 49.053 hours
Ultralytics 8.4.21 · CPU (Intel Core Ultra 7 265K)
YOLO11s (fused): 101 layers, 9,413,187 params, 21.3 GFLOPs
Class   Images  Instances  P      R      mAP50  mAP50-95
all     480     186        0.945  0.989  0.985  0.882
```

Training command:
```
yolo detect train ^
model=yolo11s.pt ^
data=data.yaml ^
imgsz=512 ^
epochs=120 ^
batch=16 ^
optimizer=AdamW ^
lr0=0.003 ^
lrf=0.01 ^
momentum=0.937 ^
weight_decay=0.0005 ^
warmup_epochs=3 ^
warmup_momentum=0.8 ^
augment=True ^
mosaic=0.7 ^
mixup=0.0 ^
device=0
```

This is the run presented as "the" Phase 3 result in the project slide deck (Key Metrics:
mAP@0.50=0.985, mAP@0.50:0.95=0.882, Precision=0.944, Recall=0.990, F1=0.966) and in the
separately-generated benchmark PDF (P=0.9448, R=0.9892, mAP50=0.9850, mAP50-95=0.8815).

### Why this matters

Hook-Tap's private drowsiness branch is a thin classifier bolted onto a **frozen** backbone
(see `mtl/hook_tap/README.md`) — so the paper's binary-classifier numbers (ROC-AUC=0.759 etc.)
depend entirely on which of these two backbones was frozen in. The reported Gaze/Behaviour
numbers matching Phase 1 exactly (proving the gaze head passes through untouched) don't tell us
anything about which drowsiness backbone was used, since drowsiness routes through the *private*
branch instead of the shared detection head.

**Action needed:** confirm which run's `best.pt` was passed as `--weights` to
`train_combined.py` for the numbers actually reported in the paper.
