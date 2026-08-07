# Results Summary

Consolidated metrics across all phases, pulled from the project slide deck and the standalone
benchmark report. Where multiple numbers exist for the same model, the canonical (paper-matching)
one is marked.

## Phase 1 — Behaviour / Gaze

| Source | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | Dataset size |
|---|---|---|---|---|---|
| Initial dataset (training-time) | ~0.62 (est.) | — | unreliable | unreliable | 192 |
| Improved dataset (training-time) | 0.987 | — | 0.951 | 0.943 | 1,565 |
| Revised dataset (training-time) | 0.976 | 0.801 | 0.947 | 0.975 | 3,254 |
| **Benchmark report (canonical, matches paper)** | **0.9557** | **0.8059** | **0.9255** | **0.9463** | 3,254 (final-train checkpoint) |

## Phase 2 — Object Detection

| Source | Model | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall |
|---|---|---|---|---|---|
| Comparison slide | YOLOv11s | 0.993 | 0.840 | 0.98 | 0.99 |
| Comparison slide | YOLOv8s | 0.991 | 0.883 | 0.97 | 0.97 |
| Benchmark report | YOLOv11s (custom) | 0.9758 | 0.7926 | 0.9654 | 0.9633 |
| Benchmark report | YOLOv8s (custom) | 0.9940 | 0.7949 | 0.9649 | 1.0000 |

**Unreconciled:** the comparison slide and the benchmark report disagree on both v8 and v11
numbers. Likely different benchmark passes / splits — not yet clear which is canonical.

## Phase 3 — Drowsiness

| Source | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | Epochs | Hardware |
|---|---|---|---|---|---|---|
| Run A (`yolo11s_640_final`) | 0.97 | 0.858 | 0.938 | 0.978 | 88 | GPU (RTX 5070 Ti), 0.886 hrs |
| Run B (slide deck + benchmark report) | 0.985 | 0.882 | 0.944–0.945 | 0.989–0.990 | 120/120 | CPU, 49.053 hrs |

**Unreconciled:** two separate, fully-completed training runs with different hyperparameters
(imgsz 640 vs 512, different lr0/mosaic). Not confirmed which produced the checkpoint used as
`base_weights` for Hook-Tap. See `phase3_drowsiness/README.md`.

## Hook-Tap (validated MTL result, as reported in the paper)

| Metric | Value |
|---|---|
| Gaze — Precision / Recall | 0.9255 / 0.9463 |
| Gaze — mAP@0.50 / mAP@0.50:0.95 | 0.9557 / 0.8059 |
| Drowsiness — ROC-AUC / PR-AUC | 0.759 / 0.664 |
| Drowsiness — F1 @ τ* | 0.667 |
| Drowsiness — Recall @ τ=0.60 | 97.3% (181/186) |
| Latency (baseline → Hook-Tap) | 8.94 ms → 5.93 ms/frame |
| Speedup | 1.51× (−33.6%) |
| Params / GFLOPs | 9.41M + ~0.3M / 21.3 |

The Gaze numbers here match the Phase 1 canonical benchmark exactly, which is expected — the
gaze head is unmodified in Hook-Tap. The Drowsiness numbers cannot be cross-checked the same way
(different task: binary classification, not detection) — see the open question above.

## Checkpoint paths (for the weights-in-repo decision)

```
Phase 1: Phase 1\Dataset\runs\detect\final-train\weights\best.pt
Phase 2 (V8):  Phase 2\Dataset V8\runs\detect\train2\weights\best.pt
Phase 2 (V11): Phase 2\Dataset\runs\detect\train\weights\best.pt
Phase 3 (Run A, 640, 88ep): Phase 3\runs\detect\yolo11s_640_final\weights\best.pt
Phase 3 (Run B, 512, 120ep): Phase 3\runs\detect\train\weights\best.pt
```

Still needed to close out reproducibility: which Phase 3 checkpoint went into Hook-Tap's
`train_combined.py --weights`, and the script(s) that actually computed the ROC-AUC/PR-AUC/
threshold-calibration numbers and the latency benchmark (not yet uploaded to any session).
