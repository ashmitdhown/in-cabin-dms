# In-Cabin AI Monitoring System

Real-time driver state monitoring for SAE Level 2 partially-automated vehicles, built on three
YOLOv11 detectors and a custom single-backbone fusion architecture (Hook-Tap) that runs two of
them through one shared network instead of two separate ones.

## Contents

- [Motivation](#motivation)
- [Detection modules](#detection-modules)
- [The multi-task fusion journey](#the-multi-task-fusion-journey)
- [Hook-Tap architecture](#hook-tap-architecture)
- [Repo structure](#repo-structure)
- [Getting started](#getting-started)
- [Data & weights](#data--weights)
- [Roadmap](#roadmap)
- [Paper](#paper)
- [Authors](#authors)

## Motivation

SAE Level 2 automation is defined as hands-off, eyes-on — the driver doesn't steer, but has to
stay ready to take over instantly. In practice that's not what happens: NHTSA data shows 57% of
drivers misuse semi-autonomous features and 23% report difficulty staying attentive during
automated driving, while existing driver-monitoring systems — steering-torque sensors, basic
eye-tracking — catch only 70–80% of dangerous behaviors.

A literature review across 2023–2026 driver-monitoring papers (`docs/paper/`) found the same gaps
recurring: architectural fragmentation across single-task models, no YOLO-family multi-task
learning in this space, prohibited-object detection treated as a marginal add-on rather than a
core signal, and no dataset with simultaneous gaze/drowsiness/object annotation. This project
builds three purpose-trained detectors to close the coverage gap, then a fused architecture
(Hook-Tap) to close the runtime-cost gap of running them all.

## Detection modules

| Module | Task | Classes | Dataset | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|---|---|---|
| Behaviour (Phase 1) | Gaze direction | False-Gaze, True-Gaze | 3,254 images, custom | 0.9255 | 0.9463 | 0.9557 | 0.8059 |
| Object (Phase 2) | Prohibited object | True-Object (phone-in-hand) | 700 images, custom | 0.9654 | 0.9633 | 0.9758 | 0.7926 |
| Drowsiness (Phase 3) | Yawning / head-drop | Drowsy-True | 7,900 images, custom + NTHU CV Lab | 0.944 | 0.990 | 0.985 | 0.882 |

All three are YOLOv11s, trained independently with AdamW and phase-specific augmentation
schedules (per-phase READMEs have the exact `yolo detect train` invocations). The behaviour
dataset was shot in a Mercedes ML250 across three lighting conditions and four facial conditions
(clean/bearded × glasses/none); the object dataset in a Chevrolet Malibu across left/right hand
and two phone colors; the drowsiness dataset combines an in-house 18-driver set with the licensed
NTHU CV Lab AVI corpus (5 scenarios: bare face, glasses, sunglasses, night–bare face,
night–glasses).

Full PR/F1/confidence curves and confusion matrices for each model are in `docs/results/`.

## The multi-task fusion journey

Three different approaches to combining gaze and drowsiness into one network were built and
benchmarked before landing on the one actually shipped:

| Approach | Design | Outcome |
|---|---|---|
| **Option 3** — scale-specific head routing | Shared backbone + neck; the behaviour head reads only the P3 feature map, the drowsiness head reads only P4+P5 — routed by scale specifically to keep the two tasks' gradients from competing | Intermediate experiment. Kept in `mtl/experiments/` as a reference point, not carried into the final architecture |
| **FPN-Fusion** | Dual-backbone: two-stage squeeze-excitation channel attention (per-task SE → cross-task fusion SE) feeding a shared PANet neck, with dual task-specific heads | **Abandoned** — negative transfer between the two tasks degraded both |
| **Hook-Tap** | Single backbone, forward hook taps a mid-network feature map, drowsiness routed to a structurally isolated private branch, gaze head left completely untouched | **Validated** — adopted, reported in the paper |

The through-line: full architectural sharing (FPN-Fusion) caused the two tasks to actively
interfere with each other, and partial sharing at the head level (Option 3) still left the
backbone contending with two loss signals. Hook-Tap sidesteps this by not training the shared
backbone against the drowsiness objective at all — the frozen backbone stays exactly as good at
gaze as the standalone Phase 1 model, and drowsiness is learned entirely downstream of a single
tapped feature map.

## Hook-Tap architecture

```
                       Input frame
                           │
                           ▼
              ┌─────────────────────────┐
              │  YOLOv11s backbone       │
              │  (101 layers, frozen)    │
              └────────────┬─────────────┘
                           │  forward hook @ layer 15
                    ┌──────┴───────┐
                    │              │
                    ▼              ▼
         ┌───────────────────┐  ┌────────────────────────┐
         │ Standard YOLO      │  │ Private branch          │
         │ neck + head        │  │ Conv 512→128→64, s=2    │
         │ (unmodified)       │  │ AdaptiveAvgPool(6×6)    │
         │                    │  │ FC(2304→64→1)           │
         └─────────┬──────────┘  └────────────┬────────────┘
                   │                          │
                   ▼                          ▼
        Gaze/behaviour boxes         Drowsiness logit
        (False-Gaze / True-Gaze)     (BCE, pos_weight=0.2)
```

The tapped tensor is the 512-channel P3 feature map, captured off layer 15 via a
`register_forward_hook`. The gaze pipeline runs through Ultralytics' stock neck/head with no
modification — the same layers, same weights, same output shapes as the standalone Phase 1
model — while drowsiness gets a lightweight private branch, ~0.3M additional parameters on top
of the 9.41M shared backbone.

**Why this is faster:** the sequential baseline runs three full forward passes per frame (one
per model). Hook-Tap collapses gaze and drowsiness into a single backbone pass, plus a cheap
downstream branch.

| | Sequential (3 models) | Hook-Tap |
|---|---|---|
| Latency | 8.94 ms/frame (~112 FPS) | 5.93 ms/frame (~168 FPS) |
| Speedup | — | 1.51× (−33.6%) |
| Backbone compute | 3 full backbones | 1 shared backbone + 1 lightweight branch |

Benchmarked on an RTX 5070 Ti.

**Reported results:**

| Metric | Value |
|---|---|
| Gaze — Precision / Recall | 0.9255 / 0.9463 |
| Gaze — mAP@0.50 / mAP@0.50:0.95 | 0.9557 / 0.8059 |
| Drowsiness — ROC-AUC / PR-AUC | 0.759 / 0.664 |
| Drowsiness — Recall @ τ=0.60 (safety threshold) | 97.3% (181/186) |
| Params / GFLOPs | 9.41M + 0.3M / 21.3 |

Object detection (Phase 2) is not folded into Hook-Tap — it stays a separate model, run
alongside in `pipeline/`. Folding it in is on the roadmap below.

## Repo structure

```
phase1_behaviour/     Gaze detector — config, training details, results
phase2_object/        Object detector — config, training details, results
phase3_drowsiness/    Drowsiness detector — config, training details, results
mtl/hook_tap/         Validated fusion model + training script
mtl/experiments/       FPN-Fusion and scale-routing attempts (not in production)
pipeline/              Sequential 3-model inference demo
docs/paper/            IEEE conference paper source
docs/results/          Metrics, curves, confusion matrices
```

## Getting started

```bash
pip install ultralytics opencv-python torch

python pipeline/pipeline.py
```

Developed against Python 3.14, PyTorch 2.10, Ultralytics 8.4. Update the model paths at the top
of `pipeline.py` to point at your own trained weights before running — checkpoints aren't
bundled in this repo.

## Data & weights

Datasets aren't included. The custom sets contain face imagery of the two authors and study
volunteers, kept out of the repo for privacy. Trained weights aren't published here yet.

## Roadmap

Work in progress on folding all three tasks — including object detection — into a single
pipeline with temporal reasoning, rather than Hook-Tap's current two-task scope:

```
Frame → Shared backbone → [Gaze head | Drowsy head | Object head]
      → per-task predictions → attention fusion layer → LSTM → risk output
```

The label format for this stage is already standardized: `Img(gaze, object, drowsy)`, e.g.
`Img(0,1,1)`, merged from the three originally-separate Roboflow projects into unified
`final_train.csv` / `final_val.csv` splits over a centralized image pool. This moves the system
from three independent binary signals to a single temporally-aware risk score.

## Paper

*Hook-Tap: Structurally Gradient-Isolated Single-Backbone Multi-Task Fusion for Real-Time
Driver Monitoring Using Pretrained YOLO Models* — IEEEtran format, submitted to a regional IEEE
conference (2026) as a first step toward mid-tier venues. Source in `docs/paper/`.

```
Citation pending publication.
```

## Authors

Aarav Rasquinha, Ashmit Dhown — BITS Pilani Dubai Campus (EEE / CS)
Advisors: Dr. Ashutosh Mishra, Dr. Ashish

## License

To be finalized.
