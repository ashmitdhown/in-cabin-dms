# Hook-Tap — VALIDATED result

Single YOLOv11s backbone, shared between both tasks:

- **Gaze/Behaviour** runs through the standard YOLO neck/head, completely unmodified.
- **Drowsiness** is served by a private branch (`SpecializedDrowsinessNeckAndHead`) that taps a
  copy of the 512-channel P3 tensor via a forward hook registered on backbone layer 15:
  `Conv2d(512→128→64, stride 2) → AdaptiveAvgPool2d(6,6) → FC(64·36→64→1)`, trained with
  `BCEWithLogitsLoss(pos_weight=0.2)`.

Files:
- `combined_model.py` — `UnifiedInCabinNet`, the model definition
- `train_combined.py` — training driver

## ⚠️ Open item 1 — backbone is frozen forever, not staged

`combined_model.py`'s `forward()` wraps the entire backbone pass in `torch.no_grad()`:

```python
with torch.no_grad():
    _ = self.yolo_wrapper.model(x)
```

This makes the shared backbone **permanently frozen** — gradients never reach it, regardless of
what `train()`/`eval()` mode it's in. If the paper describes a staged Stage 1 (frozen) → Stage 2
(unfrozen fine-tune) process, this implementation is stricter than that: it's frozen-forever.
**Not yet confirmed whether this frozen-forever version is what produced the paper's reported
numbers**, or whether an unfrozen variant was used and this file is a later/different version.

## ⚠️ Open item 2 — broken cross-folder import

```python
from train_mtl import build_yolo_dataloader
```

`train_mtl.py` isn't in this folder — it currently lives in
`mtl/experiments/option3_scale_routing/` (see that folder's README). As committed, this import
will fail unless `train_mtl.py` is duplicated here, moved to a shared location (e.g. a top-level
`mtl/utils/`), or the import path is rewritten. Needs a decision before this becomes runnable
from a fresh clone.

## Which drowsiness backbone was used?

See `phase3_drowsiness/README.md` — two different drowsiness training runs exist, and it's not
confirmed which one's `best.pt` was passed as `--weights` here.

## Results (as reported in the paper)

| | Value |
|---|---|
| Gaze — Precision / Recall | 0.9255 / 0.9463 |
| Gaze — mAP@0.50 / mAP@0.50:0.95 | 0.9557 / 0.8059 |
| Drowsiness — ROC-AUC | 0.759 |
| Drowsiness — PR-AUC | 0.664 |
| Drowsiness — F1 @ τ* | 0.667 |
| Drowsiness — Recall @ τ=0.60 | 97.3% (181/186) |
| Latency (sequential baseline → Hook-Tap) | 8.94 ms → 5.93 ms/frame (1.51×, −33.6%) |
| Params | 9.41M backbone + ~0.3M private branch |
| GFLOPs | 21.3 |

Benchmarked on RTX 5070 Ti. The exact match between the reported Gaze numbers here and Phase 1's
standalone benchmark (`phase1_behaviour/README.md`) confirms the gaze head really does pass
through unmodified, as designed.
