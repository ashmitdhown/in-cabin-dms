# Experiment — Option 3: Scale-Specific Head Routing

Shared backbone + neck; the behaviour head reads only the P3 feature map, the drowsiness head
reads only P4+P5 — routed by scale specifically to avoid gradient competition between the two
tasks.

**Status:** intermediate experiment, sits chronologically between the FPN-Fusion attempt and the
validated Hook-Tap result. **Not mentioned in the paper** — not yet decided whether to fold this
into the paper's narrative or leave it as a repo-only experiment.

## Files needed here

- `multi_task_yolo_fusion.py`
- `train_mtl.py` — also imported by `mtl/hook_tap/train_combined.py`, so this file is a shared
  dependency, not just this experiment's own driver script. Worth deciding whether it belongs
  here or in a shared `mtl/utils/` location once it's supplied.

Neither file has been uploaded in this repo-building session — **please send them (or confirm
their current path) so they can be added here.**
