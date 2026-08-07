# Pipeline — sequential 3-model inference demo

OpenCV-based demo that runs all three detectors on a single image and draws a combined overlay
(gaze/drowsy/object boxes + status panel). Two versions exist with a meaningful difference:

| | `pipeline.py` | `pipeline_object.py` |
|---|---|---|
| Gaze model | Phase 1 custom checkpoint | Phase 1 custom checkpoint |
| Drowsy model | Phase 3 custom checkpoint | Phase 3 custom checkpoint |
| **Object model** | **Phase 2 custom V11 checkpoint** | **Generic pretrained `yolov8s.pt` (COCO "cell phone" class)** |
| Gaze box drawing | Draws all detections above a running best | Only draws the single best-confidence gaze box |

`pipeline_object.py` was renamed from the original `pipeline(object).py` — parentheses in
filenames cause problems with git and shell tooling on some setups, the content is unchanged.

## Known issue: hardcoded absolute paths

Both scripts currently hardcode Windows paths like:

```python
gaze_model = YOLO(r"D:\In Cabin Ai Monitoring System - Ashmit & Aarav\...\best.pt")
```

These need to be swapped for relative paths or a config file (e.g. `config.yaml` with model
paths) before this runs on any machine other than the one it was written on. Not yet done in
this pass — flagging as a TODO before publishing.

## Which one is "the" demo?

Given `pipeline.py` uses the actual custom-trained Phase 2 object detector and
`pipeline_object.py` falls back to generic COCO weights, `pipeline.py` is probably the more
representative demo of the full 3-model system — but this hasn't been explicitly decided.
