"""
extract_frames.py — Extract still frames from the licensed NTHU driver-drowsiness AVI dataset
for use as Phase 3 training images.

Cleaned up from the original inline script (hardcoded paths, fixed start-count) into a
reusable CLI tool. Behaviour is otherwise unchanged: every Nth frame is saved, and output
filenames continue from a configurable starting index so multiple videos can be extracted
into the same output folder without overwriting each other.
"""

import argparse
import os
import cv2


def extract_frames(video_path: str, output_folder: str, frame_interval: int, start_count: int) -> int:
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_count = 0
    saved_count = start_count

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    return saved_count - start_count


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract frames from a driver-drowsiness AVI clip")
    p.add_argument("--video", type=str, required=True, help="Path to the source .avi/.mp4 file")
    p.add_argument("--output", type=str, default="frames", help="Output folder for extracted frames")
    p.add_argument("--interval", type=int, default=10, help="Save every Nth frame")
    p.add_argument("--start-count", type=int, default=0,
                    help="Starting index for saved filenames (use a nonzero value to continue "
                         "numbering across multiple videos into the same output folder, e.g. "
                         "the original run used 700 as an offset)")
    args = p.parse_args()

    n_saved = extract_frames(args.video, args.output, args.interval, args.start_count)
    print(f"Done extracting frames! Saved {n_saved} frames to '{args.output}'.")
