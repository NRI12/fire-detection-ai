"""
Single-threaded realtime webcam inference — M2 validation script.

Usage:
    python scripts/run_webcam.py
    python scripts/run_webcam.py --device 1 --weights yolov8n.pt --conf 0.45 --imgsz 320

Controls:
    q — quit

Notes:
    - Single-threaded: cap.read() and model.predict() execute sequentially.
      Effective FPS is limited by inference latency alone (no parallelism).
    - --imgsz controls inference resolution, not display resolution.
      Lower values (320) trade detection quality for speed.
    - First inference call on GPU is slower due to CUDA kernel initialisation.
"""

import sys
import argparse
from collections import deque
from pathlib import Path
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fire_detection.drawing import draw_detections
from fire_detection.inference.model import YOLOv8Detector

# Exit after this many consecutive frame-read failures.
_MAX_CONSECUTIVE_FAILURES = 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime webcam fire/smoke detection")
    parser.add_argument("--device",  default=0,            type=int,   help="Webcam device index (default: 0)")
    parser.add_argument("--weights", default="yolov8n.pt",             help="YOLOv8 weights file or model name")
    parser.add_argument("--conf",    default=0.45,         type=float, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--imgsz",   default=640,          type=int,   help="Inference input size in pixels (default: 640)")
    args = parser.parse_args()

    # --- model ---
    try:
        detector = YOLOv8Detector(args.weights)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    # --- camera ---
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"[error] Could not open webcam (device={args.device}). "
              "Check the device index and that no other process holds the camera.", file=sys.stderr)
        return 1

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"device     : {detector.device}")
    print(f"weights    : {args.weights}")
    print(f"camera     : device={args.device}  {cam_w}x{cam_h}")
    print(f"threshold  : {args.conf}")
    print(f"imgsz      : {args.imgsz}")
    print("press 'q' to quit\n")

    fps_window: deque[float] = deque(maxlen=30)
    consecutive_failures = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    print(f"\n[error] Camera read failed {_MAX_CONSECUTIVE_FAILURES} times in a row. "
                          "Camera may be disconnected.", file=sys.stderr)
                    break
                continue
            consecutive_failures = 0

            # --- inference ---
            try:
                detections, inf_ms = detector.predict(frame, conf=args.conf, imgsz=args.imgsz)
                inf_ms *= 1000
            except Exception as e:
                print(f"[warn] Inference failed on frame: {e}", file=sys.stderr)
                continue

            # --- FPS (rolling over last 30 completed frames) ---
            fps_window.append(time.monotonic())
            if len(fps_window) > 1:
                fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
            else:
                fps = 0.0

            # --- draw detections + stats overlay ---
            annotated = draw_detections(frame, detections)
            overlay = f"FPS: {fps:.1f}  inf: {inf_ms:.0f}ms  det: {len(detections)}"
            cv2.putText(annotated, overlay, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Fire Detection  [q to quit]", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[info] Quit by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
