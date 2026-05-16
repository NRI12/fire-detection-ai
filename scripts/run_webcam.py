"""
Threaded realtime webcam inference — M3.

Architecture:
    CaptureThread  →  frame_queue (maxsize=1, latest-frame)  →  main thread (inference + display)

The queue holds at most one frame. CaptureThread replaces any unread frame with the newest
one, ensuring inference always operates on the most recent camera output.

Usage:
    python scripts/run_webcam.py
    python scripts/run_webcam.py --device 1 --imgsz 320 --conf 0.45

Controls:
    q — quit
"""

import sys
import argparse
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fire_detection.capture.camera import CaptureThread
from fire_detection.drawing import draw_detections
from fire_detection.inference.model import YOLOv8Detector

_QUEUE_TIMEOUT_S = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Threaded realtime webcam fire/smoke detection")
    parser.add_argument("--device",  default=0,            type=int,   help="Webcam device index (default: 0)")
    parser.add_argument("--weights", default="yolov8n.pt",             help="YOLOv8 weights or model name")
    parser.add_argument("--conf",    default=0.45,         type=float, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--imgsz",   default=640,          type=int,   help="Inference input size in pixels (default: 640)")
    args = parser.parse_args()

    # --- model ---
    try:
        detector = YOLOv8Detector(args.weights)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    # --- shared primitives ---
    frame_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    capture = CaptureThread(args.device, frame_queue, stop_event)

    # --- start capture and wait for camera confirmation ---
    capture.start()
    if not capture.camera_ready.wait(timeout=5.0):
        print(f"[error] Timed out waiting for camera (device={args.device}).", file=sys.stderr)
        stop_event.set()
        return 1
    if capture.camera_error:
        print(f"[error] {capture.camera_error}", file=sys.stderr)
        return 1

    print(f"device     : {detector.device}")
    print(f"weights    : {args.weights}")
    print(f"camera     : device={args.device}")
    print(f"threshold  : {args.conf}")
    print(f"imgsz      : {args.imgsz}")
    print("press 'q' to quit\n")

    inf_fps_ts: deque = deque(maxlen=30)

    try:
        while True:
            # Block until CaptureThread delivers a frame or timeout elapses.
            try:
                frame = frame_queue.get(timeout=_QUEUE_TIMEOUT_S)
            except queue.Empty:
                if not capture.is_alive():
                    print("[error] Capture thread exited unexpectedly "
                          "(camera disconnected?).", file=sys.stderr)
                    break
                continue  # spurious timeout — camera momentarily slow

            # --- inference ---
            try:
                detections, inf_s = detector.predict(frame, conf=args.conf, imgsz=args.imgsz)
            except Exception as e:
                print(f"[warn] Inference failed: {e}", file=sys.stderr)
                continue

            inf_fps_ts.append(time.monotonic())
            inf_fps = (
                (len(inf_fps_ts) - 1) / (inf_fps_ts[-1] - inf_fps_ts[0])
                if len(inf_fps_ts) > 1 else 0.0
            )

            # --- draw + overlay ---
            annotated = draw_detections(frame, detections)
            overlay = (
                f"Cap:{capture.capture_fps:.0f}fps "
                f"Inf:{inf_fps:.0f}fps "
                f"Drop:{capture.frames_dropped} "
                f"inf:{inf_s * 1000:.0f}ms "
                f"det:{len(detections)}"
            )
            cv2.putText(annotated, overlay, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Fire Detection  [q to quit]", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[info] Quit by user.")
                break

    finally:
        stop_event.set()
        capture.join(timeout=3.0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
