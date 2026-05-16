"""
Threaded realtime webcam inference with temporal filtering — M4.

Architecture:
    CaptureThread  →  frame_queue (maxsize=1, latest-frame)
                   →  main thread: inference → TemporalFilter → alert

The temporal filter requires a minimum number of detections within a sliding
time window before emitting an alert, eliminating transient false positives.

Usage:
    python scripts/run_webcam.py
    python scripts/run_webcam.py --device 1 --imgsz 320 --conf 0.45
    python scripts/run_webcam.py --min-det 5 --window 2.0 --cooldown 60.0

Controls:
    q — quit
"""

import sys
import argparse
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fire_detection.capture.camera import CaptureThread
from fire_detection.detection.filter import AlertState, FilterConfig, TemporalFilter
from fire_detection.drawing import draw_detections
from fire_detection.inference.model import YOLOv8Detector

_QUEUE_TIMEOUT_S = 1.0

# Overlay colour per alert state (BGR).
_STATE_COLOUR = {
    AlertState.IDLE:      (180, 180, 180),  # grey
    AlertState.SUSPECTED: (0,   200, 255),  # amber
    AlertState.CONFIRMED: (0,   0,   255),  # red
    AlertState.COOLDOWN:  (0,   140, 255),  # orange
}


def _build_overlay(
    cap_fps: float,
    inf_fps: float,
    dropped: int,
    inf_ms: float,
    result,
) -> str:
    state_name = result.state.value.upper()
    cd = f"cd:{result.cooldown_remaining:.0f}s" if result.cooldown_remaining > 0 else ""
    return (
        f"Cap:{cap_fps:.0f}fps Inf:{inf_fps:.0f}fps Drop:{dropped} inf:{inf_ms:.0f}ms"
        f"  |  {state_name} det:{result.recent_detections} "
        f"conf:{result.smoothed_confidence:.2f} {cd}"
    ).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime fire/smoke detection with temporal filtering")
    parser.add_argument("--device",   default=0,     type=int,   help="Webcam device index (default: 0)")
    parser.add_argument("--weights",  default="yolov8n.pt",      help="YOLOv8 weights or model name")
    parser.add_argument("--conf",     default=0.45,  type=float, help="Per-frame confidence threshold (default: 0.45)")
    parser.add_argument("--imgsz",    default=640,   type=int,   help="Inference input size in pixels (default: 640)")
    parser.add_argument("--min-det",  default=3,     type=int,   help="Detections in window needed to confirm (default: 3)")
    parser.add_argument("--window",   default=1.0,   type=float, help="Detection look-back window in seconds (default: 1.0)")
    parser.add_argument("--cooldown", default=30.0,  type=float, help="Minimum seconds between alerts (default: 30.0)")
    args = parser.parse_args()

    # --- model ---
    try:
        detector = YOLOv8Detector(args.weights)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    # --- temporal filter ---
    flt_cfg = FilterConfig(
        conf_threshold=args.conf,
        window_seconds=args.window,
        min_detections=args.min_det,
        cooldown_seconds=args.cooldown,
    )
    flt = TemporalFilter(flt_cfg)

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
    print(f"filter     : min_det={args.min_det}  window={args.window}s  cooldown={args.cooldown}s")
    print("press 'q' to quit\n")

    inf_fps_ts: deque = deque(maxlen=30)

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=_QUEUE_TIMEOUT_S)
            except queue.Empty:
                if not capture.is_alive():
                    print("[error] Capture thread exited unexpectedly "
                          "(camera disconnected?).", file=sys.stderr)
                    break
                continue

            # --- inference ---
            try:
                detections, inf_s = detector.predict(frame, conf=args.conf, imgsz=args.imgsz)
            except Exception as e:
                print(f"[warn] Inference failed: {e}", file=sys.stderr)
                continue

            # --- temporal filter ---
            result = flt.update(detections)

            if result.alert_emitted:
                ts = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[ALERT] {ts}  state={result.state.value}  "
                    f"conf={result.smoothed_confidence:.2f}  "
                    f"recent_det={result.recent_detections}"
                )

            # --- FPS ---
            inf_fps_ts.append(time.monotonic())
            inf_fps = (
                (len(inf_fps_ts) - 1) / (inf_fps_ts[-1] - inf_fps_ts[0])
                if len(inf_fps_ts) > 1 else 0.0
            )

            # --- draw + overlay ---
            annotated = draw_detections(frame, detections)
            overlay = _build_overlay(
                capture.capture_fps, inf_fps, capture.frames_dropped,
                inf_s * 1000, result,
            )
            colour = _STATE_COLOUR[result.state]
            cv2.putText(annotated, overlay, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

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
