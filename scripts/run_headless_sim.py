"""
Headless webcam simulation for deployment validation.

Feeds a sorted image sequence through the full inference pipeline
(YOLOv8Detector → TemporalFilter → HUD overlay) and saves key frames.
Simulates video-like behavior without a physical camera or display.

Usage:
    venv/bin/python3 scripts/run_headless_sim.py \
        --images data/datasets/dfire_v1/val/images \
        --prefix AoF044 \
        --weights runs/detect/runs/train/dfire_v1_n_baseline/weights/best.pt \
        --out-dir report/figures/ch5_evaluation/inference_tests \
        --tag tp_fire
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

from fire_detection.detection.filter import AlertState, FilterConfig, TemporalFilter
from fire_detection.drawing import draw_detections
from fire_detection.inference.model import YOLOv8Detector

_STATE_COLOUR = {
    AlertState.IDLE:      (180, 180, 180),
    AlertState.SUSPECTED: (0,   200, 255),
    AlertState.CONFIRMED: (0,   0,   255),
    AlertState.COOLDOWN:  (0,   140, 255),
}

_STATE_LABEL_COLOUR = {
    AlertState.IDLE:      (200, 200, 200),
    AlertState.SUSPECTED: (0,   180, 220),
    AlertState.CONFIRMED: (0,   0,   220),
    AlertState.COOLDOWN:  (0,   130, 230),
}


def _draw_hud(frame, cap_fps, inf_fps, inf_ms, result, frame_idx, total):
    annotated = draw_detections(frame, [])  # draw bboxes separately below
    h, w = annotated.shape[:2]

    state = result.state
    colour = _STATE_COLOUR[state]

    # Top status bar
    bar_h = 38
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    state_text = state.value.upper()
    cd_text = f"  cd:{result.cooldown_remaining:.0f}s" if result.cooldown_remaining > 0 else ""
    line1 = (
        f"Cap:{cap_fps:.0f}fps  Inf:{inf_fps:.0f}fps  {inf_ms:.0f}ms"
        f"  |  {state_text}  det:{result.recent_detections}"
        f"  conf:{result.smoothed_confidence:.2f}{cd_text}"
        f"  [{frame_idx}/{total}]"
    )
    cv2.putText(annotated, line1, (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 2, cv2.LINE_AA)

    # State border
    if state in (AlertState.CONFIRMED, AlertState.SUSPECTED):
        thickness = 5 if state == AlertState.CONFIRMED else 2
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), colour, thickness)

    return annotated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images",  required=True, help="Directory with input images")
    p.add_argument("--prefix",  default="",    help="Filename prefix filter (e.g. AoF044)")
    p.add_argument("--weights", required=True, help="YOLOv8 .pt weights")
    p.add_argument("--conf",    default=0.45, type=float)
    p.add_argument("--imgsz",   default=640,  type=int)
    p.add_argument("--min-det", default=3,    type=int)
    p.add_argument("--window",  default=1.0,  type=float)
    p.add_argument("--cooldown",default=30.0, type=float)
    p.add_argument("--out-dir", default="report/figures/ch5_evaluation/inference_tests")
    p.add_argument("--tag",     default="sim", help="Prefix for saved frames")
    p.add_argument("--max-frames", default=80, type=int)
    args = p.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(img_dir.glob(f"{args.prefix}*.jpg"))[:args.max_frames]
    if not imgs:
        print(f"[error] No images matching prefix '{args.prefix}' in {img_dir}", file=sys.stderr)
        return 1
    print(f"Loaded {len(imgs)} images with prefix '{args.prefix}'")

    detector = YOLOv8Detector(args.weights)
    flt = TemporalFilter(FilterConfig(
        conf_threshold=args.conf,
        window_seconds=args.window,
        min_detections=args.min_det,
        cooldown_seconds=args.cooldown,
    ))

    print(f"device: {detector.device}  |  filter: conf={args.conf} window={args.window}s "
          f"min_det={args.min_det} cooldown={args.cooldown}s\n")

    # Warmup pass (first inference is CUDA init)
    warmup_frame = cv2.imread(str(imgs[0]))
    _, warmup_ms = detector.predict(warmup_frame, conf=args.conf, imgsz=args.imgsz)
    print(f"Warmup: {warmup_ms*1000:.0f}ms (excluded from FPS)\n")

    prev_state = AlertState.IDLE
    inf_times = []
    inf_fps_ts: deque = deque(maxlen=30)
    saved = []
    state_log = []
    alerts = 0
    fake_time = 0.0

    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t0 = time.monotonic()
        detections, elapsed = detector.predict(frame, conf=args.conf, imgsz=args.imgsz)
        inf_times.append(elapsed * 1000)
        inf_fps_ts.append(time.monotonic())
        fake_time += elapsed

        inf_fps = (
            (len(inf_fps_ts) - 1) / (inf_fps_ts[-1] - inf_fps_ts[0])
            if len(inf_fps_ts) > 1 else 0.0
        )

        result = flt.update(detections)

        if result.alert_emitted:
            alerts += 1
            print(f"  [ALERT] frame={i+1}  state={result.state.value}  "
                  f"conf={result.smoothed_confidence:.2f}  det={result.recent_detections}")

        state_log.append({
            "frame": i + 1,
            "img": img_path.name,
            "state": result.state.value,
            "det": result.recent_detections,
            "conf": result.smoothed_confidence,
            "classes": [d.class_name for d in detections],
            "confs": [d.confidence for d in detections],
        })

        # Draw detections + HUD
        annotated = draw_detections(frame, detections)
        annotated = _draw_hud(annotated, inf_fps, inf_fps,
                              elapsed * 1000, result, i + 1, len(imgs))

        # Save on state change or every 20 frames
        changed = result.state != prev_state
        if changed or i % 20 == 0 or result.state == AlertState.CONFIRMED:
            fname = f"{args.tag}_f{i+1:03d}_{result.state.value}.jpg"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), annotated)
            saved.append((i + 1, result.state.value, fname))
            if changed:
                print(f"  STATE CHANGE frame={i+1}:  {prev_state.value} → {result.state.value}  "
                      f"(saved {fname})")

        prev_state = result.state

    # Summary
    print(f"\n{'='*60}")
    print(f"Tag        : {args.tag}")
    print(f"Frames     : {len(imgs)}")
    print(f"Alerts     : {alerts}")
    print(f"Inf ms     : avg={sum(inf_times)/len(inf_times):.1f}  "
          f"min={min(inf_times):.1f}  max={max(inf_times):.1f}")
    inf_fps_avg = 1000 / (sum(inf_times)/len(inf_times))
    print(f"Inf FPS    : ~{inf_fps_avg:.0f} fps (post-warmup)")

    final_states = {}
    for s in state_log:
        final_states[s["state"]] = final_states.get(s["state"], 0) + 1
    print("State dist :", "  ".join(f"{k}={v}" for k, v in sorted(final_states.items())))

    print("\nSaved frames:")
    for idx, state, fname in saved:
        print(f"  [{idx:3d}] {state:12s}  {fname}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
