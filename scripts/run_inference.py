"""
Static image and video inference.

Usage (image):
    python scripts/run_inference.py --image data/samples/fire.jpg
    python scripts/run_inference.py --image data/samples/fire.jpg --weights path/to/best.pt --conf 0.45

Usage (video):
    python scripts/run_inference.py --video data/samples/fire.mp4
    python scripts/run_inference.py --video fire.mp4 --sample-rate 5 --output out.mp4
"""

import sys
import argparse
import time
from pathlib import Path

import cv2

from fire_detection.drawing import draw_detections
from fire_detection.inference.model import YOLOv8Detector


def run_image(args) -> int:
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[error] Image not found: {args.image}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[error] Could not decode image: {args.image}", file=sys.stderr)
        return 1

    try:
        detector = YOLOv8Detector(args.weights)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    h, w = frame.shape[:2]
    print(f"device     : {detector.device}")
    print(f"weights    : {args.weights}")
    print(f"image      : {image_path}  ({w}x{h})")
    print(f"threshold  : {args.conf}")
    print()

    detections, elapsed = detector.predict(frame, conf=args.conf)

    print(f"inference  : {elapsed * 1000:.1f}ms")
    print(f"detections : {len(detections)}")
    for det in detections:
        print(f"  {det.class_name:<14} conf={det.confidence:.3f}  box={det.box}")

    annotated = draw_detections(frame, detections)
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), annotated)
    print(f"\nsaved      : {output_path.resolve()}")

    return 0


def run_video(args) -> int:
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[error] Video not found: {args.video}", file=sys.stderr)
        return 1

    try:
        detector = YOLOv8Detector(args.weights)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[error] Could not open video: {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"device      : {detector.device}")
    print(f"weights     : {args.weights}")
    print(f"video       : {video_path}  ({w}x{h})  {fps:.1f}fps  {total} frames")
    print(f"threshold   : {args.conf}")
    print(f"sample_rate : every {args.sample_rate} frame(s)")
    print(f"output      : {output_path.resolve()}")
    print()

    last_detections = []
    inference_times = []
    fire_frames = smoke_frames = 0
    frame_idx = 0
    t_start = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.sample_rate == 0:
            dets, elapsed = detector.predict(frame, conf=args.conf)
            last_detections = dets
            inference_times.append(elapsed * 1000)

            if dets:
                has_fire = any(d.class_name == "fire" for d in dets)
                has_smoke = any(d.class_name == "smoke" for d in dets)
                if has_fire:
                    fire_frames += 1
                    ts = frame_idx / fps
                    print(f"  [FIRE]  frame={frame_idx:5d}  t={ts:.1f}s  "
                          f"conf={max(d.confidence for d in dets if d.class_name=='fire'):.3f}")
                if has_smoke:
                    smoke_frames += 1

        annotated = draw_detections(frame, last_detections)
        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100 if total > 0 else 0
            print(f"  progress: {frame_idx}/{total} ({pct:.0f}%)", end="\r")

    cap.release()
    writer.release()

    elapsed_total = time.monotonic() - t_start
    avg_inf = sum(inference_times) / len(inference_times) if inference_times else 0

    print(f"\n\n=== Summary ===")
    print(f"total frames : {frame_idx}")
    print(f"analyzed     : {len(inference_times)} frames (every {args.sample_rate})")
    print(f"duration     : {frame_idx/fps:.1f}s")
    print(f"fire frames  : {fire_frames}")
    print(f"smoke frames : {smoke_frames}")
    print(f"avg inference: {avg_inf:.1f}ms/frame")
    print(f"wall time    : {elapsed_total:.1f}s")
    print(f"saved        : {output_path.resolve()}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8 fire/smoke inference (image or video)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Path to input image")
    mode.add_argument("--video", help="Path to input video")
    parser.add_argument("--weights",     default="yolov8n.pt", help="YOLOv8 weights file")
    parser.add_argument("--conf",        default=0.45, type=float, help="Confidence threshold")
    parser.add_argument("--output",      default=None, help="Path for annotated output file")
    parser.add_argument("--sample-rate", default=1, type=int,
                        help="Video only: run inference every N frames (default: 1 = every frame)")
    args = parser.parse_args()

    if args.output is None:
        args.output = "output.jpg" if args.image else "output.mp4"

    if args.image:
        return run_image(args)
    return run_video(args)


if __name__ == "__main__":
    sys.exit(main())
