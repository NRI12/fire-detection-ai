"""
Static image inference — M1 validation script.

Usage:
    python scripts/run_inference.py --image data/samples/fire.jpg
    python scripts/run_inference.py --image data/samples/fire.jpg --weights yolov8n.pt --conf 0.45 --output out.jpg
"""

import sys
import argparse
from pathlib import Path

import cv2

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fire_detection.drawing import draw_detections
from fire_detection.inference.model import YOLOv8Detector


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8 static image inference")
    parser.add_argument("--image",   required=True,        help="Path to input image")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights file or model name")
    parser.add_argument("--conf",    default=0.45, type=float, help="Confidence threshold")
    parser.add_argument("--output",  default="output.jpg", help="Path for annotated output image")
    args = parser.parse_args()

    # --- image validation ---
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[error] Image not found: {args.image}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[error] Could not decode image (unsupported format or corrupt file): {args.image}",
              file=sys.stderr)
        return 1

    # --- model loading ---
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

    # --- inference ---
    # Note: first call on GPU is slower due to CUDA kernel initialisation.
    detections, elapsed = detector.predict(frame, conf=args.conf)

    # --- results ---
    print(f"inference  : {elapsed * 1000:.1f}ms")
    print(f"detections : {len(detections)}")
    for det in detections:
        print(f"  {det.class_name:<14} conf={det.confidence:.3f}  box={det.box}")

    # --- output ---
    annotated = draw_detections(frame, detections)
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), annotated)
    print(f"\nsaved      : {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
