"""
YOLOv8n baseline training trên dataset dfire_v1.

Run:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --batch 8
    python scripts/train_baseline.py --resume
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


DATASET_YAML = Path("data/datasets/dfire_v1/dataset.yaml")
RUN_NAME     = "dfire_v1_n_baseline"
PROJECT      = "runs/train"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        print(f"[error] Dataset not found: {DATASET_YAML}", file=sys.stderr)
        return 1

    if args.resume:
        last_weights = Path(PROJECT) / RUN_NAME / "weights" / "last.pt"
        if not last_weights.exists():
            print(f"[error] No checkpoint to resume: {last_weights}", file=sys.stderr)
            return 1
        model = YOLO(str(last_weights))
    else:
        model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATASET_YAML.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=42,
        name=RUN_NAME,
        project=PROJECT,
        resume=args.resume,
        verbose=True,
    )

    print(f"\n[ok] Training complete. Results: {PROJECT}/{RUN_NAME}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
