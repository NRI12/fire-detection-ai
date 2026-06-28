"""
Interactive visual review of dataset images with bounding boxes.

Draws YOLO-format labels on each image and displays them one at a time.
Lets you flag suspect images or mark duplicates without deleting anything.
Flagged paths are written to a sidecar file for manual follow-up.

Controls:
    n / Space  — next image
    p          — previous image
    f          — flag this image (written to flags file)
    d          — mark as duplicate (written to duplicates file)
    q / Esc    — quit

Usage:
    python training/sample_visualize.py data/raw/fire_videos_001/
    python training/sample_visualize.py data/datasets/fire_v1/train/ --output-dir review/
    python training/sample_visualize.py data/raw/ --shuffle --seed 99 --limit 200
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# BGR colours per class (cycles if > 10 classes)
_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (0, 255, 128),
]


def _colour(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[cls_id % len(_PALETTE)]


def _draw_labels(frame: np.ndarray, label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return frame

    h, w = frame.shape[:2]
    annotated = frame.copy()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return annotated

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = (float(p) for p in parts[1:])
        except ValueError:
            continue
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        colour = _colour(cls)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            annotated, f"cls:{cls}", (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
        )
    return annotated


def _overlay_info(
    frame: np.ndarray,
    idx: int,
    total: int,
    img_path: Path,
    flagged: bool,
    duped: bool,
) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    tags = []
    if flagged:
        tags.append("FLAGGED")
    if duped:
        tags.append("DUPLICATE")
    tag_str = "  ".join(tags)
    lines = [
        f"{idx + 1}/{total}  {img_path.name}",
        "n/Space=next  p=prev  f=flag  d=dup  q=quit",
        tag_str,
    ]
    for i, text in enumerate(lines):
        colour = (0, 0, 255) if (i == 2 and tag_str) else (200, 200, 200)
        cv2.putText(
            out, text, (8, h - 12 - (len(lines) - 1 - i) * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA,
        )
    return out


def run(
    source_dir: Path,
    output_dir: Path,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> int:
    if not source_dir.is_dir():
        print(f"[error] Not a directory: {source_dir}", file=sys.stderr)
        return 1

    image_files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    if not image_files:
        print(f"[error] No images found in {source_dir}", file=sys.stderr)
        return 1

    if shuffle:
        random.seed(seed)
        random.shuffle(image_files)

    if limit is not None:
        image_files = image_files[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    flags_path = output_dir / "flagged.txt"
    dups_path = output_dir / "duplicates.txt"

    # Load any pre-existing flags/dups so repeated sessions accumulate.
    flagged: set[str] = set()
    duped: set[str] = set()
    if flags_path.exists():
        flagged = set(flags_path.read_text().splitlines())
    if dups_path.exists():
        duped = set(dups_path.read_text().splitlines())

    print(f"images     : {len(image_files)}")
    print(f"flags file : {flags_path}")
    print(f"dups file  : {dups_path}")
    print("opening window…")

    idx = 0
    window = "Dataset Review  [n=next  p=prev  f=flag  d=dup  q=quit]"

    while 0 <= idx < len(image_files):
        img_path = image_files[idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[warn] cannot read: {img_path.name}", file=sys.stderr)
            idx += 1
            continue

        label_path = img_path.with_suffix(".txt")
        annotated = _draw_labels(frame, label_path)
        annotated = _overlay_info(
            annotated, idx, len(image_files), img_path,
            str(img_path) in flagged, str(img_path) in duped,
        )

        cv2.imshow(window, annotated)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            print(f"[info] quit at {idx + 1}/{len(image_files)}")
            break
        elif key in (ord("n"), ord(" ")):
            idx += 1
        elif key == ord("p"):
            idx = max(0, idx - 1)
        elif key == ord("f"):
            path_str = str(img_path)
            if path_str in flagged:
                flagged.discard(path_str)
                print(f"[unflagged] {img_path.name}")
            else:
                flagged.add(path_str)
                print(f"[flagged]   {img_path.name}")
            flags_path.write_text("\n".join(sorted(flagged)))
            # Redisplay without advancing
        elif key == ord("d"):
            path_str = str(img_path)
            if path_str in duped:
                duped.discard(path_str)
                print(f"[un-dup]    {img_path.name}")
            else:
                duped.add(path_str)
                print(f"[duplicate] {img_path.name}")
            dups_path.write_text("\n".join(sorted(duped)))
        else:
            idx += 1  # any other key advances

    cv2.destroyAllWindows()
    print(f"\nflagged   : {len(flagged)}")
    print(f"duplicates: {len(duped)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive visual review of dataset images"
    )
    parser.add_argument("source_dir", type=Path, help="Directory of images to review")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review"),
        help="Where to write flagged.txt / duplicates.txt (default: review/)",
    )
    parser.add_argument("--shuffle", action="store_true", help="Randomise order")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max images to show"
    )
    args = parser.parse_args()
    return run(args.source_dir, args.output_dir, args.shuffle, args.seed, args.limit)


if __name__ == "__main__":
    sys.exit(main())
