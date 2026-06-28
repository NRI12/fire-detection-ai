"""
Verify a raw dataset source directory before curation.

Checks:
  - All files have recognised image extensions
  - Every image can be opened and decoded by OpenCV (catches truncated files)
  - YOLO label files (same stem, .txt) are syntactically valid if present
  - No duplicate filenames (case-insensitive) within the directory

Does NOT modify any files. Exits 0 if all checks pass, 1 if any fail.

Usage:
    python training/verify_dataset.py data/raw/fire_videos_001/
    python training/verify_dataset.py data/raw/fire_videos_001/ --strict
"""

import argparse
import sys
from pathlib import Path

import cv2

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_MAX_LABEL_FIELDS = 5


def _check_label(label_path: Path) -> list[str]:
    errors = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try latin-1 (Windows notepad sometimes saves this)
        try:
            text = label_path.read_text(encoding="latin-1")
            errors.append(f"{label_path.name}: non-UTF-8 encoding (latin-1 fallback)")
        except Exception as e:
            return [f"{label_path.name}: cannot read — {e}"]

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != _MAX_LABEL_FIELDS:
            errors.append(
                f"{label_path.name}:{lineno}: expected 5 fields, got {len(parts)}"
            )
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            errors.append(f"{label_path.name}:{lineno}: non-numeric values")
            continue
        if cls < 0:
            errors.append(f"{label_path.name}:{lineno}: negative class_id {cls}")
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"{label_path.name}:{lineno}: {name}={val:.4f} out of [0, 1]"
                )
    return errors


def verify(source_dir: Path, strict: bool) -> int:
    if not source_dir.is_dir():
        print(f"[error] Not a directory: {source_dir}", file=sys.stderr)
        return 1

    image_files = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )
    all_files = [p for p in source_dir.iterdir() if p.is_file()]
    non_image = [
        p for p in all_files
        if p.suffix.lower() not in _IMAGE_EXTS and p.suffix.lower() != ".txt"
    ]

    print(f"source     : {source_dir}")
    print(f"images     : {len(image_files)}")
    print(f"other files: {len(non_image)}")

    failures = 0

    # --- unexpected file types ---
    if non_image:
        for p in non_image:
            print(f"[warn] unexpected file: {p.name}")
        if strict:
            failures += len(non_image)

    # --- duplicate filenames (case-insensitive) ---
    seen: dict[str, str] = {}
    for p in image_files:
        key = p.name.lower()
        if key in seen:
            print(f"[error] duplicate filename: {p.name}  vs  {seen[key]}")
            failures += 1
        else:
            seen[key] = p.name

    # --- image readability + label validation ---
    label_errors = 0
    unreadable = 0
    no_label = 0

    for img_path in image_files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[error] cannot decode: {img_path.name}")
            unreadable += 1
            failures += 1
            continue

        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            no_label += 1
            # Missing labels are allowed — hard negatives have no .txt file
            continue

        errs = _check_label(label_path)
        for msg in errs:
            print(f"[error] label: {msg}")
        label_errors += len(errs)
        failures += len(errs)

    print(f"\nresults:")
    print(f"  unreadable images : {unreadable}")
    print(f"  images without label: {no_label}  (expected for hard negatives)")
    print(f"  label errors      : {label_errors}")
    print(f"  total failures    : {failures}")

    if failures == 0:
        print("\n[ok] dataset verified — no issues found")
    else:
        print(f"\n[FAIL] {failures} issue(s) found — fix before curating")

    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a raw dataset source directory"
    )
    parser.add_argument("source_dir", type=Path, help="Directory to verify")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unexpected file types as errors (default: warnings)",
    )
    args = parser.parse_args()
    return verify(args.source_dir, args.strict)


if __name__ == "__main__":
    sys.exit(main())
