"""
Audit YOLO-format labels in a generated dataset split directory.

Validates that every image has a corresponding label file and every label
file has a corresponding image. Reports class distribution, coordinate
bounds violations, and orphaned files.

Does NOT modify any files. Exits 0 if all checks pass, 1 if any fail.

Usage:
    python training/validate_labels.py data/datasets/fire_v1/train/
    python training/validate_labels.py data/datasets/fire_v1/val/ --classes 0 1
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_label_file(label_path: Path) -> tuple[list[tuple], list[str]]:
    """Return (parsed_rows, error_messages). parsed_rows = [(class_id, cx, cy, w, h), ...]"""
    rows = []
    errors = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [], [f"{label_path.name}: non-UTF-8 encoding"]
    except OSError as e:
        return [], [f"{label_path.name}: cannot read — {e}"]

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path.name}:{lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            errors.append(f"{label_path.name}:{lineno}: non-numeric values")
            continue
        if cls < 0:
            errors.append(f"{label_path.name}:{lineno}: negative class_id {cls}")
            continue
        coord_ok = True
        for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= val <= 1.0):
                errors.append(
                    f"{label_path.name}:{lineno}: {name}={val:.4f} out of [0, 1]"
                )
                coord_ok = False
        if coord_ok:
            rows.append((cls, cx, cy, w, h))
    return rows, errors


def validate(split_dir: Path, allowed_classes: set[int] | None) -> int:
    if not split_dir.is_dir():
        print(f"[error] Not a directory: {split_dir}", file=sys.stderr)
        return 1

    # Auto-detect YOLO layout (separate images/ and labels/ subdirs)
    img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
    lbl_dir = split_dir / "labels" if (split_dir / "labels").is_dir() else split_dir

    image_files = {
        p.stem: p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    }
    label_files = {
        p.stem: p
        for p in lbl_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    }

    print(f"split      : {split_dir}")
    print(f"images     : {len(image_files)}")
    print(f"label files: {len(label_files)}")

    failures = 0
    class_counter: Counter = Counter()
    total_boxes = 0

    # --- orphaned label files (no matching image) ---
    orphan_labels = set(label_files) - set(image_files)
    for stem in sorted(orphan_labels):
        print(f"[error] orphaned label (no image): {label_files[stem].name}")
        failures += 1

    # --- per-image checks ---
    images_without_label = 0
    for stem, img_path in sorted(image_files.items()):
        if stem not in label_files:
            images_without_label += 1
            # Hard negatives intentionally have no label — not a failure
            continue

        rows, errors = _parse_label_file(label_files[stem])
        for msg in errors:
            print(f"[error] {msg}")
            failures += 1

        for cls, *_ in rows:
            class_counter[cls] += 1
            total_boxes += 1

        if allowed_classes is not None:
            unknown = {cls for cls, *_ in rows} - allowed_classes
            for cls in sorted(unknown):
                print(
                    f"[error] {label_files[stem].name}: unknown class_id {cls} "
                    f"(allowed: {sorted(allowed_classes)})"
                )
                failures += 1

    print(f"\nresults:")
    print(f"  images without label (hard negatives): {images_without_label}")
    print(f"  orphaned label files                 : {len(orphan_labels)}")
    print(f"  total bounding boxes                 : {total_boxes}")
    print(f"  total failures                       : {failures}")

    if class_counter:
        print(f"\nclass distribution:")
        for cls_id in sorted(class_counter):
            print(f"  class {cls_id}: {class_counter[cls_id]} boxes")

    if failures == 0:
        print("\n[ok] all labels valid")
    else:
        print(f"\n[FAIL] {failures} issue(s) found")

    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit YOLO-format labels in a split directory"
    )
    parser.add_argument("split_dir", type=Path, help="train/ or val/ directory to audit")
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        metavar="ID",
        help="Allowed class IDs; any other ID is flagged as an error",
    )
    args = parser.parse_args()
    allowed = set(args.classes) if args.classes is not None else None
    return validate(args.split_dir, allowed)


if __name__ == "__main__":
    sys.exit(main())
