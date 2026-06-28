"""
Generate a reproducible YOLO train/val split from curated source images.

Splits by video source (--group-prefix N) to prevent train/val leakage:
all images from the same source clip stay in the same split.

Output layout:
    <output_dir>/
        train/   ← images + .txt labels (hard negatives have no .txt)
        val/     ← images + .txt labels
        dataset.yaml
        manifest.json   ← seed, ratios, group assignments for reproducibility

Does NOT modify the source directory. Always writes to a new directory;
refuses to overwrite an existing one unless --force is given.

Usage:
    python training/generate_split.py data/raw/ data/datasets/fire_v1/
    python training/generate_split.py data/raw/ data/datasets/fire_v1/ --val-ratio 0.2
    python training/generate_split.py data/raw/ data/datasets/fire_v1/ --group-prefix 3 --seed 7
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml  # PyYAML — part of ultralytics transitive deps

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_images(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _group_by_prefix(images: list[Path], prefix_len: int) -> dict[str, list[Path]]:
    """Group image paths by the first `prefix_len` characters of their stem."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        key = img.stem[:prefix_len] if prefix_len > 0 else img.stem
        groups[key].append(img)
    return dict(groups)


def _split_groups(
    groups: dict[str, list[Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """
    Assign whole groups to train or val, targeting val_ratio by image count.
    Groups are shuffled deterministically with `seed`, then greedily packed
    into val until the ratio is met.
    """
    total = sum(len(v) for v in groups.values())
    target_val = int(round(total * val_ratio))

    keys = sorted(groups)  # sorted first for determinism, then shuffled
    rng = random.Random(seed)
    rng.shuffle(keys)

    train_imgs: list[Path] = []
    val_imgs: list[Path] = []
    val_count = 0

    for key in keys:
        group = groups[key]
        if val_count < target_val:
            val_imgs.extend(group)
            val_count += len(group)
        else:
            train_imgs.extend(group)

    return sorted(train_imgs), sorted(val_imgs)


def _copy_split(images: list[Path], dest: Path, yolo_layout: bool = False) -> None:
    """
    Copy images (and matching labels) into dest.

    yolo_layout=False (default): flat directory — image.jpg and image.txt side by side.
    yolo_layout=True: separate images/ and labels/ subdirs expected by Ultralytics.
      Labels are resolved by replacing the 'images' component in the source path
      with 'labels', or by looking in a sibling 'labels/' dir if the source is flat.
    """
    if yolo_layout:
        img_dir = dest / "images"
        lbl_dir = dest / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            shutil.copy2(img_path, img_dir / img_path.name)
            # Resolve label: sibling labels/ dir, or same directory
            label_path = img_path.with_suffix(".txt")
            if not label_path.exists():
                # Try replacing 'images' component in path with 'labels'
                parts = list(img_path.parts)
                try:
                    idx = max(i for i, p in enumerate(parts) if p == "images")
                    parts[idx] = "labels"
                    label_path = Path(*parts).with_suffix(".txt")
                except ValueError:
                    pass
            if label_path.exists():
                shutil.copy2(label_path, lbl_dir / label_path.name)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            shutil.copy2(img_path, dest / img_path.name)
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, dest / label_path.name)


def _write_yaml(output_dir: Path, class_names: list[str], yolo_layout: bool = False) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    train_path = "train/images" if yolo_layout else "train"
    val_path   = "val/images"   if yolo_layout else "val"
    data = {
        "path": str(output_dir.resolve()),
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return yaml_path


def _write_manifest(
    output_dir: Path,
    seed: int,
    val_ratio: float,
    prefix_len: int,
    groups: dict[str, list[Path]],
    train_imgs: list[Path],
    val_imgs: list[Path],
    source_dir: Path,
) -> Path:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "seed": seed,
        "val_ratio": val_ratio,
        "group_prefix_len": prefix_len,
        "total_images": len(train_imgs) + len(val_imgs),
        "train_images": len(train_imgs),
        "val_images": len(val_imgs),
        "actual_val_ratio": len(val_imgs) / max(len(train_imgs) + len(val_imgs), 1),
        "groups": {k: [str(p) for p in v] for k, v in sorted(groups.items())},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def generate(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    prefix_len: int,
    class_names: list[str],
    force: bool,
    yolo_layout: bool = False,
) -> int:
    if not source_dir.is_dir():
        print(f"[error] Not a directory: {source_dir}", file=sys.stderr)
        return 1

    if output_dir.exists() and not force:
        print(
            f"[error] Output directory already exists: {output_dir}\n"
            "        Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    images = _collect_images(source_dir)
    if not images:
        print(f"[error] No images found in {source_dir}", file=sys.stderr)
        return 1

    print(f"source     : {source_dir}")
    print(f"output     : {output_dir}")
    print(f"images     : {len(images)}")
    print(f"val ratio  : {val_ratio}")
    print(f"seed       : {seed}")
    print(f"group pfx  : {prefix_len} char(s)")

    groups = _group_by_prefix(images, prefix_len)
    print(f"groups     : {len(groups)}")

    train_imgs, val_imgs = _split_groups(groups, val_ratio, seed)

    if not train_imgs:
        print("[error] train split is empty — adjust --val-ratio or --group-prefix", file=sys.stderr)
        return 1
    if not val_imgs:
        print("[error] val split is empty — adjust --val-ratio or --group-prefix", file=sys.stderr)
        return 1

    actual_ratio = len(val_imgs) / (len(train_imgs) + len(val_imgs))
    print(f"train      : {len(train_imgs)} images")
    print(f"val        : {len(val_imgs)} images  (actual ratio: {actual_ratio:.3f})")

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    print("copying…")
    _copy_split(train_imgs, output_dir / "train", yolo_layout=yolo_layout)
    _copy_split(val_imgs, output_dir / "val", yolo_layout=yolo_layout)

    yaml_path = _write_yaml(output_dir, class_names, yolo_layout=yolo_layout)
    manifest_path = _write_manifest(
        output_dir, seed, val_ratio, prefix_len,
        groups, train_imgs, val_imgs, source_dir,
    )

    print(f"\ndataset.yaml : {yaml_path}")
    print(f"manifest.json: {manifest_path}")
    print("\n[ok] split generated")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible YOLO train/val split"
    )
    parser.add_argument("source_dir", type=Path, help="Curated source images directory")
    parser.add_argument("output_dir", type=Path, help="Destination dataset directory")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of images for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--group-prefix",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Group images by first N chars of filename stem to prevent leakage. "
            "0 = each image is its own group (default: 0)"
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["fire", "smoke"],
        metavar="NAME",
        help="Class names in class-ID order for dataset.yaml (default: fire smoke)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it already exists",
    )
    parser.add_argument(
        "--yolo-layout",
        action="store_true",
        help="Write separate images/ and labels/ subdirs (required for Ultralytics training)",
    )
    args = parser.parse_args()
    return generate(
        args.source_dir,
        args.output_dir,
        args.val_ratio,
        args.seed,
        args.group_prefix,
        args.classes,
        args.force,
        yolo_layout=args.yolo_layout,
    )


if __name__ == "__main__":
    sys.exit(main())
