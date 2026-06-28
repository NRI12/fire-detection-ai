"""
Perceptual-hash duplicate detection for dataset images.

Uses pHash (64-bit) with chunked numpy matrix multiplication for Hamming
distance, keeping memory usage bounded regardless of dataset size.

Outputs a CSV report of duplicate pairs and an optional removal list.
Does NOT delete any files — removal is a manual step.

Hamming distance formula:
    D[i, j] = popcount(H[i] XOR H[j])
    Computed as: norms[i] + norms[j] - 2 * dot(H[i], H[j])
    where H is the binary hash matrix (each row = 64 bits as float32).

Usage:
    python training/deduplicate.py data/raw/fire_videos_001/
    python training/deduplicate.py data/raw/ --threshold 6 --chunk 500
    python training/deduplicate.py data/raw/ --output-dir review/
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import imagehash
    from PIL import Image
except ImportError:
    print(
        "[error] imagehash not installed. Run: pip install imagehash>=4.3.1",
        file=sys.stderr,
    )
    sys.exit(1)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_DEFAULT_THRESHOLD = 8   # Hamming distance ≤ this → near-duplicate
_DEFAULT_CHUNK = 512     # rows per chunk (~100 MB at 64-bit hashes)


def _compute_hashes(image_files: list[Path]) -> tuple[list[Path], np.ndarray]:
    """Return (valid_paths, hash_matrix) where hash_matrix[i] is 64 float32 bits."""
    valid: list[Path] = []
    rows: list[np.ndarray] = []
    for path in image_files:
        try:
            img = Image.open(path).convert("L")
            h = imagehash.phash(img, hash_size=8)  # 8×8 = 64-bit hash
            # hash_array gives the flattened bool array; cast to float32 for matmul
            bits = np.array(h.hash.flatten(), dtype=np.float32)
            rows.append(bits)
            valid.append(path)
        except Exception as e:
            print(f"[warn] skipping {path.name}: {e}", file=sys.stderr)
    if not rows:
        return [], np.empty((0, 64), dtype=np.float32)
    return valid, np.stack(rows)


def _find_duplicates(
    paths: list[Path],
    hashes: np.ndarray,
    threshold: int,
    chunk_size: int,
) -> list[tuple[Path, Path, int]]:
    """
    Return list of (path_a, path_b, hamming_distance) for pairs below threshold.
    Uses chunked matrix multiplication to bound memory.
    """
    n = len(paths)
    norms = (hashes * hashes).sum(axis=1)  # per-row popcount of the hash

    pairs: list[tuple[Path, Path, int]] = []
    # Only upper triangle: for each row chunk i, compare against all j > max(i_chunk)
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        chunk = hashes[i_start:i_end]                      # shape: (ci, 64)
        dot = chunk @ hashes[i_end:].T                     # shape: (ci, n - i_end)
        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            for local_j, global_j in enumerate(range(i_end, n)):
                dist = int(
                    norms[global_i] + norms[global_j] - 2.0 * dot[local_i, local_j]
                )
                if dist <= threshold:
                    pairs.append((paths[global_i], paths[global_j], dist))

    return pairs


def _select_removal_candidates(
    pairs: list[tuple[Path, Path, int]],
) -> list[Path]:
    """
    For each duplicate group, keep the lexicographically first path, mark the rest
    for removal. Uses a simple union-find to cluster transitive duplicates.
    """
    parent: dict[Path, Path] = {}

    def find(x: Path) -> Path:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: Path, b: Path) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the lexicographically smaller name as root
            if str(ra) <= str(rb):
                parent[rb] = ra
            else:
                parent[ra] = rb

    for a, b, _ in pairs:
        union(a, b)

    groups: dict[Path, list[Path]] = {}
    all_paths = {p for pair in pairs for p in pair[:2]}
    for p in all_paths:
        root = find(p)
        groups.setdefault(root, []).append(p)

    to_remove: list[Path] = []
    for root, members in groups.items():
        keep = min(members, key=lambda p: str(p))
        to_remove.extend(p for p in members if p != keep)

    return sorted(to_remove)


def deduplicate(
    source_dir: Path,
    threshold: int,
    chunk_size: int,
    output_dir: Path,
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

    print(f"source     : {source_dir}")
    print(f"images     : {len(image_files)}")
    print(f"threshold  : hamming ≤ {threshold}")
    print(f"chunk size : {chunk_size}")
    print("computing hashes…")

    paths, hashes = _compute_hashes(image_files)
    print(f"hashed     : {len(paths)} images ({len(image_files) - len(paths)} skipped)")

    if len(paths) < 2:
        print("[ok] fewer than 2 hashable images — nothing to compare")
        return 0

    print("comparing…")
    pairs = _find_duplicates(paths, hashes, threshold, chunk_size)
    print(f"duplicate pairs: {len(pairs)}")

    if not pairs:
        print("[ok] no duplicates found")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = output_dir / "duplicate_pairs.csv"
    removal_txt = output_dir / "duplicates_to_remove.txt"

    with pairs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path_a", "path_b", "hamming_distance"])
        for a, b, dist in sorted(pairs, key=lambda t: t[2]):
            writer.writerow([a, b, dist])

    to_remove = _select_removal_candidates(pairs)
    removal_txt.write_text("\n".join(str(p) for p in to_remove))

    print(f"\npairs CSV        : {pairs_csv}")
    print(f"removal list     : {removal_txt}  ({len(to_remove)} files)")
    print("\nReview the list before deleting anything. No files were modified.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect near-duplicate images via perceptual hashing"
    )
    parser.add_argument("source_dir", type=Path, help="Directory to scan")
    parser.add_argument(
        "--threshold",
        type=int,
        default=_DEFAULT_THRESHOLD,
        help=f"Max Hamming distance to consider a duplicate (default: {_DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=_DEFAULT_CHUNK,
        help=f"Rows per matrix-multiply chunk (default: {_DEFAULT_CHUNK})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review"),
        help="Where to write duplicate_pairs.csv and duplicates_to_remove.txt (default: review/)",
    )
    args = parser.parse_args()
    return deduplicate(args.source_dir, args.threshold, args.chunk, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
