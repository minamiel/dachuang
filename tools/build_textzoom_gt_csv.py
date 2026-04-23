import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import lmdb


def parse_textzoom_name(file_name: str) -> Tuple[str, int]:
    stem = Path(file_name).stem
    if "_" not in stem:
        raise ValueError(f"Unexpected TextZoom filename format: {file_name}")
    split, idx_str = stem.split("_", 1)
    return split, int(idx_str)


def load_labels_from_split(split_dir: Path) -> Dict[int, str]:
    env = lmdb.open(
        str(split_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )

    labels: Dict[int, str] = {}
    with env.begin(write=False) as txn:
        n_samples_raw = txn.get(b"num-samples")
        if n_samples_raw is None:
            raise RuntimeError(f"{split_dir} missing key 'num-samples'")
        n_samples = int(n_samples_raw)
        for one_based_idx in range(1, n_samples + 1):
            label_key = f"label-{one_based_idx:09d}".encode()
            label_raw = txn.get(label_key)
            label = label_raw.decode("utf-8", errors="ignore") if label_raw else ""
            labels[one_based_idx - 1] = label
    env.close()
    return labels


def collect_split_maps(textzoom_root: Path, splits):
    split_maps: Dict[str, Dict[int, str]] = {}
    for split in splits:
        split_dir = textzoom_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split LMDB not found: {split_dir}")
        print(f"Loading labels from {split_dir} ...")
        split_maps[split] = load_labels_from_split(split_dir)
        print(f"  loaded {len(split_maps[split])} labels")
    return split_maps


def main():
    parser = argparse.ArgumentParser(description="Build GT CSV for existing TextZoom-style extracted PNGs.")
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory with existing extracted HR images")
    parser.add_argument("--textzoom_root", type=str, required=True, help="Root folder containing train1/train2/test LMDB")
    parser.add_argument("--splits", type=str, default="train1,train2", help="Comma-separated split names to load")
    parser.add_argument("--output_csv", type=str, required=True, help="Output GT CSV path")
    parser.add_argument("--image_col", type=str, default="image", help="Image column name in output CSV")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name in output CSV")
    args = parser.parse_args()

    hr_dir = Path(args.hr_dir)
    textzoom_root = Path(args.textzoom_root)
    output_csv = Path(args.output_csv)

    if not hr_dir.is_dir():
        raise FileNotFoundError(f"hr_dir not found: {hr_dir}")
    if not textzoom_root.is_dir():
        raise FileNotFoundError(f"textzoom_root not found: {textzoom_root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("splits is empty")

    split_maps = collect_split_maps(textzoom_root, splits)
    files = sorted(p for p in hr_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    if not files:
        raise FileNotFoundError(f"No image files found under {hr_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    missing = 0
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[args.image_col, args.text_col, "split", "index"])
        writer.writeheader()

        for p in files:
            split, zero_based_idx = parse_textzoom_name(p.name)
            split_labels = split_maps.get(split)
            if split_labels is None:
                missing += 1
                continue
            label = split_labels.get(zero_based_idx)
            if label is None:
                missing += 1
                continue

            writer.writerow(
                {
                    args.image_col: p.name,
                    args.text_col: label,
                    "split": split,
                    "index": zero_based_idx,
                }
            )
            matched += 1

    print("=================================")
    print(f"Images scanned: {len(files)}")
    print(f"Matched labels: {matched}")
    print(f"Missing labels: {missing}")
    print(f"Output CSV: {output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
