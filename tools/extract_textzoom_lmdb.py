import argparse
import csv
import io
from pathlib import Path
from typing import Optional

import lmdb
from PIL import Image


def _read_image(txn, key: bytes):
    raw = txn.get(key)
    if raw is None:
        return None
    return Image.open(io.BytesIO(raw)).convert("RGB")


def export_split(split_dir: Path, split_name: str, out_hr: Path, out_lr: Optional[Path], max_count: int, writer):
    env = lmdb.open(
        str(split_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )

    exported = 0
    with env.begin(write=False) as txn:
        n_samples_raw = txn.get(b"num-samples")
        if n_samples_raw is None:
            raise RuntimeError(f"{split_dir} missing key 'num-samples'; not a valid TextZoom LMDB")

        n_samples = int(n_samples_raw)
        if max_count > 0:
            n_samples = min(n_samples, max_count)

        for i in range(1, n_samples + 1):
            hr_key = f"image_hr-{i:09d}".encode()
            lr_key = f"image_lr-{i:09d}".encode()
            label_key = f"label-{i:09d}".encode()

            img_hr = _read_image(txn, hr_key)
            if img_hr is None:
                continue

            img_lr = _read_image(txn, lr_key) if out_lr is not None else None
            label_raw = txn.get(label_key)
            label = label_raw.decode("utf-8", errors="ignore") if label_raw else ""

            file_name = f"tz_{split_name}_{i:09d}.png"
            hr_path = out_hr / file_name
            img_hr.save(hr_path)

            lr_path_str = ""
            if out_lr is not None and img_lr is not None:
                lr_path = out_lr / file_name
                img_lr.save(lr_path)
                lr_path_str = str(lr_path)

            writer.writerow([file_name, split_name, i, label, str(hr_path), lr_path_str])
            exported += 1

            if i % 500 == 0:
                print(f"[{split_name}] exported {i}/{n_samples}")

    env.close()
    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Export TextZoom LMDB (train1/train2/test) to regular images for this repo."
    )
    parser.add_argument("--textzoom_root", type=str, required=True, help="TextZoom root, e.g. D:/datasets/TextZoom")
    parser.add_argument("--splits", type=str, default="train1,train2", help="Comma-separated split names")
    parser.add_argument("--out_hr", type=str, default="dataset/HR", help="Output folder for HR images")
    parser.add_argument("--out_lr", type=str, default="", help="Optional output folder for LR images")
    parser.add_argument("--max_per_split", type=int, default=0, help="Max images per split, 0 means all")
    parser.add_argument("--meta_csv", type=str, default="dataset/textzoom_meta.csv", help="Output metadata CSV path")
    args = parser.parse_args()

    textzoom_root = Path(args.textzoom_root)
    out_hr = Path(args.out_hr)
    out_lr = Path(args.out_lr) if args.out_lr.strip() else None
    meta_csv = Path(args.meta_csv)

    out_hr.mkdir(parents=True, exist_ok=True)
    if out_lr is not None:
        out_lr.mkdir(parents=True, exist_ok=True)
    meta_csv.parent.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("splits is empty")

    total = 0
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "split", "index", "label", "hr_path", "lr_path"])

        for split in splits:
            split_dir = textzoom_root / split
            if not split_dir.exists():
                print(f"skip: {split_dir} not found")
                continue
            print(f"start: {split_dir}")
            n = export_split(split_dir, split, out_hr, out_lr, args.max_per_split, writer)
            print(f"{split} done: {n} images")
            total += n

    print("=================================")
    print(f"total exported: {total}")
    print(f"HR folder: {out_hr}")
    if out_lr is not None:
        print(f"LR folder: {out_lr}")
    print(f"metadata: {meta_csv}")
    print("done")


if __name__ == "__main__":
    main()
