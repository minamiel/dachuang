import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Optional


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def collect_images(hr_dir: Path):
    return sorted([p for p in hr_dir.iterdir() if p.is_file() and is_image(p)])


def find_pair(base_dir: Optional[Path], stem: str) -> Optional[Path]:
    if base_dir is None:
        return None
    for ext in IMG_EXTS:
        candidate = base_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a small HR/LR subset and manifest for SAM-assisted mask experiments."
    )
    parser.add_argument("--hr_dir", type=str, required=True, help="Directory containing HR images")
    parser.add_argument("--out_root", type=str, required=True, help="Output directory for the sampled subset")
    parser.add_argument("--count", type=int, default=120, help="Number of images to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lr_dir",
        type=str,
        default="",
        help="Optional directory containing paired LR images with the same stem names",
    )
    parser.add_argument(
        "--copy_lr",
        action="store_true",
        help="Copy paired LR images when --lr_dir is provided",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="samseed",
        help="Output filename prefix, e.g. samseed_001.png",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite manifest and copied files in the output directory",
    )
    args = parser.parse_args()

    hr_dir = Path(args.hr_dir)
    out_root = Path(args.out_root)
    lr_dir = Path(args.lr_dir) if args.lr_dir.strip() else None

    if not hr_dir.exists():
        raise FileNotFoundError(f"hr_dir not found: {hr_dir}")
    if lr_dir is not None and not lr_dir.exists():
        raise FileNotFoundError(f"lr_dir not found: {lr_dir}")

    images = collect_images(hr_dir)
    if not images:
        raise RuntimeError(f"No images found in: {hr_dir}")

    sample_count = min(args.count, len(images))
    rng = random.Random(args.seed)
    sampled = rng.sample(images, sample_count)
    sampled = sorted(sampled)

    hr_out = out_root / "HR"
    lr_out = out_root / "LR"
    manifest_path = out_root / "manifest.csv"

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. Use --overwrite to rebuild this subset."
        )

    hr_out.mkdir(parents=True, exist_ok=True)
    if args.copy_lr:
        lr_out.mkdir(parents=True, exist_ok=True)

    rows = []
    lr_copied = 0
    for idx, src in enumerate(sampled, start=1):
        out_name = f"{args.prefix}_{idx:03d}{src.suffix.lower()}"
        copy_image(src, hr_out / out_name)

        lr_name = ""
        lr_src = find_pair(lr_dir, src.stem)
        if args.copy_lr and lr_src is not None:
            lr_name = out_name
            copy_image(lr_src, lr_out / lr_name)
            lr_copied += 1

        rows.append(
            {
                "sample_id": f"{args.prefix}_{idx:03d}",
                "out_name": out_name,
                "source_hr": str(src.resolve()),
                "source_lr": str(lr_src.resolve()) if lr_src is not None else "",
                "has_lr_copy": int(bool(lr_name)),
                "mask_status": "pending",
                "notes": "",
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "out_name", "source_hr", "source_lr", "has_lr_copy", "mask_status", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared subset: {out_root}")
    print(f"HR copied: {len(rows)}")
    if args.copy_lr:
        print(f"LR copied: {lr_copied}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
