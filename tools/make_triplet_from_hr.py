import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def add_gaussian_noise(img: np.ndarray, sigma_min: float, sigma_max: float) -> np.ndarray:
    sigma = random.uniform(sigma_min, sigma_max)
    noise = np.random.randn(*img.shape) * sigma
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_jpeg_artifact(img: np.ndarray, q_min: int, q_max: int) -> np.ndarray:
    q = random.randint(q_min, q_max)
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def random_blur(img: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5, 7])
    sigma = random.uniform(0.2, 2.0)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def random_resize(img: np.ndarray, min_scale: float = 0.5, max_scale: float = 1.5) -> np.ndarray:
    mode = random.choice(["up", "down", "keep"])
    if mode == "up":
        scale = random.uniform(1.0, max_scale)
    elif mode == "down":
        scale = random.uniform(min_scale, 1.0)
    else:
        scale = 1.0

    h, w = img.shape[:2]
    new_h = max(8, int(h * scale))
    new_w = max(8, int(w * scale))
    interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def degrade_realesrgan_style(hr: np.ndarray, scale: int, noise_sigma: tuple, jpeg_q: tuple) -> np.ndarray:
    x = random_blur(hr)
    x = random_resize(x)

    h, w = hr.shape[:2]
    lr_h = max(1, h // scale)
    lr_w = max(1, w // scale)
    interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
    x = cv2.resize(x, (lr_w, lr_h), interpolation=interp)

    if random.random() < 0.8:
        x = add_gaussian_noise(x, noise_sigma[0], noise_sigma[1])
    if random.random() < 0.8:
        x = add_jpeg_artifact(x, jpeg_q[0], jpeg_q[1])

    return x


def find_mask(mask_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Build triplet dataset (HR/LR/masks) with Real-ESRGAN style degradations.")
    parser.add_argument("--hr_dir", type=str, required=True, help="Input HR image directory")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory")
    parser.add_argument("--mask_dir", type=str, default="", help="Optional mask directory with same stem names")
    parser.add_argument("--scale", type=int, default=4, help="Downsample scale for LR")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--noise_sigma_min", type=float, default=1.0)
    parser.add_argument("--noise_sigma_max", type=float, default=12.0)
    parser.add_argument("--jpeg_q_min", type=int, default=40)
    parser.add_argument("--jpeg_q_max", type=int, default=95)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    hr_dir = Path(args.hr_dir)
    out_root = Path(args.out_root)
    out_hr = out_root / "HR"
    out_lr = out_root / "LR"
    out_masks = out_root / "masks"

    out_hr.mkdir(parents=True, exist_ok=True)
    out_lr.mkdir(parents=True, exist_ok=True)

    use_mask = bool(args.mask_dir.strip())
    mask_dir = Path(args.mask_dir) if use_mask else None
    if use_mask:
        out_masks.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in hr_dir.rglob("*") if p.is_file() and is_image(p)])
    if not files:
        raise FileNotFoundError(f"No image files found under {hr_dir}")

    saved = 0
    copied_masks = 0
    for idx, src in enumerate(files, start=1):
        hr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if hr is None:
            continue

        stem = src.stem
        out_name = f"{stem}.png"

        # Save normalized HR copy
        cv2.imwrite(str(out_hr / out_name), hr)

        # Build LR
        lr = degrade_realesrgan_style(
            hr,
            scale=args.scale,
            noise_sigma=(args.noise_sigma_min, args.noise_sigma_max),
            jpeg_q=(args.jpeg_q_min, args.jpeg_q_max),
        )
        cv2.imwrite(str(out_lr / out_name), lr)

        # Optional mask copy
        if use_mask and mask_dir is not None:
            mp = find_mask(mask_dir, stem)
            if mp is not None:
                shutil.copy2(mp, out_masks / out_name)
                copied_masks += 1

        saved += 1
        if idx % 500 == 0:
            print(f"processed={idx} | saved={saved}")

    print("==============================")
    print(f"Triplet dataset built: {out_root}")
    print(f"HR images: {saved}")
    print(f"LR images: {saved}")
    if use_mask:
        print(f"Masks copied: {copied_masks}")
    print("Done.")


if __name__ == "__main__":
    main()
