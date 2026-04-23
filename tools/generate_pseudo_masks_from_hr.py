import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class MaskCandidate:
    name: str
    mask: np.ndarray
    score: float
    fg_ratio: float
    component_count: int


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def ensure_uint8_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_for_processing(img: np.ndarray, min_side: int = 128, max_side: int = 512):
    h, w = img.shape[:2]
    scale = 1.0
    short_side = min(h, w)
    long_side = max(h, w)
    if short_side < min_side:
        scale = min_side / float(short_side)
    if long_side * scale > max_side:
        scale = max_side / float(long_side)

    if abs(scale - 1.0) < 1e-6:
        return img.copy(), 1.0

    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def normalize_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.uint8)
    out = (x - x_min) / (x_max - x_min)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def threshold_feature(feature: np.ndarray, inverse: bool = False) -> np.ndarray:
    feature = normalize_uint8(feature)
    threshold_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, mask = cv2.threshold(feature, 0, 255, threshold_type + cv2.THRESH_OTSU)
    return mask


def remove_small_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    area = h * w
    min_area = max(8, int(area * 0.0008))
    max_area = int(area * 0.9)
    min_h = max(2, int(h * 0.08))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_idx in range(1, num_labels):
        x, y, bw, bh, comp_area = stats[label_idx]
        if comp_area < min_area or comp_area > max_area:
            continue
        if bh < min_h:
            continue
        comp = labels == label_idx
        cleaned[comp] = 255
    return cleaned


def refine_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    base = max(1, int(round(min(h, w) / 48)))
    k1 = 2 * base + 1
    k2 = 2 * max(1, base // 2) + 1

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2))
    kernel_main = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_main)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = remove_small_components(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    return mask


def score_mask(mask: np.ndarray) -> Tuple[float, float, int]:
    h, w = mask.shape[:2]
    total = h * w
    fg_ratio = float((mask > 0).sum()) / float(total)

    if fg_ratio <= 0.001:
        return -999.0, fg_ratio, 0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[1:]
    component_count = len(stats)
    if component_count == 0:
        return -999.0, fg_ratio, 0

    xs = stats[:, cv2.CC_STAT_LEFT]
    ys = stats[:, cv2.CC_STAT_TOP]
    ws = stats[:, cv2.CC_STAT_WIDTH]
    hs = stats[:, cv2.CC_STAT_HEIGHT]
    areas = stats[:, cv2.CC_STAT_AREA]

    x0 = xs.min()
    y0 = ys.min()
    x1 = np.max(xs + ws)
    y1 = np.max(ys + hs)
    bbox_w_ratio = float(x1 - x0) / float(w)
    bbox_h_ratio = float(y1 - y0) / float(h)
    largest_comp_ratio = float(areas.max()) / float(total)
    median_h_ratio = float(np.median(hs)) / float(h)

    score = 0.0
    score -= abs(fg_ratio - 0.18) * 6.0
    score -= abs(bbox_w_ratio - 0.75) * 1.5
    score -= abs(bbox_h_ratio - 0.45) * 1.2
    score -= abs(median_h_ratio - 0.28) * 1.0

    if 0.02 <= fg_ratio <= 0.55:
        score += 1.4
    else:
        score -= 2.0

    if 2 <= component_count <= 96:
        score += 1.0
    else:
        score -= 1.5

    if largest_comp_ratio > 0.65:
        score -= 2.5

    return score, fg_ratio, component_count


def build_candidates(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    candidates: List[Tuple[str, np.ndarray]] = []

    _, otsu_bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(("otsu_bright", otsu_bright))
    candidates.append(("otsu_dark", otsu_dark))

    block_size = max(15, int(round(min(h, w) / 4)) | 1)
    adapt_bright = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 7
    )
    adapt_dark = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 7
    )
    candidates.append(("adaptive_bright", adapt_bright))
    candidates.append(("adaptive_dark", adapt_dark))

    rect_w = max(9, int(round(w / 12)) | 1)
    rect_h = max(5, int(round(h / 5)) | 1)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rect_w, rect_h))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rect_kernel)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, rect_kernel)
    candidates.append(("blackhat_otsu", threshold_feature(blackhat, inverse=False)))
    candidates.append(("tophat_otsu", threshold_feature(tophat, inverse=False)))

    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = np.abs(grad_x)
    candidates.append(("gradx_otsu", threshold_feature(grad_x, inverse=False)))

    return candidates


def build_pseudo_mask(img_bgr: np.ndarray) -> Tuple[np.ndarray, MaskCandidate]:
    proc_img, _ = resize_for_processing(img_bgr)
    gray = ensure_uint8_gray(proc_img)

    evaluated: List[MaskCandidate] = []
    for name, raw_mask in build_candidates(gray):
        refined = refine_mask(raw_mask)
        score, fg_ratio, component_count = score_mask(refined)
        evaluated.append(
            MaskCandidate(
                name=name,
                mask=refined,
                score=score,
                fg_ratio=fg_ratio,
                component_count=component_count,
            )
        )

    best = max(evaluated, key=lambda item: item.score)
    final_mask = cv2.resize(
        best.mask,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    final_mask = (final_mask > 0).astype(np.uint8) * 255
    return final_mask, best


def make_preview(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros_like(img_bgr)
    color_mask[:, :, 1] = mask
    overlay = cv2.addWeighted(img_bgr, 0.72, color_mask, 0.28, 0.0)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.concatenate([img_bgr, overlay, mask_rgb], axis=1)


def save_mask(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and is_image(path):
            yield path


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo text masks from HR text crops.")
    parser.add_argument("--hr_dir", type=str, required=True, help="Input HR image directory")
    parser.add_argument("--out_mask_dir", type=str, required=True, help="Output mask directory")
    parser.add_argument("--preview_dir", type=str, default="", help="Optional preview output directory")
    parser.add_argument("--report_path", type=str, default="", help="Optional CSV summary path")
    parser.add_argument("--limit", type=int, default=0, help="Limit image count for debugging")
    parser.add_argument("--preview_count", type=int, default=64, help="How many previews to save")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    args = parser.parse_args()

    hr_dir = Path(args.hr_dir)
    out_mask_dir = Path(args.out_mask_dir)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    report_path = Path(args.report_path) if args.report_path else None

    files = list(iter_images(hr_dir))
    if not files:
        raise FileNotFoundError(f"No image files found under {hr_dir}")
    if args.limit > 0:
        files = files[: args.limit]

    out_mask_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0
    skipped = 0
    preview_saved = 0

    for idx, src in enumerate(files, start=1):
        out_name = f"{src.stem}.png"
        mask_path = out_mask_dir / out_name
        if mask_path.exists() and not args.overwrite:
            skipped += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue

        mask, best = build_pseudo_mask(img)
        save_mask(mask, mask_path)
        saved += 1

        if preview_dir is not None and preview_saved < args.preview_count:
            preview = make_preview(img, mask)
            cv2.imwrite(str(preview_dir / out_name), preview)
            preview_saved += 1

        rows.append(
            {
                "file_name": src.name,
                "mask_name": out_name,
                "method": best.name,
                "score": f"{best.score:.4f}",
                "fg_ratio": f"{best.fg_ratio:.4f}",
                "component_count": best.component_count,
            }
        )

        if idx % 500 == 0:
            print(f"processed={idx} | saved={saved} | skipped={skipped}")

    if report_path is not None:
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file_name", "mask_name", "method", "score", "fg_ratio", "component_count"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print("==============================")
    print(f"Input images: {len(files)}")
    print(f"Masks saved: {saved}")
    print(f"Skipped existing: {skipped}")
    if preview_dir is not None:
        print(f"Previews saved: {preview_saved}")
    if report_path is not None:
        print(f"Report: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()


