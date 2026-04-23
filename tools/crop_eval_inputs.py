import argparse
import csv
import glob
import os

import cv2


VALID_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def collect_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    paths = []
    for pattern in VALID_EXTS:
        paths.extend(glob.glob(os.path.join(input_path, pattern)))
    return sorted(paths)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def next_index(output_dir):
    existing = glob.glob(os.path.join(output_dir, "eval_*.png"))
    indices = []
    for path in existing:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            indices.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=0) + 1


def append_manifest(manifest_path, rows):
    file_exists = os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["crop_name", "source_image", "x", "y", "w", "h"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Interactively crop real text regions into eval_inputs.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image file or folder")
    parser.add_argument("-o", "--output", type=str, default="eval_inputs", help="Output folder for cropped text images")
    parser.add_argument(
        "--min_size",
        type=int,
        default=64,
        help="Minimum width and height for a valid crop",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="eval",
        help="Output file prefix",
    )
    args = parser.parse_args()

    paths = collect_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    output_dir = ensure_dir(args.output)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    current_index = next_index(output_dir)
    manifest_rows = []

    print("Interactive crop mode")
    print("Controls:")
    print("  - Drag to select a text crop")
    print("  - Press ENTER or SPACE to accept the current selection")
    print("  - Press c to skip the current image")
    print("  - Press q in the terminal after closing windows if you want to stop early")

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable image: {path}")
            continue

        while True:
            title = f"Crop text region: {os.path.basename(path)}"
            x, y, w, h = cv2.selectROI(title, img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(title)

            if w == 0 or h == 0:
                print(f"Skip image without crop: {path}")
                break

            if w < args.min_size or h < args.min_size:
                print(f"Crop too small ({w}x{h}). Select a larger text region.")
                continue

            crop = img[y:y + h, x:x + w]
            crop_name = f"{args.prefix}_{current_index:03d}.png"
            crop_path = os.path.join(output_dir, crop_name)
            cv2.imwrite(crop_path, crop)
            manifest_rows.append([crop_name, path, x, y, w, h])
            print(f"Saved crop: {crop_path}")
            current_index += 1
            break

    if manifest_rows:
        append_manifest(manifest_path, manifest_rows)
        print(f"Updated manifest: {manifest_path}")
        print(f"Total new crops: {len(manifest_rows)}")
    else:
        print("No crops were saved.")


if __name__ == "__main__":
    main()
