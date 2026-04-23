import argparse
import csv
import os
from typing import Dict, List, Tuple


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load_manifest(manifest_path: str) -> Dict[str, Dict[str, str]]:
    if not manifest_path or not os.path.isfile(manifest_path):
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            crop_name = (row.get("crop_name") or "").strip()
            if crop_name:
                rows[crop_name] = dict(row)
    return rows


def load_existing_labels(labels_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.isfile(labels_path):
        return {}

    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: Dict[str, Dict[str, str]] = {}
        for row in reader:
            image_name = (row.get("image") or "").strip()
            if image_name:
                rows[image_name] = dict(row)
    return rows


def collect_images(input_dir: str) -> List[str]:
    names = []
    for name in os.listdir(input_dir):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and name.lower().endswith(IMAGE_EXTS):
            names.append(name)
    names.sort()
    return names


def build_row(
    image_name: str,
    existing_row: Dict[str, str],
    manifest_row: Dict[str, str],
) -> Dict[str, str]:
    return {
        "image": image_name,
        "text": (existing_row.get("text") or "").strip(),
        "source_image": (manifest_row.get("source_image") or existing_row.get("source_image") or "").strip(),
        "x": (manifest_row.get("x") or existing_row.get("x") or "").strip(),
        "y": (manifest_row.get("y") or existing_row.get("y") or "").strip(),
        "w": (manifest_row.get("w") or existing_row.get("w") or "").strip(),
        "h": (manifest_row.get("h") or existing_row.get("h") or "").strip(),
        "notes": (existing_row.get("notes") or "").strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Initialize or refresh labels.csv for real OCR evaluation.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing eval crop images")
    parser.add_argument("--output_csv", type=str, required=True, help="labels.csv output path")
    parser.add_argument("--manifest_csv", type=str, default=None, help="Optional crop manifest.csv path")
    parser.add_argument("--keep_missing", action="store_true", help="Keep existing rows even if image file is currently missing")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"input_dir not found: {args.input_dir}")

    manifest_path = args.manifest_csv or os.path.join(args.input_dir, "manifest.csv")
    manifest_rows = load_manifest(manifest_path)
    existing_rows = load_existing_labels(args.output_csv)
    image_names = collect_images(args.input_dir)

    rows: List[Dict[str, str]] = []
    seen = set()

    for image_name in image_names:
        rows.append(
            build_row(
                image_name=image_name,
                existing_row=existing_rows.get(image_name, {}),
                manifest_row=manifest_rows.get(image_name, {}),
            )
        )
        seen.add(image_name)

    if args.keep_missing:
        for image_name in sorted(existing_rows.keys()):
            if image_name in seen:
                continue
            rows.append(
                build_row(
                    image_name=image_name,
                    existing_row=existing_rows.get(image_name, {}),
                    manifest_row=manifest_rows.get(image_name, {}),
                )
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "text", "source_image", "x", "y", "w", "h", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    filled_text = sum(1 for row in rows if row.get("text"))
    print("labels csv ready")
    print(
        {
            "input_dir": os.path.abspath(args.input_dir),
            "output_csv": os.path.abspath(args.output_csv),
            "manifest_csv": os.path.abspath(manifest_path) if os.path.isfile(manifest_path) else None,
            "rows": len(rows),
            "filled_text": filled_text,
            "empty_text": len(rows) - filled_text,
        }
    )


if __name__ == "__main__":
    main()
