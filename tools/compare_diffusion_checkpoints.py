import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_diffusion import DiffusionSampler, build_model, restore_image  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_model_specs(items):
    specs = []
    for item in items:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).stem
        specs.append({"label": label.strip(), "path": path.strip()})
    if not specs:
        raise ValueError("At least one --model item is required")
    return specs


def collect_inputs(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if files:
            return files
    raise FileNotFoundError(f"No input images found under: {input_path}")


def ensure_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def fit_to_size(img, width, height):
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def add_label(tile, text):
    tile = tile.copy()
    h, w = tile.shape[:2]
    bar_h = max(28, h // 12)
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 18), -1)
    tile = cv2.addWeighted(overlay, 0.65, tile, 0.35, 0.0)
    font_scale = max(0.5, min(1.0, w / 520.0))
    cv2.putText(
        tile,
        text,
        (10, int(bar_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return tile


def make_panel(input_bgr, outputs):
    height, width = input_bgr.shape[:2]
    tiles = [add_label(fit_to_size(input_bgr, width, height), "input")]
    for label, img in outputs:
        tiles.append(add_label(fit_to_size(img, width, height), label))
    return cv2.hconcat(tiles)


def main():
    parser = argparse.ArgumentParser(description="Run multiple diffusion checkpoints on the same inputs and save side-by-side panels.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image file or folder")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output folder")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in the form label=path or just path. Can be passed multiple times.",
    )
    parser.add_argument("--timesteps", type=int, default=100, help="Sampling steps for all models")
    parser.add_argument("--target_min_side", type=int, default=128, help="Resize minimum side before inference")
    parser.add_argument("--outscale", type=float, default=1.0, help="Output scale passed to inference")
    parser.add_argument("--init_mode", type=str, default="condition_noise", choices=["noise", "condition", "condition_noise"])
    parser.add_argument("--noise_strength", type=float, default=0.1)
    parser.add_argument("--cond_mode", type=str, default="concat", choices=["concat", "film"])
    parser.add_argument("--decoder_attn", action="store_true", help="Fallback decoder attention switch when ckpt lacks config")
    parser.add_argument("--suffix", type=str, default="compare", help="Output filename suffix")
    args = parser.parse_args()

    model_specs = parse_model_specs(args.model)
    input_files = collect_inputs(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() >= 0 else "cpu"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    print(f"Loading {len(model_specs)} checkpoints on {device} ...")
    loaded = []
    for spec in model_specs:
        model = build_model(
            spec["path"],
            device=device,
            cond_mode=args.cond_mode,
            use_decoder_attn=args.decoder_attn if args.decoder_attn else None,
        )
        sampler = DiffusionSampler(num_timesteps=1000, device=device)
        loaded.append({"label": spec["label"], "path": spec["path"], "model": model, "sampler": sampler})
        print(f"  - {spec['label']}: {spec['path']}")

    summary = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(str(output_dir)),
        "timesteps": args.timesteps,
        "target_min_side": args.target_min_side,
        "outscale": args.outscale,
        "init_mode": args.init_mode,
        "noise_strength": args.noise_strength,
        "models": [{"label": item["label"], "path": os.path.abspath(item["path"])} for item in loaded],
        "items": [],
    }

    start = time.time()
    for idx, input_path in enumerate(input_files, start=1):
        print(f"[{idx}/{len(input_files)}] {input_path.name}")
        input_bgr = ensure_bgr(input_path)
        outputs = []
        item_record = {"input": input_path.name, "panel": None, "outputs": []}

        for item in loaded:
            restored = restore_image(
                item["model"],
                item["sampler"],
                input_bgr,
                device=device,
                target_min_side=args.target_min_side,
                timesteps=args.timesteps,
                outscale=args.outscale,
                init_mode=args.init_mode,
                noise_strength=args.noise_strength,
            )
            outputs.append((item["label"], restored))

            per_model_dir = output_dir / item["label"]
            per_model_dir.mkdir(parents=True, exist_ok=True)
            per_model_path = per_model_dir / f"{input_path.stem}_{args.suffix}.png"
            cv2.imwrite(str(per_model_path), restored)
            item_record["outputs"].append({"label": item["label"], "path": str(per_model_path)})

        panel = make_panel(input_bgr, outputs)
        panel_path = output_dir / f"{input_path.stem}_{args.suffix}_panel.png"
        cv2.imwrite(str(panel_path), panel)
        item_record["panel"] = str(panel_path)
        summary["items"].append(item_record)

    summary["elapsed_sec"] = round(time.time() - start, 3)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved comparison panels to: {output_dir}")


if __name__ == "__main__":
    main()
