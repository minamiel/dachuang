import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inference_diffusion import DiffusionSampler, build_model as build_diffusion_model, restore_image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def collect_paths(input_dir):
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(paths)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_realesrgan(args):
    model_name = args.realesrgan_model.split(".")[0]
    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"]
    elif model_name == "RealESRNet_x4plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"]
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"]
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_urls = ["https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"]
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
        netscale = 4
        file_urls = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]
    else:
        raise ValueError(f"Unsupported Real-ESRGAN model: {model_name}")

    if args.realesrgan_model_path:
        model_path = args.realesrgan_model_path
    else:
        model_path = os.path.join("weights", model_name + ".pth")
        if not os.path.isfile(model_path):
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for url in file_urls:
                model_path = load_file_from_url(
                    url=url,
                    model_dir=os.path.join(root_dir, "weights"),
                    progress=True,
                    file_name=None,
                )

    return RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=0,
        half=not args.fp32,
        gpu_id=args.gpu_id,
    )


def upsample_input(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def bicubic_output(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def add_label(img, label):
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def resize_for_panel(img, target_shape):
    target_h, target_w = target_shape[:2]
    if img.shape[:2] == (target_h, target_w):
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def save_method_output(output_root, method_name, base_name, image):
    method_dir = ensure_dir(os.path.join(output_root, method_name))
    path = os.path.join(method_dir, f"{base_name}.png")
    cv2.imwrite(path, image)
    return path


def main():
    parser = argparse.ArgumentParser(description="Run a unified text-crop evaluation workflow.")
    parser.add_argument("--input_dir", type=str, default="eval_inputs", help="Folder containing text crop images")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Folder for evaluation outputs")
    parser.add_argument(
        "--methods",
        type=str,
        default="bicubic,realesrgan,diffusion",
        help="Comma-separated methods: bicubic,realesrgan,diffusion",
    )
    parser.add_argument("--outscale", type=int, default=4, help="Output scale for baseline methods")
    parser.add_argument("--realesrgan_model", type=str, default="RealESRGAN_x4plus", help="Real-ESRGAN model name")
    parser.add_argument("--realesrgan_model_path", type=str, default=None, help="Optional local Real-ESRGAN checkpoint")
    parser.add_argument("--diffusion_model_path", type=str, default="diffusion_model_latest.pth", help="Diffusion checkpoint path")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Diffusion sampling steps")
    parser.add_argument("--diffusion_min_side", type=int, default=256, help="Diffusion resize minimum side")
    parser.add_argument("--tile", type=int, default=0, help="Real-ESRGAN tile size")
    parser.add_argument("--tile_pad", type=int, default=10, help="Real-ESRGAN tile padding")
    parser.add_argument("--fp32", action="store_true", help="Run Real-ESRGAN in fp32")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU id for Real-ESRGAN")
    args = parser.parse_args()

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    paths = collect_paths(args.input_dir)
    if not paths:
        raise FileNotFoundError(f"No evaluation inputs found under: {args.input_dir}")

    ensure_dir(args.output_dir)
    comparison_dir = ensure_dir(os.path.join(args.output_dir, "comparisons"))

    realesrgan = None
    if "realesrgan" in methods:
        realesrgan = build_realesrgan(args)

    diffusion_model = None
    diffusion_sampler = None
    diffusion_device = None
    if "diffusion" in methods and os.path.exists(args.diffusion_model_path):
        diffusion_device = "cuda" if torch.cuda.is_available() else "cpu"
        diffusion_model = build_diffusion_model(args.diffusion_model_path, diffusion_device)
        diffusion_sampler = DiffusionSampler(args.diffusion_steps, diffusion_device)
    elif "diffusion" in methods:
        print(f"Skip diffusion: model not found at {args.diffusion_model_path}")
        methods = [method for method in methods if method != "diffusion"]

    print(f"Run evaluation on {len(paths)} images | methods={methods}")
    for idx, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable image: {path}")
            continue

        base_name = os.path.splitext(os.path.basename(path))[0]
        previews = []
        display_shape = bicubic_output(img, args.outscale).shape

        input_preview = upsample_input(img, args.outscale)
        previews.append(add_label(input_preview, "input_x4_nearest"))

        if "bicubic" in methods:
            bicubic = bicubic_output(img, args.outscale)
            save_method_output(args.output_dir, "bicubic", base_name, bicubic)
            previews.append(add_label(bicubic, "bicubic"))

        if realesrgan is not None:
            realesrgan_img, _ = realesrgan.enhance(img, outscale=args.outscale)
            save_method_output(args.output_dir, "realesrgan", base_name, realesrgan_img)
            previews.append(add_label(resize_for_panel(realesrgan_img, display_shape), f"realesrgan_{args.realesrgan_model}"))

        if diffusion_model is not None and diffusion_sampler is not None:
            diffusion_img = restore_image(
                model=diffusion_model,
                sampler=diffusion_sampler,
                img_bgr=img,
                device=diffusion_device,
                target_min_side=args.diffusion_min_side,
                timesteps=args.diffusion_steps,
            )
            save_method_output(args.output_dir, "diffusion", base_name, diffusion_img)
            previews.append(add_label(resize_for_panel(diffusion_img, display_shape), "diffusion"))

        comparison = np.hstack(previews)
        comparison_path = os.path.join(comparison_dir, f"{base_name}_compare.png")
        cv2.imwrite(comparison_path, comparison)
        print(f"[{idx + 1}/{len(paths)}] saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
