import argparse
from collections import Counter, defaultdict
import csv
import glob
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inference_diffusion import DiffusionSampler, build_model as build_diffusion_model, restore_image


def collect_paths(input_dir):
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(paths)


def calculate_psnr(img_a, img_b):
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)
    mse = np.mean((img_a - img_b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(img_a, img_b):
    # 标准窗口化 SSIM（高斯核），在灰度域计算
    if img_a.ndim == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    if img_b.ndim == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    img_a = img_a.astype(np.float64)
    img_b = img_b.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel_1d = cv2.getGaussianKernel(11, 1.5)
    window = kernel_1d @ kernel_1d.T

    mu_a = cv2.filter2D(img_a, -1, window, borderType=cv2.BORDER_REFLECT)
    mu_b = cv2.filter2D(img_b, -1, window, borderType=cv2.BORDER_REFLECT)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.filter2D(img_a * img_a, -1, window, borderType=cv2.BORDER_REFLECT) - mu_a_sq
    sigma_b_sq = cv2.filter2D(img_b * img_b, -1, window, borderType=cv2.BORDER_REFLECT) - mu_b_sq
    sigma_ab = cv2.filter2D(img_a * img_b, -1, window, borderType=cv2.BORDER_REFLECT) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def build_lpips_model(net="alex", device="cpu"):
    try:
        import lpips  # type: ignore
    except Exception as err:
        raise RuntimeError(
            "LPIPS requested but lpips package is not available. Install with: pip install lpips"
        ) from err

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    return model


def bgr_to_lpips_tensor(image_bgr, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    return tensor * 2.0 - 1.0


def calculate_lpips(lpips_model, img_a, img_b, device):
    with torch.no_grad():
        t_a = bgr_to_lpips_tensor(img_a, device)
        t_b = bgr_to_lpips_tensor(img_b, device)
        value = lpips_model(t_a, t_b)
    return float(value.item())


def find_gt_image(gt_dir, base_name):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    for ext in exts:
        candidate = os.path.join(gt_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_realesrgan(args):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact

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


def resolve_under_output(output_dir, maybe_relative_path):
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    return os.path.join(output_dir, maybe_relative_path)


def build_metric_row(base_name, method_name, pred_img, gt_img, lpips_model, lpips_device):
    row = {
        "image": base_name,
        "method": method_name,
        "psnr": calculate_psnr(pred_img, gt_img),
        "ssim": calculate_ssim(pred_img, gt_img),
    }
    if lpips_model is not None:
        row["lpips"] = calculate_lpips(lpips_model, pred_img, gt_img, lpips_device)
    return row


def run_diffusion_warmup(model, sampler, device, timesteps, target_min_side, outscale):
    warmup_img = np.full((max(64, target_min_side), max(64, target_min_side), 3), 127, dtype=np.uint8)
    t0 = time.time()
    _ = restore_image(
        model=model,
        sampler=sampler,
        img_bgr=warmup_img,
        device=device,
        target_min_side=target_min_side,
        timesteps=timesteps,
        outscale=outscale,
    )
    return float(time.time() - t0)


def is_sample_completed(output_dir, comparison_dir, base_name, methods):
    compare_path = os.path.join(comparison_dir, f"{base_name}_compare.png")
    if not os.path.exists(compare_path):
        return False

    method_to_dir = {
        "bicubic": "bicubic",
        "realesrgan": "realesrgan",
        "diffusion": "diffusion",
    }
    for method in methods:
        method_dir = method_to_dir.get(method)
        if not method_dir:
            continue
        out_path = os.path.join(output_dir, method_dir, f"{base_name}.png")
        if not os.path.exists(out_path):
            return False
    return True


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
    parser.add_argument("--diffusion_model_path", type=str, default="model/diffusion_textzoom_bs8_latest.pth", help="Diffusion checkpoint path")
    parser.add_argument("--diffusion_model_profile", type=str, default="text-priority", help="Diffusion model profile label")
    parser.add_argument("--diffusion_profile_name", type=str, default="custom", help="Diffusion profile name from unified runner")
    parser.add_argument("--diffusion_cond_mode", type=str, default="concat", choices=["concat", "film"], help="Fallback cond_mode when checkpoint has no config")
    parser.add_argument("--diffusion_steps", type=int, default=1000, help="Diffusion sampling steps")
    parser.add_argument("--diffusion_min_side", type=int, default=256, help="Diffusion resize minimum side")
    parser.add_argument("--diffusion_outscale", type=float, default=4.0, help="Fixed output scale factor for diffusion inference")
    parser.add_argument("--diffusion_preserve_color", action="store_true", help="Preserve color in diffusion output")
    parser.add_argument("--diffusion_strict_color_lock", action="store_true", help="Use strict color lock for diffusion output")
    parser.add_argument("--diffusion_luma_strength", type=float, default=1.0, help="Luma strength for strict color lock")
    parser.add_argument("--diffusion_max_luma_delta", type=float, default=24.0, help="Max luma delta for strict color lock")
    parser.add_argument("--diffusion_color_lock_strength", type=float, default=1.0, help="Color lock strength for strict color lock")
    parser.add_argument("--diffusion_edge_sharpen_strength", type=float, default=0.0, help="Edge sharpen strength after diffusion blending")
    parser.add_argument("--diffusion_tile_size", type=int, default=0, help="Tile size for diffusion inference, 0 means disabled")
    parser.add_argument("--diffusion_tile_overlap", type=int, default=32, help="Tile overlap for diffusion tiled inference")
    parser.add_argument("--diffusion_no_tile_blend", action="store_true", help="Disable overlap blending in diffusion tiled inference")
    parser.add_argument("--diffusion_no_match_luma_stats", action="store_true", help="Disable luma stats matching in strict color lock")
    parser.add_argument("--diffusion_enhance_strength", type=float, default=1.0, help="Blend strength between input and diffusion output")
    parser.add_argument("--diffusion_decoder_attn", action="store_true", help="Enable decoder attention in diffusion model (high memory)")
    parser.add_argument("--diffusion_no_warmup", action="store_true", help="Disable diffusion one-time warmup")
    parser.add_argument(
        "--diffusion_fallback_min_side",
        type=int,
        default=256,
        help="Fallback min side when CUDA OOM occurs during diffusion inference",
    )
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional GT directory for quantitative metrics")
    parser.add_argument("--metrics_csv", type=str, default="metrics.csv", help="Metrics csv filename under output_dir")
    parser.add_argument("--lpips", action="store_true", help="Enable LPIPS metric (requires lpips package)")
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone network")
    parser.add_argument("--resume", action="store_true", help="Skip samples with existing outputs")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately when a sample fails")
    parser.add_argument("--failure_csv", type=str, default="failures.csv", help="Failure csv filename under output_dir")
    parser.add_argument("--summary_json", type=str, default="summary.json", help="Summary json filename under output_dir")
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
    diffusion_model_cpu = None
    diffusion_sampler_cpu = None
    if "diffusion" in methods and os.path.exists(args.diffusion_model_path):
        diffusion_device = "cuda" if torch.cuda.is_available() else "cpu"
        diffusion_model = build_diffusion_model(
            args.diffusion_model_path,
            diffusion_device,
            cond_mode=args.diffusion_cond_mode,
            use_decoder_attn=args.diffusion_decoder_attn,
        )
        diffusion_sampler = DiffusionSampler(args.diffusion_steps, diffusion_device)
        diffusion_warmup_sec = None
        if not args.diffusion_no_warmup:
            try:
                diffusion_warmup_sec = run_diffusion_warmup(
                    model=diffusion_model,
                    sampler=diffusion_sampler,
                    device=diffusion_device,
                    timesteps=max(1, min(args.diffusion_steps, 24)),
                    target_min_side=max(64, min(args.diffusion_min_side, 256)),
                    outscale=args.diffusion_outscale,
                )
                print(f"Diffusion warmup complete: {diffusion_warmup_sec:.4f}s")
            except Exception as err:
                print(f"Diffusion warmup skipped due to error: {err}")
    elif "diffusion" in methods:
        print(f"Skip diffusion: model not found at {args.diffusion_model_path}")
        methods = [method for method in methods if method != "diffusion"]

    print(f"Run evaluation on {len(paths)} images | methods={methods}")
    start_time = time.time()
    metric_rows = []
    metric_fields = ["image", "method", "psnr", "ssim"]
    if args.lpips:
        metric_fields.append("lpips")
    failure_rows = []
    skipped_existing = 0
    success_count = 0
    method_elapsed = defaultdict(float)
    method_success = defaultdict(int)
    method_latency_samples = defaultdict(list)
    error_type_counts = Counter()
    sample_elapsed = []

    lpips_model = None
    lpips_device = None
    if args.lpips and args.gt_dir:
        lpips_device = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_model = build_lpips_model(net=args.lpips_net, device=lpips_device)
        print(f"LPIPS enabled | net={args.lpips_net} | device={lpips_device}")
    elif args.lpips:
        print("LPIPS enabled but gt_dir is missing; LPIPS will be skipped.")

    for idx, path in enumerate(paths):
        base_name = os.path.splitext(os.path.basename(path))[0]
        if args.resume and is_sample_completed(args.output_dir, comparison_dir, base_name, methods):
            skipped_existing += 1
            print(f"[{idx + 1}/{len(paths)}] skip existing sample: {base_name}")
            continue

        try:
            sample_t0 = time.time()
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Unreadable image: {path}")

            previews = []
            display_shape = bicubic_output(img, args.outscale).shape

            input_preview = upsample_input(img, args.outscale)
            previews.append(add_label(input_preview, f"input_x{args.outscale}_nearest"))

            if "bicubic" in methods:
                t0 = time.time()
                bicubic = bicubic_output(img, args.outscale)
                save_method_output(args.output_dir, "bicubic", base_name, bicubic)
                previews.append(add_label(bicubic, "bicubic"))
                method_duration = time.time() - t0
                method_elapsed["bicubic"] += method_duration
                method_success["bicubic"] += 1
                method_latency_samples["bicubic"].append(method_duration)
                if args.gt_dir:
                    gt_path = find_gt_image(args.gt_dir, base_name)
                    if gt_path:
                        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                        if gt_img is not None:
                            bicubic_metric = resize_for_panel(bicubic, gt_img.shape)
                            metric_rows.append(
                                build_metric_row(
                                    base_name=base_name,
                                    method_name="bicubic",
                                    pred_img=bicubic_metric,
                                    gt_img=gt_img,
                                    lpips_model=lpips_model,
                                    lpips_device=lpips_device,
                                )
                            )

            if realesrgan is not None:
                t0 = time.time()
                realesrgan_img, _ = realesrgan.enhance(img, outscale=args.outscale)
                save_method_output(args.output_dir, "realesrgan", base_name, realesrgan_img)
                previews.append(add_label(resize_for_panel(realesrgan_img, display_shape), f"realesrgan_{args.realesrgan_model}"))
                method_duration = time.time() - t0
                method_elapsed["realesrgan"] += method_duration
                method_success["realesrgan"] += 1
                method_latency_samples["realesrgan"].append(method_duration)
                if args.gt_dir:
                    gt_path = find_gt_image(args.gt_dir, base_name)
                    if gt_path:
                        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                        if gt_img is not None:
                            realesrgan_metric = resize_for_panel(realesrgan_img, gt_img.shape)
                            metric_rows.append(
                                build_metric_row(
                                    base_name=base_name,
                                    method_name="realesrgan",
                                    pred_img=realesrgan_metric,
                                    gt_img=gt_img,
                                    lpips_model=lpips_model,
                                    lpips_device=lpips_device,
                                )
                            )

            if diffusion_model is not None and diffusion_sampler is not None:
                t0 = time.time()
                try:
                    diffusion_img = restore_image(
                        model=diffusion_model,
                        sampler=diffusion_sampler,
                        img_bgr=img,
                        device=diffusion_device,
                        target_min_side=args.diffusion_min_side,
                        timesteps=args.diffusion_steps,
                        outscale=args.diffusion_outscale,
                        preserve_color=args.diffusion_preserve_color,
                        enhance_strength=args.diffusion_enhance_strength,
                        strict_color_lock=args.diffusion_strict_color_lock,
                        luma_strength=args.diffusion_luma_strength,
                        max_luma_delta=args.diffusion_max_luma_delta,
                        color_lock_strength=args.diffusion_color_lock_strength,
                        edge_sharpen_strength=args.diffusion_edge_sharpen_strength,
                        tile_size=args.diffusion_tile_size,
                        tile_overlap=args.diffusion_tile_overlap,
                        tile_blend=(not args.diffusion_no_tile_blend),
                        match_luma_stats=(not args.diffusion_no_match_luma_stats),
                    )
                except RuntimeError as err:
                    msg = str(err).lower()
                    is_oom = "out of memory" in msg
                    can_retry = (
                        is_oom
                        and diffusion_device == "cuda"
                        and args.diffusion_fallback_min_side < args.diffusion_min_side
                    )
                    if not can_retry:
                        raise

                    torch.cuda.empty_cache()
                    print(
                        f"OOM on image {base_name} with min_side={args.diffusion_min_side}; "
                        f"retry with min_side={args.diffusion_fallback_min_side}"
                    )
                    try:
                        diffusion_img = restore_image(
                            model=diffusion_model,
                            sampler=diffusion_sampler,
                            img_bgr=img,
                            device=diffusion_device,
                            target_min_side=args.diffusion_fallback_min_side,
                            timesteps=args.diffusion_steps,
                            outscale=args.diffusion_outscale,
                            preserve_color=args.diffusion_preserve_color,
                            enhance_strength=args.diffusion_enhance_strength,
                            strict_color_lock=args.diffusion_strict_color_lock,
                            luma_strength=args.diffusion_luma_strength,
                            max_luma_delta=args.diffusion_max_luma_delta,
                            color_lock_strength=args.diffusion_color_lock_strength,
                            edge_sharpen_strength=args.diffusion_edge_sharpen_strength,
                            tile_size=args.diffusion_tile_size,
                            tile_overlap=args.diffusion_tile_overlap,
                            tile_blend=(not args.diffusion_no_tile_blend),
                            match_luma_stats=(not args.diffusion_no_match_luma_stats),
                        )
                    except RuntimeError as err_retry:
                        if "out of memory" not in str(err_retry).lower():
                            raise

                        print(f"Still OOM on CUDA for {base_name}; fallback to CPU")
                        if diffusion_model_cpu is None or diffusion_sampler_cpu is None:
                            diffusion_model_cpu = build_diffusion_model(
                                args.diffusion_model_path,
                                "cpu",
                                cond_mode=args.diffusion_cond_mode,
                                use_decoder_attn=args.diffusion_decoder_attn,
                            )
                            diffusion_sampler_cpu = DiffusionSampler(args.diffusion_steps, "cpu")

                        diffusion_img = restore_image(
                            model=diffusion_model_cpu,
                            sampler=diffusion_sampler_cpu,
                            img_bgr=img,
                            device="cpu",
                            target_min_side=args.diffusion_fallback_min_side,
                            timesteps=args.diffusion_steps,
                            outscale=args.diffusion_outscale,
                            preserve_color=args.diffusion_preserve_color,
                            enhance_strength=args.diffusion_enhance_strength,
                            strict_color_lock=args.diffusion_strict_color_lock,
                            luma_strength=args.diffusion_luma_strength,
                            max_luma_delta=args.diffusion_max_luma_delta,
                            color_lock_strength=args.diffusion_color_lock_strength,
                            edge_sharpen_strength=args.diffusion_edge_sharpen_strength,
                            tile_size=args.diffusion_tile_size,
                            tile_overlap=args.diffusion_tile_overlap,
                            tile_blend=(not args.diffusion_no_tile_blend),
                            match_luma_stats=(not args.diffusion_no_match_luma_stats),
                        )
                method_duration = time.time() - t0
                method_elapsed["diffusion"] += method_duration
                method_success["diffusion"] += 1
                method_latency_samples["diffusion"].append(method_duration)
                save_method_output(args.output_dir, "diffusion", base_name, diffusion_img)
                previews.append(add_label(resize_for_panel(diffusion_img, display_shape), "diffusion"))
                if args.gt_dir:
                    gt_path = find_gt_image(args.gt_dir, base_name)
                    if gt_path:
                        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                        if gt_img is not None:
                            diffusion_metric = resize_for_panel(diffusion_img, gt_img.shape)
                            metric_rows.append(
                                build_metric_row(
                                    base_name=base_name,
                                    method_name="diffusion",
                                    pred_img=diffusion_metric,
                                    gt_img=gt_img,
                                    lpips_model=lpips_model,
                                    lpips_device=lpips_device,
                                )
                            )

            comparison = np.hstack(previews)
            comparison_path = os.path.join(comparison_dir, f"{base_name}_compare.png")
            cv2.imwrite(comparison_path, comparison)
            success_count += 1
            sample_elapsed.append(
                {
                    "image": base_name,
                    "elapsed_sec": round(time.time() - sample_t0, 4),
                }
            )
            print(f"[{idx + 1}/{len(paths)}] saved comparison: {comparison_path}")
        except Exception as err:
            error_type = type(err).__name__
            error_type_counts[error_type] += 1
            failure_rows.append(
                {
                    "image": base_name,
                    "path": path,
                    "error_type": error_type,
                    "elapsed_sec": round(time.time() - sample_t0, 4),
                    "error": f"{err}",
                }
            )
            print(f"[{idx + 1}/{len(paths)}] failed: {base_name} | {err}")
            print(traceback.format_exc())
            if args.fail_fast:
                raise

    if metric_rows:
        metrics_path = os.path.join(args.output_dir, args.metrics_csv)
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"Saved metrics csv: {metrics_path} | rows={len(metric_rows)}")
    else:
        metrics_path = os.path.join(args.output_dir, args.metrics_csv)

    failure_csv_path = resolve_under_output(args.output_dir, args.failure_csv)
    if failure_rows:
        with open(failure_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "path", "error_type", "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(failure_rows)
        print(f"Saved failure csv: {failure_csv_path} | rows={len(failure_rows)}")

    elapsed_sec = time.time() - start_time
    method_stats = {}
    for method in methods:
        s = int(method_success.get(method, 0))
        total = float(method_elapsed.get(method, 0.0))
        latency_arr = np.array(method_latency_samples.get(method, []), dtype=np.float64)
        method_stats[method] = {
            "success": s,
            "total_sec": round(total, 4),
            "avg_sec": round(total / s, 4) if s > 0 else None,
            "p50_sec": round(float(np.percentile(latency_arr, 50)), 4) if latency_arr.size > 0 else None,
            "p90_sec": round(float(np.percentile(latency_arr, 90)), 4) if latency_arr.size > 0 else None,
        }

    summary = {
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "methods": methods,
        "total_images": len(paths),
        "success": success_count,
        "failed": len(failure_rows),
        "skipped_existing": skipped_existing,
        "metric_rows": len(metric_rows),
        "resume": args.resume,
        "fail_fast": args.fail_fast,
        "elapsed_sec": round(elapsed_sec, 4),
    "throughput_img_per_sec": round(success_count / elapsed_sec, 4) if elapsed_sec > 1e-12 else None,
        "avg_sec_per_success": round(elapsed_sec / success_count, 4) if success_count > 0 else None,
        "method_stats": method_stats,
        "error_type_counts": dict(error_type_counts),
        "top_slowest_samples": sorted(sample_elapsed, key=lambda x: x["elapsed_sec"], reverse=True)[:10],
        "top_failure_samples": [
            {
                "image": row.get("image"),
                "error_type": row.get("error_type"),
                "elapsed_sec": row.get("elapsed_sec"),
                "error": row.get("error"),
            }
            for row in failure_rows[:10]
        ],
        "failure_csv": os.path.abspath(failure_csv_path) if failure_rows else None,
        "metrics_csv": os.path.abspath(metrics_path) if metric_rows else None,
        "lpips_enabled": bool(args.lpips and lpips_model is not None),
        "lpips_net": args.lpips_net if args.lpips else None,
        "diffusion_profile_name": args.diffusion_profile_name,
    "diffusion_model_profile": args.diffusion_model_profile,
    "diffusion_warmup_enabled": not args.diffusion_no_warmup,
    "diffusion_warmup_sec": round(diffusion_warmup_sec, 4) if 'diffusion_warmup_sec' in locals() and diffusion_warmup_sec is not None else None,
        "diffusion_config": {
            "steps": args.diffusion_steps,
            "min_side": args.diffusion_min_side,
            "fallback_min_side": args.diffusion_fallback_min_side,
            "outscale": args.diffusion_outscale,
            "enhance_strength": args.diffusion_enhance_strength,
            "strict_color_lock": bool(args.diffusion_strict_color_lock),
            "luma_strength": args.diffusion_luma_strength,
            "max_luma_delta": args.diffusion_max_luma_delta,
            "color_lock_strength": args.diffusion_color_lock_strength,
            "edge_sharpen_strength": args.diffusion_edge_sharpen_strength,
            "tile_size": args.diffusion_tile_size,
            "tile_overlap": args.diffusion_tile_overlap,
            "tile_blend": (not args.diffusion_no_tile_blend),
        },
    }
    summary_path = resolve_under_output(args.output_dir, args.summary_json)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary json: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
