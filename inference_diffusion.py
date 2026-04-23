# -*- coding: utf-8 -*-
import argparse
from collections import Counter
import csv
import glob
import json
import os
import time
import traceback

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model_unet import SimpleUNet


def unpack_model_outputs(model_output):
    if isinstance(model_output, dict):
        return model_output["noise_pred"], model_output.get("mask_pred", None)
    if isinstance(model_output, (tuple, list)):
        noise_pred = model_output[0]
        mask_pred = model_output[1] if len(model_output) > 1 else None
        return noise_pred, mask_pred
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def smart_resize(img, target_min_side=256):
    h, w, _ = img.shape
    scale = target_min_side / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = max(32, (new_h // 32) * 32)
    new_w = max(32, (new_w // 32) * 32)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def upscale_input(img, outscale=4.0):
    if outscale is None or outscale <= 1.0:
        return img
    h, w = img.shape[:2]
    out_h = max(1, int(round(h * outscale)))
    out_w = max(1, int(round(w * outscale)))
    return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


class DiffusionSampler:
    def __init__(self, num_timesteps=1000, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def sample_step(self, model, x, t, condition_img, structure_prior=None):
        cond_mode = getattr(model, "cond_mode", "concat")
        if cond_mode == "film":
            model_output = model(x, t, cond=condition_img, structure_prior=structure_prior)
        else:
            model_input = torch.cat((x, condition_img), dim=1)
            model_output = model(model_input, t, structure_prior=structure_prior)
        predicted_noise, _ = unpack_model_outputs(model_output)

        beta_t = self.beta[t][:, None, None, None]
        alpha_t = self.alpha[t][:, None, None, None]
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]

        coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x - coeff * predicted_noise)

        if t[0] > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            return mean + sigma_t * noise
        return mean

    def make_schedule(self, sample_steps):
        sample_steps = int(max(1, sample_steps))
        if sample_steps >= self.num_timesteps:
            return list(range(self.num_timesteps - 1, -1, -1))
        # 关键修复：少步采样时也要覆盖完整训练时间轴，而不是只跑低噪声前缀。
        values = torch.linspace(self.num_timesteps - 1, 0, steps=sample_steps)
        schedule = values.round().to(torch.long).tolist()
        dedup = []
        last = None
        for value in schedule:
            if value != last:
                dedup.append(value)
                last = value
        return dedup


def image_to_tensor(img_bgr, device):
    tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
    tensor = (tensor / 127.5) - 1.0
    return tensor.unsqueeze(0).to(device)


def rgb_to_luma_tensor(img):
    img_01 = img.clamp(-1.0, 1.0).add(1.0).mul(0.5)
    if img_01.shape[1] == 1:
        return img_01
    r = img_01[:, 0:1]
    g = img_01[:, 1:2]
    b = img_01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def sobel_gradients_tensor(img):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3)
    grad_x = torch.nn.functional.conv2d(img, kernel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(img, kernel_y, padding=1)
    return grad_x, grad_y


def build_structure_prior_tensor(cond_img, strength=1.0):
    strength = float(max(0.0, strength))
    luma = rgb_to_luma_tensor(cond_img)
    grad_x, grad_y = sobel_gradients_tensor(luma)
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
    edge_norm = grad_mag.mean(dim=(-2, -1), keepdim=True)
    edge_map = grad_mag / (edge_norm + 1e-6)
    edge_map = edge_map / (1.0 + edge_map)
    return (strength * edge_map).clamp(0.0, 1.0)


def tensor_to_image(tensor):
    result = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = (result + 1.0) / 2.0
    result = np.clip(result, 0, 1)
    result = (result * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def blend_images(base_bgr, enhanced_bgr, strength=1.0):
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength >= 1.0:
        return enhanced_bgr
    if strength <= 0.0:
        return base_bgr
    mixed = cv2.addWeighted(base_bgr, 1.0 - strength, enhanced_bgr, strength, 0.0)
    return mixed


def apply_edge_sharpen(image_bgr, strength=0.0):
    strength = float(max(0.0, strength))
    if strength <= 1e-8:
        return image_bgr
    # 轻量 unsharp mask：对文本边缘更友好，默认不过度放大噪声
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + strength, blurred, -strength, 0.0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def sample_condition_image(model, sampler, condition, timesteps, device, init_mode="noise", noise_strength=0.1):
    noise_strength = float(max(0.0, noise_strength))
    if init_mode == "noise":
        current = torch.randn_like(condition)
    elif init_mode == "condition":
        current = condition.clone()
    elif init_mode == "condition_noise":
        current = condition + noise_strength * torch.randn_like(condition)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    schedule = sampler.make_schedule(timesteps)
    structure_prior = None
    if getattr(model, "use_structure_prior", False) or getattr(model, "use_decoder_structure_gate", False):
        structure_prior = build_structure_prior_tensor(
            condition,
            strength=float(getattr(model, "structure_prior_strength", 1.0)),
        )
    with torch.no_grad():
        for i in tqdm(schedule, total=len(schedule), leave=False):
            t = torch.tensor([i], device=device)
            current = sampler.sample_step(model, current, t, condition, structure_prior=structure_prior)
    return current


def preserve_color_with_luminance(reference_bgr, enhanced_bgr):
    """保留原图色彩（Cr/Cb），仅使用增强图亮度（Y）。"""
    ref_ycc = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2YCrCb)
    enh_ycc = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2YCrCb)
    out_ycc = ref_ycc.copy()
    out_ycc[..., 0] = enh_ycc[..., 0]
    return cv2.cvtColor(out_ycc, cv2.COLOR_YCrCb2BGR)


def preserve_color_strict(
    reference_bgr,
    enhanced_bgr,
    luma_strength=1.0,
    max_luma_delta=24.0,
    match_luma_stats=True,
    color_lock_strength=1.0,
):
    """严格保色：锁定原图色彩通道，只在亮度上做受控增强。

    - 保留原图 Cr/Cb（色彩）
    - 可选对增强亮度做均值/方差对齐，避免整体观感漂移
    - 限制单像素亮度改变量，避免“过增强”
    """
    luma_strength = float(np.clip(luma_strength, 0.0, 1.0))
    max_luma_delta = float(max(0.0, max_luma_delta))
    color_lock_strength = float(np.clip(color_lock_strength, 0.0, 1.0))

    ref_ycc = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    enh_ycc = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    y_ref = ref_ycc[..., 0]
    y_enh = enh_ycc[..., 0]

    if match_luma_stats:
        ref_mean, ref_std = float(y_ref.mean()), float(y_ref.std())
        enh_mean, enh_std = float(y_enh.mean()), float(y_enh.std())
        if enh_std > 1e-6:
            y_enh = (y_enh - enh_mean) * (ref_std / enh_std) + ref_mean
        else:
            y_enh = y_ref.copy()

    delta = y_enh - y_ref
    if max_luma_delta > 0:
        delta = np.clip(delta, -max_luma_delta, max_luma_delta)

    y_out = y_ref + luma_strength * delta
    y_out = np.clip(y_out, 0.0, 255.0)

    out_ycc = enh_ycc.copy()
    out_ycc[..., 0] = y_out
    # color_lock_strength=1.0 时完全锁色；=0.0 时保留增强图色彩
    out_ycc[..., 1] = color_lock_strength * ref_ycc[..., 1] + (1.0 - color_lock_strength) * enh_ycc[..., 1]
    out_ycc[..., 2] = color_lock_strength * ref_ycc[..., 2] + (1.0 - color_lock_strength) * enh_ycc[..., 2]
    out = cv2.cvtColor(out_ycc.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return out


def load_checkpoint_safely(model_path, device):
    """优先使用 weights_only=True 减少反序列化风险，并兼容旧版 torch。"""
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # 兼容旧版 PyTorch（不支持 weights_only 参数）
        return torch.load(model_path, map_location=device)


def build_model(model_path, device, cond_mode="concat", use_decoder_attn=None):
    ckpt = load_checkpoint_safely(model_path, device)
    model_cfg = {}
    model_state = ckpt
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
        model_cfg = ckpt.get("config", {}) or {}

    ckpt_cond_mode = model_cfg.get("cond_mode", cond_mode)
    use_structure_prior = bool(model_cfg.get("use_structure_prior", False))
    structure_prior_strength = float(model_cfg.get("structure_prior_strength", 1.0))
    use_decoder_structure_gate = bool(model_cfg.get("use_decoder_structure_gate", False))
    structure_gate_strength = float(model_cfg.get("structure_gate_strength", 1.0))
    structure_gate_time_power = float(model_cfg.get("structure_gate_time_power", 1.0))
    has_decoder_attn_weights = any(k.startswith("attn_ups.") for k in model_state.keys())
    if use_decoder_attn is None:
        use_decoder_attn = has_decoder_attn_weights

    model = SimpleUNet(
        cond_mode=ckpt_cond_mode,
        use_decoder_attn=use_decoder_attn,
        use_structure_prior=use_structure_prior,
        use_decoder_structure_gate=use_decoder_structure_gate,
        structure_gate_strength=structure_gate_strength,
        structure_gate_time_power=structure_gate_time_power,
    ).to(device)
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        # 显式关闭 decoder attention 时，允许兼容加载含 attn_ups 权重的旧 checkpoint
        model.load_state_dict(model_state, strict=False)
    model.structure_prior_strength = structure_prior_strength
    model.structure_gate_strength = structure_gate_strength
    model.structure_gate_time_power = structure_gate_time_power
    model.eval()
    return model


def restore_image(
    model,
    sampler,
    img_bgr,
    device,
    target_min_side,
    timesteps,
    outscale=4.0,
    preserve_color=False,
    enhance_strength=1.0,
    strict_color_lock=False,
    luma_strength=1.0,
    max_luma_delta=24.0,
    match_luma_stats=True,
    color_lock_strength=1.0,
    edge_sharpen_strength=0.0,
    tile_size=0,
    tile_overlap=32,
    tile_blend=True,
    init_mode="noise",
    noise_strength=0.1,
):
    sr_input = upscale_input(img_bgr, outscale=outscale)
    target_h, target_w = sr_input.shape[:2]

    def restore_core(input_bgr):
        in_h, in_w = input_bgr.shape[:2]
        resized = smart_resize(input_bgr, target_min_side=target_min_side)
        condition = image_to_tensor(resized, device)
        current = sample_condition_image(
            model,
            sampler,
            condition,
            timesteps,
            device,
            init_mode=init_mode,
            noise_strength=noise_strength,
        )
        restored_work = tensor_to_image(current)
        return cv2.resize(restored_work, (in_w, in_h), interpolation=cv2.INTER_CUBIC)

    use_tile = int(tile_size) > 0 and (target_h > int(tile_size) or target_w > int(tile_size))
    if not use_tile:
        restored = restore_core(sr_input)
    else:
        tile_size = int(max(32, tile_size))
        tile_overlap = int(max(0, min(tile_overlap, tile_size // 2)))
        stride = max(1, tile_size - tile_overlap)

        accum = np.zeros((target_h, target_w, 3), dtype=np.float32)
        weight = np.zeros((target_h, target_w, 1), dtype=np.float32)

        y_starts = list(range(0, target_h, stride))
        x_starts = list(range(0, target_w, stride))
        if y_starts and y_starts[-1] + tile_size < target_h:
            y_starts.append(max(0, target_h - tile_size))
        if x_starts and x_starts[-1] + tile_size < target_w:
            x_starts.append(max(0, target_w - tile_size))

        def make_blend_mask(h, w, overlap):
            m = np.ones((h, w), dtype=np.float32)
            if overlap <= 0:
                return m
            ramp_y = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            ramp_x = np.linspace(0.0, 1.0, overlap, dtype=np.float32)

            m[:overlap, :] *= ramp_y[:, None]
            m[-overlap:, :] *= ramp_y[::-1, None]
            m[:, :overlap] *= ramp_x[None, :]
            m[:, -overlap:] *= ramp_x[None, ::-1]
            return m

        tile_mask_full = make_blend_mask(tile_size, tile_size, tile_overlap) if tile_blend else np.ones((tile_size, tile_size), dtype=np.float32)

        for y in y_starts:
            for x in x_starts:
                y2 = min(target_h, y + tile_size)
                x2 = min(target_w, x + tile_size)
                tile = sr_input[y:y2, x:x2]
                tile_restored = restore_core(tile).astype(np.float32)

                tile_mask = tile_mask_full[: y2 - y, : x2 - x][..., None]
                accum[y:y2, x:x2] += tile_restored * tile_mask
                weight[y:y2, x:x2] += tile_mask

        restored = (accum / np.clip(weight, 1e-6, None)).astype(np.uint8)

    if strict_color_lock:
        restored = preserve_color_strict(
            reference_bgr=sr_input,
            enhanced_bgr=restored,
            luma_strength=luma_strength,
            max_luma_delta=max_luma_delta,
            match_luma_stats=match_luma_stats,
            color_lock_strength=color_lock_strength,
        )
    elif preserve_color:
        restored = preserve_color_with_luminance(reference_bgr=sr_input, enhanced_bgr=restored)

    restored = blend_images(base_bgr=sr_input, enhanced_bgr=restored, strength=enhance_strength)
    restored = apply_edge_sharpen(restored, strength=edge_sharpen_strength)
    return restored


def collect_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(input_path, pattern)))
    return sorted(paths)


def run_warmup(
    model,
    sampler,
    device,
    timesteps,
    target_min_side,
    outscale,
    preserve_color,
    enhance_strength,
    strict_color_lock,
    luma_strength,
    max_luma_delta,
    color_lock_strength,
    edge_sharpen_strength,
    match_luma_stats,
    init_mode,
    noise_strength,
):
    warmup_size = max(32, int(target_min_side))
    warmup_img = np.full((warmup_size, warmup_size, 3), 127, dtype=np.uint8)
    t0 = time.time()
    _ = restore_image(
        model=model,
        sampler=sampler,
        img_bgr=warmup_img,
        device=device,
        target_min_side=target_min_side,
        timesteps=timesteps,
        outscale=outscale,
        preserve_color=preserve_color,
        enhance_strength=enhance_strength,
        strict_color_lock=strict_color_lock,
        luma_strength=luma_strength,
        max_luma_delta=max_luma_delta,
        color_lock_strength=color_lock_strength,
        edge_sharpen_strength=edge_sharpen_strength,
        match_luma_stats=match_luma_stats,
        init_mode=init_mode,
        noise_strength=noise_strength,
    )
    return float(time.time() - t0)


def resolve_under_output(output_dir, maybe_relative_path):
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    return os.path.join(output_dir, maybe_relative_path)


def main():
    parser = argparse.ArgumentParser(description="Diffusion text-restoration inference for crop images.")
    parser.add_argument("-i", "--input", type=str, default="eval_inputs", help="Input image file or folder")
    parser.add_argument("-o", "--output", type=str, default="eval_outputs/diffusion", help="Output folder")
    parser.add_argument("--model_path", type=str, default="model/diffusion_textzoom_bs8_latest.pth", help="Path to diffusion checkpoint")
    parser.add_argument("--model_profile", type=str, default="text-priority", help="Model profile label for reporting")
    parser.add_argument("--timesteps", type=int, default=1000, help="Sampling steps")
    parser.add_argument("--target_min_side", type=int, default=256, help="Resize minimum side before inference")
    parser.add_argument("--outscale", type=float, default=4.0, help="Fixed super-resolution scale factor")
    parser.add_argument("--preserve_color", action="store_true", help="Preserve original image chroma and only enhance luminance")
    parser.add_argument("--enhance_strength", type=float, default=1.0, help="Blend strength between original and enhanced image, range [0,1]")
    parser.add_argument("--strict_color_lock", action="store_true", help="Stricter color lock with controlled luminance enhancement")
    parser.add_argument("--luma_strength", type=float, default=1.0, help="Luminance enhancement strength for strict color lock, range [0,1]")
    parser.add_argument("--max_luma_delta", type=float, default=24.0, help="Max per-pixel luminance delta for strict color lock")
    parser.add_argument("--color_lock_strength", type=float, default=1.0, help="Color lock strength in strict lock mode, range [0,1]")
    parser.add_argument("--edge_sharpen_strength", type=float, default=0.0, help="Edge sharpen strength after blending, range [0,+inf)")
    parser.add_argument("--no_match_luma_stats", action="store_true", help="Disable luminance mean/std matching in strict color lock")
    parser.add_argument("--tile_size", type=int, default=0, help="Tile size for large image inference, 0 means disabled")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Tile overlap for seam smoothing")
    parser.add_argument("--no_tile_blend", action="store_true", help="Disable overlap blend mask in tiled mode")
    parser.add_argument("--decoder_attn", action="store_true", help="Enable decoder attention (very high memory); default off for OOM safety")
    parser.add_argument("--cond_mode", type=str, default="concat", choices=["concat", "film"], help="Fallback cond_mode when checkpoint has no config")
    parser.add_argument("--init_mode", type=str, default="condition_noise", choices=["noise", "condition", "condition_noise"], help="Reverse diffusion initialization mode")
    parser.add_argument("--noise_strength", type=float, default=0.1, help="Noise strength used by condition_noise init")
    parser.add_argument("--suffix", type=str, default="diffusion", help="Output suffix")
    parser.add_argument("--save_comparison", action="store_true", help="Save side-by-side input/output comparisons")
    parser.add_argument("--resume", action="store_true", help="Skip samples with existing outputs")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately when a sample fails")
    parser.add_argument("--failure_csv", type=str, default="failures.csv", help="Failure csv filename under output")
    parser.add_argument("--summary_json", type=str, default="summary.json", help="Summary json filename under output")
    parser.add_argument("--no_warmup", action="store_true", help="Disable one-time warmup before processing")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    os.makedirs(args.output, exist_ok=True)
    paths = collect_paths(args.input)
    if not paths:
        raise FileNotFoundError(f"No images found under: {args.input}")

    print(
        f"Start diffusion inference | device={device} | images={len(paths)} | "
        f"timesteps={args.timesteps} | decoder_attn={args.decoder_attn} | "
        f"init_mode={args.init_mode} | noise_strength={args.noise_strength}"
    )
    start_time = time.time()
    failure_rows = []
    skipped_existing = 0
    success_count = 0
    error_type_counts = Counter()
    sample_elapsed = []

    model = build_model(
        args.model_path,
        device,
        cond_mode=args.cond_mode,
        use_decoder_attn=args.decoder_attn,
    )
    sampler = DiffusionSampler(args.timesteps, device)
    warmup_sec = None
    if not args.no_warmup:
        try:
            warmup_sec = run_warmup(
                model=model,
                sampler=sampler,
                device=device,
                timesteps=max(1, min(args.timesteps, 24)),
                target_min_side=max(64, min(args.target_min_side, 256)),
                outscale=args.outscale,
                preserve_color=args.preserve_color,
                enhance_strength=args.enhance_strength,
                strict_color_lock=args.strict_color_lock,
                luma_strength=args.luma_strength,
                max_luma_delta=args.max_luma_delta,
                color_lock_strength=args.color_lock_strength,
                edge_sharpen_strength=args.edge_sharpen_strength,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                tile_blend=(not args.no_tile_blend),
                match_luma_stats=(not args.no_match_luma_stats),
                init_mode=args.init_mode,
                noise_strength=args.noise_strength,
            )
            print(f"Warmup complete: {warmup_sec:.4f}s")
        except Exception as err:
            print(f"Warmup skipped due to error: {err}")

    for idx, path in enumerate(paths):
        imgname, ext = os.path.splitext(os.path.basename(path))
        output_path = os.path.join(args.output, f"{imgname}_{args.suffix}{ext or '.png'}")

        if args.resume and os.path.exists(output_path):
            skipped_existing += 1
            print(f"[{idx + 1}/{len(paths)}] skip existing: {output_path}")
            continue

        try:
            sample_t0 = time.time()
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Unreadable image: {path}")

            restored = restore_image(
                model=model,
                sampler=sampler,
                img_bgr=img,
                device=device,
                target_min_side=args.target_min_side,
                timesteps=args.timesteps,
                outscale=args.outscale,
                preserve_color=args.preserve_color,
                enhance_strength=args.enhance_strength,
                strict_color_lock=args.strict_color_lock,
                luma_strength=args.luma_strength,
                max_luma_delta=args.max_luma_delta,
                color_lock_strength=args.color_lock_strength,
                edge_sharpen_strength=args.edge_sharpen_strength,
                match_luma_stats=(not args.no_match_luma_stats),
                init_mode=args.init_mode,
                noise_strength=args.noise_strength,
            )
            cv2.imwrite(output_path, restored)

            if args.save_comparison:
                preview_input = cv2.resize(img, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_NEAREST)
                comparison = np.hstack((preview_input, restored))
                comparison_path = os.path.join(args.output, f"{imgname}_{args.suffix}_compare.png")
                cv2.imwrite(comparison_path, comparison)

            success_count += 1
            sample_elapsed.append(
                {
                    "image": imgname,
                    "elapsed_sec": round(time.time() - sample_t0, 4),
                }
            )
            print(f"[{idx + 1}/{len(paths)}] saved {output_path}")
        except Exception as err:
            error_type = type(err).__name__
            error_type_counts[error_type] += 1
            failure_rows.append(
                {
                    "image": imgname,
                    "path": path,
                    "error_type": error_type,
                    "elapsed_sec": round(time.time() - sample_t0, 4),
                    "error": f"{err}",
                }
            )
            print(f"[{idx + 1}/{len(paths)}] failed: {imgname} | {err}")
            print(traceback.format_exc())
            if args.fail_fast:
                raise

    failure_csv_path = resolve_under_output(args.output, args.failure_csv)
    if failure_rows:
        with open(failure_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "path", "error_type", "elapsed_sec", "error"])
            writer.writeheader()
            writer.writerows(failure_rows)
        print(f"Saved failure csv: {failure_csv_path} | rows={len(failure_rows)}")

    elapsed_sec = time.time() - start_time
    summary = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "total_images": len(paths),
        "success": success_count,
        "failed": len(failure_rows),
        "skipped_existing": skipped_existing,
        "resume": args.resume,
        "fail_fast": args.fail_fast,
        "timesteps": args.timesteps,
    "model_profile": args.model_profile,
    "tile_size": args.tile_size,
    "tile_overlap": args.tile_overlap,
    "tile_blend": (not args.no_tile_blend),
        "outscale": args.outscale,
        "elapsed_sec": round(elapsed_sec, 4),
    "throughput_img_per_sec": round(success_count / elapsed_sec, 4) if elapsed_sec > 1e-12 else None,
        "avg_sec_per_success": round(elapsed_sec / success_count, 4) if success_count > 0 else None,
    "warmup_enabled": not args.no_warmup,
    "warmup_sec": round(warmup_sec, 4) if warmup_sec is not None else None,
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
    }
    summary_path = resolve_under_output(args.output, args.summary_json)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary json: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
