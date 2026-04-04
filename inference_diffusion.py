# -*- coding: utf-8 -*-
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model_unet import SimpleUNet


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

    def sample_step(self, model, x, t, condition_img):
        cond_mode = getattr(model, "cond_mode", "concat")
        if cond_mode == "film":
            predicted_noise, _ = model(x, t, cond=condition_img)
        else:
            model_input = torch.cat((x, condition_img), dim=1)
            predicted_noise, _ = model(model_input, t)

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


def image_to_tensor(img_bgr, device):
    tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
    tensor = (tensor / 127.5) - 1.0
    return tensor.unsqueeze(0).to(device)


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
):
    """严格保色：锁定原图色彩通道，只在亮度上做受控增强。

    - 保留原图 Cr/Cb（色彩）
    - 可选对增强亮度做均值/方差对齐，避免整体观感漂移
    - 限制单像素亮度改变量，避免“过增强”
    """
    luma_strength = float(np.clip(luma_strength, 0.0, 1.0))
    max_luma_delta = float(max(0.0, max_luma_delta))

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

    out_ycc = ref_ycc.copy()
    out_ycc[..., 0] = y_out
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
    has_decoder_attn_weights = any(k.startswith("attn_ups.") for k in model_state.keys())
    if use_decoder_attn is None:
        use_decoder_attn = has_decoder_attn_weights

    model = SimpleUNet(cond_mode=ckpt_cond_mode, use_decoder_attn=use_decoder_attn).to(device)
    try:
        model.load_state_dict(model_state)
    except RuntimeError:
        # 显式关闭 decoder attention 时，允许兼容加载含 attn_ups 权重的旧 checkpoint
        model.load_state_dict(model_state, strict=False)
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
):
    sr_input = upscale_input(img_bgr, outscale=outscale)
    target_h, target_w = sr_input.shape[:2]

    resized = smart_resize(sr_input, target_min_side=target_min_side)
    condition = image_to_tensor(resized, device)
    current = torch.randn_like(condition)

    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)), total=timesteps, leave=False):
            t = torch.tensor([i], device=device)
            current = sampler.sample_step(model, current, t, condition)

    restored_work = tensor_to_image(current)
    restored = cv2.resize(restored_work, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    if strict_color_lock:
        restored = preserve_color_strict(
            reference_bgr=sr_input,
            enhanced_bgr=restored,
            luma_strength=luma_strength,
            max_luma_delta=max_luma_delta,
            match_luma_stats=match_luma_stats,
        )
    elif preserve_color:
        restored = preserve_color_with_luminance(reference_bgr=sr_input, enhanced_bgr=restored)

    restored = blend_images(base_bgr=sr_input, enhanced_bgr=restored, strength=enhance_strength)
    return restored


def collect_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(input_path, pattern)))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Diffusion text-restoration inference for crop images.")
    parser.add_argument("-i", "--input", type=str, default="eval_inputs", help="Input image file or folder")
    parser.add_argument("-o", "--output", type=str, default="eval_outputs/diffusion", help="Output folder")
    parser.add_argument("--model_path", type=str, default="model/diffusion_textzoom_bs8_latest.pth", help="Path to diffusion checkpoint")
    parser.add_argument("--timesteps", type=int, default=1000, help="Sampling steps")
    parser.add_argument("--target_min_side", type=int, default=256, help="Resize minimum side before inference")
    parser.add_argument("--outscale", type=float, default=4.0, help="Fixed super-resolution scale factor")
    parser.add_argument("--preserve_color", action="store_true", help="Preserve original image chroma and only enhance luminance")
    parser.add_argument("--enhance_strength", type=float, default=1.0, help="Blend strength between original and enhanced image, range [0,1]")
    parser.add_argument("--strict_color_lock", action="store_true", help="Stricter color lock with controlled luminance enhancement")
    parser.add_argument("--luma_strength", type=float, default=1.0, help="Luminance enhancement strength for strict color lock, range [0,1]")
    parser.add_argument("--max_luma_delta", type=float, default=24.0, help="Max per-pixel luminance delta for strict color lock")
    parser.add_argument("--no_match_luma_stats", action="store_true", help="Disable luminance mean/std matching in strict color lock")
    parser.add_argument("--decoder_attn", action="store_true", help="Enable decoder attention (very high memory); default off for OOM safety")
    parser.add_argument("--cond_mode", type=str, default="concat", choices=["concat", "film"], help="Fallback cond_mode when checkpoint has no config")
    parser.add_argument("--suffix", type=str, default="diffusion", help="Output suffix")
    parser.add_argument("--save_comparison", action="store_true", help="Save side-by-side input/output comparisons")
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
        f"timesteps={args.timesteps} | decoder_attn={args.decoder_attn}"
    )
    model = build_model(
        args.model_path,
        device,
        cond_mode=args.cond_mode,
        use_decoder_attn=args.decoder_attn,
    )
    sampler = DiffusionSampler(args.timesteps, device)

    for idx, path in enumerate(paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable image: {path}")
            continue

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
            match_luma_stats=(not args.no_match_luma_stats),
        )
        imgname, ext = os.path.splitext(os.path.basename(path))
        output_path = os.path.join(args.output, f"{imgname}_{args.suffix}{ext or '.png'}")
        cv2.imwrite(output_path, restored)

        if args.save_comparison:
            preview_input = cv2.resize(img, (restored.shape[1], restored.shape[0]), interpolation=cv2.INTER_NEAREST)
            comparison = np.hstack((preview_input, restored))
            comparison_path = os.path.join(args.output, f"{imgname}_{args.suffix}_compare.png")
            cv2.imwrite(comparison_path, comparison)

        print(f"[{idx + 1}/{len(paths)}] saved {output_path}")


if __name__ == "__main__":
    main()
