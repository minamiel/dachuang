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


def build_model(model_path, device, cond_mode="concat"):
    ckpt = torch.load(model_path, map_location=device)
    model_cfg = {}
    model_state = ckpt
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model_state = ckpt["model_state"]
        model_cfg = ckpt.get("config", {}) or {}

    ckpt_cond_mode = model_cfg.get("cond_mode", cond_mode)
    model = SimpleUNet(cond_mode=ckpt_cond_mode).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model


def restore_image(model, sampler, img_bgr, device, target_min_side, timesteps, outscale=4.0):
    sr_input = upscale_input(img_bgr, outscale=outscale)
    resized = smart_resize(sr_input, target_min_side=target_min_side)
    condition = image_to_tensor(resized, device)
    current = torch.randn_like(condition)

    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)), total=timesteps, leave=False):
            t = torch.tensor([i], device=device)
            current = sampler.sample_step(model, current, t, condition)

    return tensor_to_image(current)


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

    print(f"Start diffusion inference | device={device} | images={len(paths)} | timesteps={args.timesteps}")
    model = build_model(args.model_path, device, cond_mode=args.cond_mode)
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
