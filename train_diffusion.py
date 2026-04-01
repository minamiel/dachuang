import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import TextSRDataset
from model_unet import SimpleUNet


class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return noisy_image, epsilon


def dice_loss(pred, target, eps=1e-6):
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1).float()
    intersection = (pred_flat * target_flat).sum(1)
    sums = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + eps) / (sums + eps)
    return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        target = target.float()
        pt = torch.where(target == 1, pred, 1 - pred)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def build_dataloader(hr_dir, hr_size, batch_size, num_workers, device):
    dataset = TextSRDataset(hr_dir, scale=4, hr_size=hr_size, augment=True, blur_prob=0.5, noise_std=0.0)
    pin_memory = device.startswith("cuda")
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataset, dataloader, pin_memory


def train(
    cond_mode="concat",
    batch_size=8,
    epochs=1000,
    hr_size=256,
    train_size=128,
    lr=1e-4,
    resume=False,
    device=None,
    lambda_seg=0.0,
    num_workers=4,
    hr_dir="dataset/HR",
    save_dir="experiments",
    experiment_name="diffusion_textzoom_bs8",
    save_every=10,
    archive_every=100,
):
    device_name = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    latest_path = os.path.join(save_dir, f"{experiment_name}_latest.pth")

    print(
        f"Start diffusion training | device={device_name} | cond_mode={cond_mode} | "
        f"batch={batch_size} | epochs={epochs} | hr_size={hr_size} | train_size={train_size} | "
        f"lambda_seg={lambda_seg}"
    )

    dataset, dataloader, pin_memory = build_dataloader(
        hr_dir=hr_dir,
        hr_size=hr_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_name,
    )
    print(
        f"Dataset size={len(dataset)} | hr_dir={hr_dir} | num_workers={num_workers} | "
        f"pin_memory={pin_memory}"
    )
    print(f"Checkpoint latest path={latest_path}")

    model = SimpleUNet(cond_mode=cond_mode).to(device_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    focal_loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)

    if resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device_name)
        model.load_state_dict(ckpt)
        print(f"Resumed from {latest_path}")

    scheduler = DiffusionScheduler(num_timesteps=1000, device=device_name)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        avg_loss = 0.0

        for batch in pbar:
            hr_imgs = batch["HR"].to(device_name)
            lr_imgs = batch["LR"].to(device_name)

            mask_gt = batch.get("mask", None)
            if mask_gt is not None:
                mask_gt = mask_gt.to(device_name)
                if mask_gt.max() > 1.0:
                    mask_gt = (mask_gt / 255.0).clamp(0.0, 1.0)

            if train_size != hr_size:
                hr_imgs = F.interpolate(hr_imgs, size=(train_size, train_size), mode="bilinear")

            lr_upsampled = F.interpolate(lr_imgs, size=(train_size, train_size), mode="bilinear")
            t = torch.randint(0, scheduler.num_timesteps, (hr_imgs.shape[0],), device=device_name).long()
            noisy_hr, noise_target = scheduler.add_noise(hr_imgs, t)

            if cond_mode == "concat":
                model_input = torch.cat((noisy_hr, lr_upsampled), dim=1)
                noise_pred, mask_pred = model(model_input, t)
            elif cond_mode == "film":
                noise_pred, mask_pred = model(noisy_hr, t, cond=lr_imgs)
            else:
                raise ValueError(f"Unknown cond_mode: {cond_mode}")

            diffusion_loss = loss_fn(noise_pred, noise_target)

            if mask_gt is not None and lambda_seg > 0:
                if mask_pred.shape[-2:] != mask_gt.shape[-2:]:
                    mask_pred = F.interpolate(
                        mask_pred,
                        size=mask_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                seg_loss = dice_loss(mask_pred, mask_gt) + focal_loss_fn(mask_pred, mask_gt)
                total_loss = diffusion_loss + lambda_seg * seg_loss
            else:
                seg_loss = torch.tensor(0.0, device=device_name)
                total_loss = diffusion_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            avg_loss += total_loss.item()
            pbar.set_postfix(MSE=diffusion_loss.item(), Lseg=seg_loss.item())

        print(f"Epoch {epoch + 1} done | avg_loss={avg_loss / len(dataloader):.6f}")

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), latest_path)
            print(f"Saved latest checkpoint at epoch {epoch + 1}")
            if archive_every > 0 and (epoch + 1) % archive_every == 0:
                archive_path = os.path.join(save_dir, f"{experiment_name}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), archive_path)
                print(f"Saved archive checkpoint: {archive_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_mode", type=str, default="concat", choices=["concat", "film"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hr_size", type=int, default=256, help="HR crop size used by the dataset")
    parser.add_argument("--train_size", type=int, default=128, help="Training size after optional resize")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lambda_seg", type=float, default=0.0, help="Segmentation loss weight")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for Linux training")
    parser.add_argument("--hr_dir", type=str, default="dataset/HR", help="Directory containing HR text crops")
    parser.add_argument("--save_dir", type=str, default="experiments", help="Directory for checkpoints")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="diffusion_textzoom_bs8",
        help="Checkpoint file prefix",
    )
    parser.add_argument("--save_every", type=int, default=10, help="Save latest checkpoint every N epochs")
    parser.add_argument(
        "--archive_every",
        type=int,
        default=100,
        help="Save numbered checkpoints every N epochs; 0 disables archive checkpoints",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda or cpu")
    args = parser.parse_args()

    train(
        cond_mode=args.cond_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hr_size=args.hr_size,
        train_size=args.train_size,
        lr=args.lr,
        resume=args.resume,
        device=args.device,
        lambda_seg=args.lambda_seg,
        num_workers=args.num_workers,
        hr_dir=args.hr_dir,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        save_every=args.save_every,
        archive_every=args.archive_every,
    )
