import argparse
import csv
import json
import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataloader import TextSRDataset
from model_unet import SimpleUNet

# ==========================================================
# 本文件作用：
# 1) 训练一个条件扩散模型（SimpleUNet）用于文本图像超分。
# 2) 使用前向加噪 + 噪声预测（MSE）作为扩散主损失。
# 3) 可选叠加文字区域分割辅助损失（Dice + Focal）提升文字边缘恢复。
# 4) 提供命令行参数，支持断点续训与周期性保存 checkpoint。
# ==========================================================


class DiffusionScheduler:
    """扩散过程中的噪声调度器。

    这里采用线性 beta 调度：
    - beta_t: 每一步加入噪声的强度
    - alpha_t = 1 - beta_t
    - alpha_hat_t = _{i=1..t} alpha_i （累计保真系数）

    在训练时，我们从随机 t 直接采样 x_t，而不是逐步模拟整条链，
    这样可以更高效地训练模型去预测噪声 epsilon。
    """

    def __init__(self, num_timesteps=1000, device="cuda"):
        # 扩散总步数（常见取值：1000）
        self.num_timesteps = num_timesteps
        self.device = device
        # 线性噪声日程：从很小噪声到较大噪声
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        # alpha 的累乘，表示从 x0 到 xt 的信号保留比例
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        """按时间步 t 给干净图像 x 加噪。

        参数：
            x: 干净图像 x0, 形状 [B, C, H, W]
            t: 每个样本的时间步，形状 [B]

        返回：
            noisy_image: x_t
            epsilon: 实际注入噪声（训练监督目标）
        """
        # 取出每个样本对应 t 的 sqrt(alpha_hat_t)，并扩展到图像维度
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # 取出 sqrt(1 - alpha_hat_t)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        # 标准高斯噪声 epsilon
        epsilon = torch.randn_like(x)
        # x_t = sqrt(alpha_hat_t) * x0 + sqrt(1-alpha_hat_t) * epsilon
        noisy_image = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon
        return noisy_image, epsilon


def dice_loss(pred, target, eps=1e-6):
    """Dice 损失：用于衡量预测 mask 与真值 mask 的重叠程度。

    pred/target 都会被展平到 [B, N]。
    返回值越小越好，0 表示完全重叠。
    """
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1).float()
    # 交集
    intersection = (pred_flat * target_flat).sum(1)
    # 预测面积 + 真值面积
    sums = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + eps) / (sums + eps)
    return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    """二分类 Focal Loss。

    用于缓解前景/背景不平衡：
    - gamma 越大，越关注难样本；
    - alpha 用于类别权重平衡。
    """

    def __init__(self, gamma=2.0, alpha=0.25, eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        # 避免 log(0) 导致数值问题
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        target = target.float()
        # pt 表示“当前标签对应的预测概率”
        pt = torch.where(target == 1, pred, 1 - pred)
        # 标准 BCE
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        # Focal 调制：((1-pt)^gamma) 抑制易样本
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def build_dataloader(
    hr_dir,
    lr_dir,
    mask_dir,
    scale,
    hr_size,
    batch_size,
    num_workers,
    device,
    distributed=False,
    rank=0,
    world_size=1,
):
    """构建训练集与 DataLoader。

    - 数据来源：TextSRDataset（从 HR 构造 LR/HR 对）
    - 在 CUDA 下启用 pin_memory，并可开启 cudnn.benchmark 提升吞吐
    - num_workers > 0 时使用 persistent_workers / prefetch_factor
    """
    dataset = TextSRDataset(
        hr_dir,
        lr_dir=lr_dir,
        mask_dir=mask_dir,
        scale=scale,
        hr_size=hr_size,
        augment=True,
        blur_prob=0.5,
        noise_std=0.0,
    )
    # pin_memory=True 可加速 Host->GPU 传输
    pin_memory = device.startswith("cuda")
    if device.startswith("cuda"):
        # 输入尺寸相对稳定时可让 cuDNN 自动选择更优算法
        torch.backends.cudnn.benchmark = True

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        # 持久化 worker，避免每个 epoch 重建进程
        loader_kwargs["persistent_workers"] = True
        # 每个 worker 预取 2 个 batch
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataset, dataloader, pin_memory, sampler


def setup_distributed_training(enable_ddp: bool, backend: str = "nccl"):
    if not enable_ddp:
        return False, 0, 1, 0

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA. Please disable --ddp or run on GPU server.")

    required_envs = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    missing = [k for k in required_envs if k not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP mode requires torchrun launch environment. "
            f"Missing env vars: {missing}. "
            "Please launch with: torchrun --nproc_per_node=<gpu_num> train_diffusion.py --ddp ..."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def cleanup_distributed_training(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def train(
    cond_mode="concat",
    batch_size=8,
    epochs=1000,
    hr_size=256,
    train_size=128,
    lr=1e-4,
    scale=4,
    resume=False,
    device=None,
    lambda_seg=0.0,
    num_workers=4,
    hr_dir="dataset/HR",
    lr_dir=None,
    mask_dir=None,
    save_dir="model",
    experiment_name="diffusion_textzoom_bs8",
    save_every=10,
    archive_every=100,
    save_best=True,
    ddp=False,
    dist_backend="nccl",
):
    """扩散模型训练主函数。

    训练流程：
    1) 构建数据与模型
    2) 随机采样时间步 t，对 HR 加噪得到 x_t
    3) 模型预测噪声 epsilon_hat
    4) 计算扩散损失（MSE），可选叠加分割损失
    5) 反向传播与参数更新
    6) 周期保存 checkpoint
    """

    is_distributed, rank, world_size, local_rank = setup_distributed_training(ddp, dist_backend)
    is_main_process = rank == 0

    def save_checkpoint(path: str, epoch_idx: int, avg_epoch_loss: float):
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        payload = {
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch_idx,
            "avg_loss": float(avg_epoch_loss),
            "config": {
                "cond_mode": cond_mode,
                "batch_size": batch_size,
                "epochs": epochs,
                "hr_size": hr_size,
                "train_size": train_size,
                "lr": lr,
                "scale": scale,
                "lambda_seg": lambda_seg,
                "num_workers": num_workers,
                "hr_dir": hr_dir,
                "lr_dir": lr_dir,
                "mask_dir": mask_dir,
                "save_dir": save_dir,
                "experiment_name": experiment_name,
                "save_every": save_every,
                "archive_every": archive_every,
                "ddp": ddp,
                "world_size": world_size,
            },
        }
        torch.save(payload, path)

    def parse_checkpoint(ckpt_obj: Any) -> Dict[str, Any]:
        # 新版格式：包含 model_state/config 等字段
        if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
            return ckpt_obj
        # 旧版兼容：直接是 state_dict
        if isinstance(ckpt_obj, dict):
            return {
                "model_state": ckpt_obj,
                "optimizer_state": None,
                "epoch": -1,
                "avg_loss": None,
                "config": {},
            }
        raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")

    # 自动设备选择：DDP 下固定到本进程 local_rank 对应 GPU
    if is_distributed:
        device_name = f"cuda:{local_rank}"
    else:
        device_name = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)
    # latest checkpoint 路径（会被周期性覆盖）
    latest_path = os.path.join(save_dir, f"{experiment_name}_latest.pth")
    best_path = os.path.join(save_dir, f"{experiment_name}_best.pth")
    train_log_csv = os.path.join(save_dir, f"{experiment_name}_train_log.csv")
    summary_json = os.path.join(save_dir, f"{experiment_name}_summary.json")

    if is_main_process:
        print(
            f"Start diffusion training | device={device_name} | cond_mode={cond_mode} | "
            f"batch={batch_size} | epochs={epochs} | hr_size={hr_size} | train_size={train_size} | "
            f"lambda_seg={lambda_seg} | scale={scale} | ddp={is_distributed} | world_size={world_size}"
        )

    dataset, dataloader, pin_memory, train_sampler = build_dataloader(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        mask_dir=mask_dir,
    scale=scale,
        hr_size=hr_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device_name,
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process:
        print(
            f"Dataset size={len(dataset)} | hr_dir={hr_dir} | num_workers={num_workers} | "
            f"pin_memory={pin_memory} | lr_dir={lr_dir} | mask_dir={mask_dir} | scale={scale}"
        )
        print(f"Checkpoint latest path={latest_path}")

    # 构建条件 U-Net
    model = SimpleUNet(cond_mode=cond_mode).to(device_name)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # AdamW：对扩散训练通常比较稳定
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # 扩散主损失：预测噪声的 MSE
    loss_fn = nn.MSELoss()
    # 分割辅助损失：Focal
    focal_loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)

    # 断点续训：加载模型参数 + 优化器状态（若可用）
    start_epoch = 0
    best_loss = float("inf")
    if resume and os.path.exists(latest_path):
        ckpt_raw = torch.load(latest_path, map_location=device_name)
        ckpt = parse_checkpoint(ckpt_raw)
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        ckpt_cfg: Optional[Dict[str, Any]] = ckpt.get("config") or {}
        ckpt_cond_mode = ckpt_cfg.get("cond_mode", "unknown")
        if is_main_process:
            print(
                f"Resumed from {latest_path} | start_epoch={start_epoch + 1} | "
                f"ckpt_cond_mode={ckpt_cond_mode}"
            )

    if resume and os.path.exists(best_path):
        try:
            best_ckpt_raw = torch.load(best_path, map_location=device_name)
            best_ckpt = parse_checkpoint(best_ckpt_raw)
            ckpt_best_loss = best_ckpt.get("avg_loss", None)
            if ckpt_best_loss is not None:
                best_loss = float(ckpt_best_loss)
                if is_main_process:
                    print(f"Loaded historical best loss: {best_loss:.6f}")
        except Exception as err:
            if is_main_process:
                print(f"Warning: failed to read best checkpoint metadata: {err}")

    if is_main_process and (not resume or not os.path.exists(train_log_csv)):
        with open(train_log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "avg_loss", "best_loss", "lr", "world_size", "scale"])

    # 扩散噪声日程
    scheduler = DiffusionScheduler(num_timesteps=1000, device=device_name)

    # =========================
    #        训练循环
    # =========================
    for epoch in range(start_epoch, epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not is_main_process)
        avg_loss = 0.0
        num_batches = 0

        for batch in pbar:
            # 读取 HR / LR 图像
            hr_imgs = batch["HR"].to(device_name)
            lr_imgs = batch["LR"].to(device_name)

            # 可选读取文字区域 mask（若数据集提供）
            mask_gt = batch.get("mask", None)
            if mask_gt is not None:
                mask_gt = mask_gt.to(device_name)
                # 如果是 0~255，归一化到 0~1
                if mask_gt.max() > 1.0:
                    mask_gt = (mask_gt / 255.0).clamp(0.0, 1.0)

            # 可选将训练目标缩放到 train_size（与数据裁剪尺寸解耦）
            if train_size != hr_size:
                hr_imgs = F.interpolate(hr_imgs, size=(train_size, train_size), mode="bilinear")

            # 将 LR 上采样到训练尺寸，供条件输入使用
            lr_upsampled = F.interpolate(lr_imgs, size=(train_size, train_size), mode="bilinear")

            # 为每个样本随机采样扩散时间步 t
            t = torch.randint(0, scheduler.num_timesteps, (hr_imgs.shape[0],), device=device_name).long()
            # 按 t 对 HR 加噪，得到 noisy_hr 与监督噪声 noise_target
            noisy_hr, noise_target = scheduler.add_noise(hr_imgs, t)

            # 条件方式 1：concat（将 noisy_hr 与 upsampled LR 在通道维拼接）
            if cond_mode == "concat":
                model_input = torch.cat((noisy_hr, lr_upsampled), dim=1)
                noise_pred, mask_pred = model(model_input, t)
            # 条件方式 2：film（将 LR 作为 cond 传入模型内部调制）
            elif cond_mode == "film":
                noise_pred, mask_pred = model(noisy_hr, t, cond=lr_imgs)
            else:
                raise ValueError(f"Unknown cond_mode: {cond_mode}")

            # 扩散主损失：预测噪声 vs 真实噪声
            diffusion_loss = loss_fn(noise_pred, noise_target)

            # 若存在 mask 且启用分割损失权重，则叠加辅助监督
            if mask_gt is not None and lambda_seg > 0:
                # 尺寸不匹配时，将预测 mask 对齐到 GT 尺寸
                if mask_pred.shape[-2:] != mask_gt.shape[-2:]:
                    mask_pred = F.interpolate(
                        mask_pred,
                        size=mask_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                # 分割损失 = Dice + Focal
                seg_loss = dice_loss(mask_pred, mask_gt) + focal_loss_fn(mask_pred, mask_gt)
                # 总损失 = 扩散损失 + λ * 分割损失
                total_loss = diffusion_loss + lambda_seg * seg_loss
            else:
                # 未启用分割监督时，seg_loss 仅用于日志展示
                seg_loss = torch.tensor(0.0, device=device_name)
                total_loss = diffusion_loss

            # 标准反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            avg_loss += total_loss.item()
            num_batches += 1
            # tqdm 实时展示当前 batch 的扩散损失与分割损失
            if is_main_process:
                pbar.set_postfix(MSE=diffusion_loss.item(), Lseg=seg_loss.item())

        # 每个 epoch 的平均损失（DDP 下做全局汇总）
        if is_distributed:
            stats = torch.tensor([avg_loss, float(num_batches)], device=device_name)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_loss, global_batches = stats.tolist()
            avg_epoch_loss = global_loss / max(global_batches, 1.0)
        else:
            avg_epoch_loss = avg_loss / max(num_batches, 1)

        if is_main_process:
            print(f"Epoch {epoch + 1} done | avg_loss={avg_epoch_loss:.6f}")

            # 训练日志：每个 epoch 记录一行
            cur_lr = optimizer.param_groups[0].get("lr", lr)
            with open(train_log_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{avg_epoch_loss:.8f}", f"{best_loss:.8f}", f"{cur_lr:.10f}", world_size, scale])

            # 自动选优：保存训练损失最低的 checkpoint
            if save_best and avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_checkpoint(best_path, epoch, avg_epoch_loss)
                print(f"New best checkpoint at epoch {epoch + 1} | best_loss={best_loss:.6f}")

            # 同步写 summary，便于工程化追踪
            summary = {
                "experiment_name": experiment_name,
                "latest_path": os.path.abspath(latest_path),
                "best_path": os.path.abspath(best_path),
                "train_log_csv": os.path.abspath(train_log_csv),
                "summary_json": os.path.abspath(summary_json),
                "epoch": epoch + 1,
                "avg_loss": float(avg_epoch_loss),
                "best_loss": float(best_loss),
                "scale": scale,
                "world_size": world_size,
                "ddp": bool(is_distributed),
                "cond_mode": cond_mode,
            }
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        # 周期性保存 latest checkpoint
        if is_main_process and (epoch + 1) % save_every == 0:
            save_checkpoint(latest_path, epoch, avg_epoch_loss)
            print(f"Saved latest checkpoint at epoch {epoch + 1}")
            # 可选保存归档 checkpoint（防止 latest 被覆盖后无法回退）
            if archive_every > 0 and (epoch + 1) % archive_every == 0:
                archive_path = os.path.join(save_dir, f"{experiment_name}_epoch_{epoch + 1}.pth")
                save_checkpoint(archive_path, epoch, avg_epoch_loss)
                print(f"Saved archive checkpoint: {archive_path}")

    # 训练结束时强制落盘一次 latest，避免 epochs 与 save_every 不整除时没有最终模型
    if is_main_process and epochs > 0:
        save_checkpoint(latest_path, epochs - 1, avg_epoch_loss if 'avg_epoch_loss' in locals() else 0.0)
        print(f"Training completed. Final latest checkpoint saved to: {latest_path}")

    cleanup_distributed_training(is_distributed)


if __name__ == "__main__":
    # 命令行参数：用于灵活控制训练配置
    parser = argparse.ArgumentParser()
    # 条件注入模式：concat / film
    parser.add_argument("--cond_mode", type=str, default="concat", choices=["concat", "film"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    # 数据集 HR 裁剪尺寸
    parser.add_argument("--hr_size", type=int, default=256, help="HR crop size used by the dataset")
    # 实际训练分辨率（可与 hr_size 不同）
    parser.add_argument("--train_size", type=int, default=128, help="Training size after optional resize")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor, e.g. 2 or 4")
    # 是否从 latest checkpoint 恢复
    parser.add_argument("--resume", action="store_true")
    # 分割辅助损失权重，0 表示关闭
    parser.add_argument("--lambda_seg", type=float, default=0.0, help="Segmentation loss weight")
    # DataLoader 并行读取线程数
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for Linux training")
    # 数据目录（HR 文本图）
    parser.add_argument("--hr_dir", type=str, default="dataset/HR", help="Directory containing HR text crops")
    parser.add_argument("--lr_dir", type=str, default=None, help="Optional directory containing paired LR images")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional directory containing paired masks")
    # checkpoint 保存目录
    parser.add_argument("--save_dir", type=str, default="model", help="Directory for checkpoints")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="diffusion_textzoom_bs8",
        # 文件前缀：例如 diffusion_textzoom_bs8_latest.pth
        help="Checkpoint file prefix",
    )
    # 每 N 个 epoch 更新一次 latest
    parser.add_argument("--save_every", type=int, default=10, help="Save latest checkpoint every N epochs")
    parser.add_argument(
        "--archive_every",
        type=int,
        default=100,
        # 每 N 个 epoch 额外保存一个带 epoch 后缀的归档
        help="Save numbered checkpoints every N epochs; 0 disables archive checkpoints",
    )
    parser.add_argument("--save_best", action="store_true", help="Save best checkpoint by lowest epoch avg loss")
    # 手动指定设备，如 cuda / cpu
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda or cpu")
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel (launch with torchrun)")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend for DDP")
    args = parser.parse_args()

    # 将命令行参数传入训练函数
    train(
        cond_mode=args.cond_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hr_size=args.hr_size,
        train_size=args.train_size,
        lr=args.lr,
    scale=args.scale,
        resume=args.resume,
        device=args.device,
        lambda_seg=args.lambda_seg,
        num_workers=args.num_workers,
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        mask_dir=args.mask_dir,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        save_every=args.save_every,
        archive_every=args.archive_every,
        save_best=args.save_best,
        ddp=args.ddp,
        dist_backend=args.dist_backend,
    )
