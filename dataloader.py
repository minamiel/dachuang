import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TextImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        # 只读取常见的图片格式
        self.file_names = [f for f in os.listdir(hr_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        hr_path = os.path.join(self.hr_dir, file_name)
        lr_path = os.path.join(self.lr_dir, file_name)

        # 1. 读取图片
        img_hr = cv2.imread(hr_path)
        img_lr = cv2.imread(lr_path)

        # 2. 容错
        if img_hr is None or img_lr is None:
            # 读不到就给黑图，防止崩坏
            img_hr = np.zeros((512, 512, 3), dtype=np.uint8)
            img_lr = np.zeros((128, 128, 3), dtype=np.uint8)

        # 3. 转 RGB
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

        # 4. 强制统一尺寸 (LR=128, HR=512)
        # 这一步是为了防止“Stack Error”
        img_lr = cv2.resize(img_lr, (128, 128), interpolation=cv2.INTER_CUBIC)
        img_hr = cv2.resize(img_hr, (512, 512), interpolation=cv2.INTER_CUBIC)

        # 5. 【关键修复】手动强制转 Tensor 并掉头 (H, W, C) -> (C, H, W)
        # 这种写法最稳，绝对不会错
        # 原代码是 / 255.0
        # 新代码：(数值 / 127.5) - 1.0  ---> 这样范围就变成了 -1.0 到 1.0
        img_lr = (torch.from_numpy(img_lr).permute(2, 0, 1).float() / 127.5) - 1.0
        img_hr = (torch.from_numpy(img_hr).permute(2, 0, 1).float() / 127.5) - 1.0

        return {'LR': img_lr, 'HR': img_hr}


# ==========================================
# 新增：TextSRDataset
# 功能：输入原始 HR 图像，动态生成对应的 LR（下采样 + 可选模糊 + 噪声），返回 Tensor
# 适用于文本图像超分任务（例如 HR:256 -> LR:64）
# ==========================================
class TextSRDataset(Dataset):
    def __init__(
            self,
            hr_dir,
            lr_dir=None,
            mask_dir=None,
            scale=4,
            hr_size=256,
            augment=False,
            blur_prob=0.5,
            noise_std=0.0):
        """
        hr_dir: 存放 HR 图像的目录
        scale: 下采样倍率 (例如 4 表示 256 -> 64)
        hr_size: 输出的 HR 尺寸 (会把原图 resize/crop 到 hr_size)
        augment: 是否打开数据增强（对文本有用的二值化/扰动）
        blur_prob: 生成 LR 时加入高斯模糊的概率
        noise_std: 在 LR 上加入高斯噪声的标准差 (0.0 表示不加)
        """
        self.hr_dir = hr_dir
        self.file_names = [f for f in os.listdir(hr_dir) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        self.file_names.sort()
        self.lr_dir = lr_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.hr_size = hr_size if isinstance(hr_size, int) else hr_size[0]
        self.augment = augment
        self.blur_prob = blur_prob
        self.noise_std = noise_std

    def __len__(self):
        return len(self.file_names)

    def _make_lr(self, img_hr):
        # 输入 img_hr 为 numpy RGB uint8
        # 1) 先保证 HR 是 hr_size
        img_hr = cv2.resize(img_hr, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)

        # 2) 下采样生成 LR
        lr_size = self.hr_size // self.scale
        img_lr = cv2.resize(img_hr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)

        # 3) 随机高斯模糊
        if np.random.rand() < self.blur_prob:
            # 随机核大小 (3 或 5)
            k = 3 if np.random.rand() < 0.7 else 5
            sigma = np.random.uniform(0.2, 1.5)
            img_lr = cv2.GaussianBlur(img_lr, (k, k), sigmaX=sigma)

        # 4) 可选噪声
        if self.noise_std > 0:
            noise = np.random.randn(*img_lr.shape) * (self.noise_std * 255.0)
            img_lr = img_lr.astype(np.float32) + noise
            img_lr = np.clip(img_lr, 0, 255).astype(np.uint8)

        return img_hr, img_lr

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        hr_path = os.path.join(self.hr_dir, file_name)

        img_hr = cv2.imread(hr_path)
        if img_hr is None:
            # 返回空白图（避免训练中断）
            img_hr = np.zeros((self.hr_size, self.hr_size, 3), dtype=np.uint8)

        # BGR -> RGB
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

        # 可选增强（针对文本图像：二值化/形态学变换/亮度对比度扰动）
        if self.augment:
            # 随机对比度/亮度
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-10, 10)
            img_hr = np.clip(img_hr * alpha + beta, 0, 255).astype(np.uint8)

            # 小概率做二值化（模拟扫描文本）
            if np.random.rand() < 0.2:
                gray = cv2.cvtColor(img_hr, cv2.COLOR_RGB2GRAY)
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img_hr = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

        if self.lr_dir is not None:
            lr_path = os.path.join(self.lr_dir, file_name)
            img_lr = cv2.imread(lr_path)
            if img_lr is None:
                # 兼容后缀不一致，尝试同 stem 的其他后缀
                stem = os.path.splitext(file_name)[0]
                for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                    alt = os.path.join(self.lr_dir, stem + ext)
                    img_lr = cv2.imread(alt)
                    if img_lr is not None:
                        break

            if img_lr is None:
                # LR 缺失时回退到在线退化，保证训练不中断
                img_hr, img_lr = self._make_lr(img_hr)
            else:
                # 使用预生成 LR，同时统一尺寸
                img_hr = cv2.resize(img_hr, (self.hr_size, self.hr_size), interpolation=cv2.INTER_CUBIC)
                lr_size = self.hr_size // self.scale
                img_lr = cv2.resize(img_lr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        else:
            img_hr, img_lr = self._make_lr(img_hr)

        # 转为 Tensor 并归一化到 [-1, 1]
        img_lr_t = (torch.from_numpy(img_lr).permute(2, 0, 1).float() / 127.5) - 1.0
        img_hr_t = (torch.from_numpy(img_hr).permute(2, 0, 1).float() / 127.5) - 1.0

        # 新增：尝试加载对应的 mask（dataset/masks/<filename>），如果存在则返回 'mask'
        if self.mask_dir is not None:
            mask_path_options = [
                os.path.join(self.mask_dir, file_name),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.png'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.jpg'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.jpeg'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.bmp'),
                os.path.join(self.mask_dir, os.path.splitext(file_name)[0] + '.webp'),
            ]
        else:
            mask_path_options = [
                os.path.join(os.path.dirname(self.hr_dir), 'masks', file_name),  # sibling /dataset/masks
                os.path.join(self.hr_dir, '..', 'masks', file_name),  # another attempt
                os.path.join('dataset', 'masks', file_name)  # fallback
            ]
        mask_tensor = None
        for mp in mask_path_options:
            mp_abs = os.path.abspath(mp)
            if os.path.exists(mp_abs):
                try:
                    m = cv2.imread(mp_abs, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        # ensure same size as HR; resize if needed
                        if (m.shape[0] != self.hr_size) or (m.shape[1] != self.hr_size):
                            m = cv2.resize(m, (self.hr_size, self.hr_size), interpolation=cv2.INTER_NEAREST)
                        # convert to tensor [1, H, W]
                        m_t = torch.from_numpy(m).unsqueeze(0)
                        mask_tensor = m_t
                        break
                except Exception:
                    # ignore read errors and continue
                    pass

        ret = {'LR': img_lr_t, 'HR': img_hr_t, 'fname': file_name}
        if mask_tensor is not None:
            ret['mask'] = mask_tensor

        return ret
