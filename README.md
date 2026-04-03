# 文本图像超分（扩散模型）

本仓库当前支持一条可在本地运行的完整流程：

1. 从 TextZoom 的 LMDB 提取 HR 图片
2. 生成三联数据（`HR/LR/masks`，其中 LR 由 Real-ESRGAN 风格退化生成）
3. 训练扩散模型
4. 推理并导出结果
5. 使用统一脚本对比 `input_x4_nearest / bicubic / diffusion`

---

>`input_x4_nearest`：把输入图像直接做 最近邻插值 放大到 4 倍（x4）。不会新增细节，只是把像素块放大
>`bicubic`：用 双三次插值（Bicubic interpolation） 放大图像

input_x4_nearest：粗糙放大参考
bicubic：平滑插值参考
diffusion：模型恢复结果
要判断模型有没有价值，就看 diffusion 是否在文字边缘、断笔、可读性上明显优于前两者。

目前diffusion 是核心模型；input_x4_nearest 和 bicubic 只是对比基线。
未完全实现：严格定义的固定倍率超分（如“输入必定 x4 输出”）

---

> 建议环境：Windows + Conda（`pytorch` 环境），RTX 4060 Laptop（8GB 显存）可跑通本流程。

---

## 1. 环境准备

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
```

---

## 2. 从 TextZoom 提取 HR 图像到 `dataset/HR`

TextZoom 目录示例：

- `TextZoom/train1`（`data.mdb`, `lock.mdb`）
- `TextZoom/train2`（`data.mdb`, `lock.mdb`）
- `TextZoom/test/...`（用于后续评估，可暂不参与训练）

执行命令：

```powershell
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train1 --out_dir .\dataset\HR --prefix train1 --only_hr
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train2 --out_dir .\dataset\HR --prefix train2 --only_hr
```

参数说明：

- `--lmdb_dir`：输入 LMDB 目录
- `--out_dir`：输出图片目录
- `--prefix`：输出文件名前缀，避免重名
- `--only_hr`：仅提取包含 HR 标记的图像 key

检查是否提取成功：

```powershell
Test-Path .\dataset\HR
(Get-ChildItem .\dataset\HR -File -Recurse | Measure-Object).Count
```

---

## 3. 生成三联数据（HR/LR/masks）

脚本：`tools/make_triplet_from_hr.py`

### 3.1 无 mask（先跑通）

```powershell
python .\tools\make_triplet_from_hr.py `
	--hr_dir .\dataset\HR `
	--out_root .\dataset_triplet\train `
	--scale 4
```

### 3.2 有 mask（同名）

```powershell
python .\tools\make_triplet_from_hr.py `
	--hr_dir .\dataset\HR `
	--mask_dir .\dataset\masks `
	--out_root .\dataset_triplet\train `
	--scale 4
```

输出结构：

```text
dataset_triplet/
	train/
		HR/
		LR/
		masks/   # 仅在传入 --mask_dir 时生成
```

关键参数说明：

- `--scale 4`：按 x4 关系生成 LR（尺寸约为 HR 的 1/4）
- `--mask_dir`：可选，复制同名 mask 形成三联

---

## 4. 本地训练（1-3 小时模板）

命令：

```powershell
python .\train_diffusion.py `
	--cond_mode concat `
	--batch_size 4 `
	--epochs 30 `
	--scale 4 `
	--hr_size 128 `
	--train_size 128 `
	--lr 1e-4 `
	--lambda_seg 0 `
	--num_workers 2 `
	--hr_dir .\dataset_triplet\train\HR `
	--lr_dir .\dataset_triplet\train\LR `
	--mask_dir .\dataset_triplet\train\masks `
	--save_dir .\model `
	--experiment_name diffusion_local_1to3h `
	--save_best `
	--save_every 2
```

参数说明：

- `--cond_mode concat`：条件输入模式（推荐起步使用）
- `--batch_size`：显存敏感参数，OOM 时优先减小
- `--scale`：超分倍率（默认 4，可改为 2）
- `--hr_size / --train_size`：训练分辨率
- `--lambda_seg 0`：无可靠 mask 时建议先设 0
- `--hr_dir / --lr_dir / --mask_dir`：三联数据目录
- `--experiment_name`：输出模型文件名前缀
- `--save_best`：保存训练损失最低的 best checkpoint

训练产物示例：

- `model/diffusion_local_1to3h_latest.pth`

---

## 5. 推理

```powershell
python .\inference_diffusion.py `
	-i .\eval_inputs `
	-o .\eval_outputs\diffusion_local_1to3h `
	--model_path .\model\diffusion_local_1to3h_latest.pth `
	--outscale 4 `
	--timesteps 120
```

参数说明：

- `-i`：输入图片目录
- `-o`：输出目录
- `--model_path`：训练得到的 checkpoint
- `--outscale`：固定超分倍率
- `--timesteps`：采样步数（更高通常更慢但可能更好）

---

## 6. 统一对比评测（输入放大 vs bicubic vs diffusion）

```powershell
python .\tools\evaluate_text_models.py `
	--input_dir .\eval_inputs `
	--output_dir .\eval_outputs\cmp_local `
	--methods bicubic,diffusion `
	--outscale 4 `
	--diffusion_model_path .\model\diffusion_local_1to3h_latest.pth `
	--diffusion_outscale 4 `
	--diffusion_steps 80 `
	--diffusion_min_side 256 `
	--diffusion_fallback_min_side 192
```

打开对比图目录：

```powershell
Start-Process .\eval_outputs\cmp_local\comparisons
```

说明：

- `--methods bicubic,diffusion`：只跑双三次 + 扩散，不依赖 `basicsr`
- `--diffusion_min_side`：扩散推理尺度，过大可能 OOM
- `--diffusion_fallback_min_side`：OOM 自动回退尺度

---

## 7. 常见问题

### Q1: `ModuleNotFoundError: basicsr`

- 若只跑 `bicubic,diffusion`，当前脚本已支持不安装 `basicsr`
- 若要加 `realesrgan` 方法，请安装：

```powershell
python -m pip install basicsr
```

### Q2: CUDA OOM

优先按顺序调整：

1. 训练：`--batch_size 4 -> 2`
2. 训练：`--train_size 128 -> 96`
3. 评测：`--diffusion_min_side 256 -> 192`
4. 评测：`--diffusion_steps 80 -> 60`

### Q3: 看不出效果差异

- 提升训练轮数（如 `epochs 80+`）
- 评测提高采样步数（如 `diffusion_steps 120~200`）
- 对比时重点看文字边缘、断笔、重影、可读性

---

## 8. 最短复现路径

```powershell
conda activate pytorch
python -m pip install -r requirements.txt
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train1 --out_dir .\dataset\HR --prefix train1 --only_hr
python .\tools\extract_lmdb_images_generic.py --lmdb_dir .\TextZoom\train2 --out_dir .\dataset\HR --prefix train2 --only_hr
python .\tools\make_triplet_from_hr.py --hr_dir .\dataset\HR --out_root .\dataset_triplet\train --scale 4
python .\train_diffusion.py --cond_mode concat --batch_size 4 --epochs 30 --scale 4 --hr_size 128 --train_size 128 --lr 1e-4 --lambda_seg 0 --num_workers 2 --hr_dir .\dataset_triplet\train\HR --lr_dir .\dataset_triplet\train\LR --save_dir .\model --experiment_name diffusion_local_1to3h --save_every 2
python .\tools\evaluate_text_models.py --input_dir .\eval_inputs --output_dir .\eval_outputs\cmp_local --methods bicubic,diffusion --outscale 4 --diffusion_model_path .\model\diffusion_local_1to3h_latest.pth --diffusion_outscale 4 --diffusion_steps 80 --diffusion_min_side 256 --diffusion_fallback_min_side 192
```

> 训练完成后会在 `./model` 下生成：
>
> - `*_latest.pth`：最新 checkpoint
> - `*_best.pth`：按最低 epoch 平均损失自动选择的最佳 checkpoint（需开启 `--save_best`）
> - `*_train_log.csv`：训练日志（每 epoch）
> - `*_summary.json`：实验摘要（best/loss/路径等）

---

## 9. 服务器推荐命令

> 说明：`train_diffusion.py` 已支持 `--ddp`，请使用 `torchrun` 启动多卡同步训练。

### 9.1 6卡 DDP 训练（质量优先）

```bash
torchrun --nproc_per_node=6 train_diffusion.py \
	--ddp \
	--dist_backend nccl \
	--cond_mode concat \
	--batch_size 16 \
	--epochs 200 \
	--scale 4 \
	--hr_size 256 \
	--train_size 256 \
	--lr 8e-5 \
	--lambda_seg 0.2 \
	--num_workers 8 \
	--hr_dir ./dataset_triplet/train/HR \
	--lr_dir ./dataset_triplet/train/LR \
	--mask_dir ./dataset_triplet/train/masks \
	--save_dir ./model \
	--experiment_name diffusion_ddp_6x4090 \
	--save_best \
	--save_every 5 \
	--archive_every 20
```

### 9.2 6卡 DDP 续训

```bash
torchrun --nproc_per_node=6 train_diffusion.py \
	--ddp \
	--dist_backend nccl \
	--resume \
	--cond_mode concat \
	--batch_size 16 \
	--epochs 200 \
	--scale 4 \
	--hr_size 256 \
	--train_size 256 \
	--lr 8e-5 \
	--lambda_seg 0.2 \
	--num_workers 8 \
	--hr_dir ./dataset_triplet/train/HR \
	--lr_dir ./dataset_triplet/train/LR \
	--mask_dir ./dataset_triplet/train/masks \
	--save_dir ./model \
	--experiment_name diffusion_ddp_6x4090 \
	--save_best \
	--save_every 5 \
	--archive_every 20
```

---

### 9.3 服务器测试（单模型推理）

```bash
CUDA_VISIBLE_DEVICES=0 python inference_diffusion.py \
	-i ./eval_inputs \
	-o ./eval_outputs/diffusion_ddp_6x4090 \
	--model_path ./model/diffusion_ddp_6x4090_latest.pth \
	--outscale 4 \
	--timesteps 200 \
	--target_min_side 384
```

- 质量优先：`timesteps=200~300`
- 速度优先：`timesteps=80~120`

### 9.4 服务器统一评测（推荐）

```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/evaluate_text_models.py \
	--input_dir ./eval_inputs \
	--output_dir ./eval_outputs/cmp_ddp_6x4090 \
	--methods bicubic,diffusion \
	--outscale 4 \
	--diffusion_model_path ./model/diffusion_ddp_6x4090_latest.pth \
	--diffusion_outscale 4 \
	--diffusion_steps 160 \
	--diffusion_min_side 384 \
	--diffusion_fallback_min_side 256
```

如果有 GT，可加：

```bash
--gt_dir ./eval_gt --metrics_csv metrics_ddp_6x4090.csv
```

---

### 9.5 如何从当前实验选最佳

优先级建议：

1. `metrics.csv`（若有 GT）：按 `SSIM`、`PSNR` 选 Top-2 checkpoint
2. 人工目检：重点看文字边缘、断笔、重影、可读性
3. 最终保留：`*_best.pth` + 对应训练命令（可复现）

---

### 9.6 服务器环境一键检查清单（GPU）

> 目标：在开跑训练/评测前，快速确认驱动、CUDA、Python 环境、Torch/Paddle GPU 能力和 PP-OCRv5 模型目录是否就绪。

```powershell
nvidia-smi
python -V
python -m pip -V
python -m pip list | findstr /I "torch torchvision paddle paddleocr opencv"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
python -c "import paddle; print('paddle', paddle.__version__, 'compiled_with_cuda', paddle.is_compiled_with_cuda(), 'device', paddle.device.get_device())"
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "from pathlib import Path; p=Path('./PPOCRv5'); print('ppocr_root_exists', p.exists()); print('det_dirs', [x.name for x in p.glob('*det*')] if p.exists() else []); print('rec_dirs', [x.name for x in p.glob('*rec*')] if p.exists() else []); print('cls_dirs', [x.name for x in p.glob('*cls*')] if p.exists() else [])"
python .\tools\evaluate_ocr_metrics.py --help
```

建议判定标准：

- `nvidia-smi` 能看到 GPU 与驱动版本。
- `torch.cuda.is_available()` 为 `True`，且 `device_count >= 1`。
- `paddle.is_compiled_with_cuda()` 为 `True`，`device` 显示 `gpu:*`。
- `./PPOCRv5` 下至少有 det/rec/cls 三类目录。

> 关于 `requirements.txt`：**不建议**把 `paddlepaddle-gpu` 固定写死在通用依赖里。
>
> 原因：`paddlepaddle-gpu` 轮子与 CUDA/驱动强绑定，不同服务器（如 CUDA 11.8 / 12.2）需要不同安装源和版本；写死后很容易出现安装成功但运行失败、或直接无法安装。
>
> 推荐策略：`requirements.txt` 保留通用依赖（如 `paddleocr`），`paddlepaddle-gpu` 在服务器按当前 CUDA 版本单独安装。

---

## 10. OCR 指标评测（Acc / CER / WER）

脚本：`tools/evaluate_ocr_metrics.py`

用途：

- 对预测图目录运行 OCR
- 与 GT 文本标签比对
- 输出逐样本明细和汇总指标（适合交付报告）

先安装依赖（推荐 PaddleOCR）：

```powershell
python -m pip install paddleocr
```

> 若希望使用 GPU，请额外安装匹配 CUDA 的 `paddlepaddle-gpu` 版本。

将 PP-OCRv5_server 模型放到 `./PPOCRv5`，建议目录示例：

```text
./PPOCRv5/
  ch_PP-OCRv5_server_det/
  ch_PP-OCRv5_server_rec/
  ch_ppocr_mobile_v2.0_cls/
```

命令示例：

```powershell
python .\tools\evaluate_ocr_metrics.py `
	--pred_dir .\eval_outputs\diffusion_local_1to3h `
	--gt_csv .\eval_inputs\labels.csv `
	--image_col image `
	--text_col text `
	--suffix _diffusion `
	--ocr_backend paddleocr `
	--ppocr_root .\PPOCRv5 `
	--lang ch `
	--use_angle_cls `
	--output_csv .\eval_outputs\ocr_metrics_detail.csv `
	--output_json .\eval_outputs\ocr_metrics_summary.json
```

参数说明：

- `--pred_dir`：预测图片目录
- `--gt_csv`：GT 文本标注 CSV
- `--image_col` / `--text_col`：CSV 中图片列和文本列
- `--suffix`：预测文件名后缀（例如 `eval_001_diffusion.png` 的 `_diffusion`）
- `--ocr_backend`：OCR后端（默认 `paddleocr`，兼容 `tesseract`）
- `--ppocr_root`：PP-OCR 模型根目录（默认 `./PPOCRv5`）
- `--det_model_dir` / `--rec_model_dir` / `--cls_model_dir`：显式指定模型目录（可选）
- `--lang`：OCR语言（如 `ch`、`en`，也兼容 `eng`、`chi_sim`）
- `--use_angle_cls`：启用方向分类器（推荐开启）
- `--use_gpu`：使用 GPU 推理（需安装 `paddlepaddle-gpu`）

输出文件：

- `ocr_metrics_detail.csv`：每张图的 `pred_text / gt_text / CER / WER / exact_match`
- `ocr_metrics_summary.json`：整体 `accuracy / CER / WER / 样本数`
