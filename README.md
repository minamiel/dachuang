# 基于扩散模型的文本图像超分

本仓库当前支持一条可在本地运行的完整流程：

1. 从 TextZoom 的 LMDB 提取 HR 图片
2. 生成三联数据（`HR/LR/masks`，其中 LR 由 Real-ESRGAN 风格退化生成）
3. 训练扩散模型
4. 推理并导出结果
5. 使用统一脚本对比 `input_x4_nearest / bicubic / diffusion`

---

## 任务目标定义

本项目是**基于扩散模型的文本图像超分辨率（Text Image Super-Resolution）**，目标不是只处理文字区域，而是：

- 对整张低分辨率图像进行重建与增强（文字 + 背景同时提升）
- 在整体增强的前提下，**优先保证文字区域的边缘、笔画与可读性**
- 最终输出以“给人看”为导向：清晰度显著提升，同时尽量保持原图色彩与观感一致

因此更准确地说：**整张图片都会变清晰，但文字区域清晰度提升是核心目标**。

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

## 4.固定倍率超分（扩大分辨率）

> 默认**关闭** decoder attention（更省显存，推荐）。只有在显存充足时才加 `--decoder_attn`。

```powershell
python .\inference_diffusion.py `
	-i .\eval_inputs `
	-o .\eval_outputs\diffusion_local_1to3h `
	--model_path .\model\diffusion_local_1to3h_latest.pth `
	--outscale 4 `
	--timesteps 120
```

> 根据需要修改模型名称

### 4.0 指定要用的卡

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,7
```

### 4.1 服务器 2卡 DDP 训练

```bash
torchrun --nproc_per_node=2 train_diffusion.py \
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

### 4.2 服务器 2卡 DDP 续训

```bash
torchrun --nproc_per_node=2 train_diffusion.py \
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


### 4.3 服务器测试

#### 4.3.1 单模型推理

```bash
CUDA_VISIBLE_DEVICES=0 python inference_diffusion.py \
	-i ./eval_inputs \
	-o ./eval_outputs/diffusion_ddp_2x4090 \
	--model_path ./model/diffusion_ddp_2x4090_best.pth \
	--outscale 4 \
	--timesteps 200 \
	--target_min_side 384
```

- 质量优先：`timesteps=200~300`
- 速度优先：`timesteps=80~120`



#### 4.3.2 统一画廊评测（PSNR/SSIM）：

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

#### 4.3.3 OCR 指标评测（PaddleOCR + GPU）：

```bash
python ./tools/evaluate_ocr_metrics.py \
	--pred_dir ./eval_outputs/diffusion_ddp_6x4090 \
	--gt_csv ./eval_inputs/labels.csv \
	--image_col image \
	--text_col text \
	--suffix _diffusion \
	--ocr_backend paddleocr \
	--lang ch \
	--device gpu \
	--output_csv ./eval_outputs/ocr_metrics_detail.csv \
	--output_json ./eval_outputs/ocr_metrics_summary.json
```

### 4.4 服务器统一评测

```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/evaluate_text_models.py \
	--input_dir ./eval_inputs \
	--output_dir ./eval_outputs/cmp_ddp_2x4090 \
	--methods bicubic,diffusion \
	--outscale 4 \
	--diffusion_model_path ./model/diffusion_ddp_2x4090_best.pth \
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



## 5. 同分辨率清晰化

> 目标：保持原图尺寸与色彩风格，只提升清晰度。

> 说明：如果你当前模型是用 `--scale 4` 训练得到，它可以用于 x1 清晰化，但并非目标完全对齐。
> 若你的最终目标是“在原图基础上清晰化且尽量不改色”，建议专门训练一版 `--scale 1` 模型（见下文 5.3）。

### 5.0 指定要用的卡

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,7
```

### 5.1 服务器 x1 清晰化目标专用训练

如果最终目标不是放大，而是“原图基础上清晰化”，建议数据与训练都按 x1 目标对齐：

#### 5.1.1 数据集对齐

```shell
python .\tools\make_triplet_from_hr.py `
	--hr_dir .\dataset\HR `
	--out_root .\dataset_triplet_x1\train `
	--scale 1
```

#### 5.1.2 单卡训练

```shell
python .\train_diffusion.py `
	--cond_mode concat `
	--batch_size 4 `
	--epochs 200 `
	--scale 1 `
	--hr_size 128 `
	--train_size 128 `
	--lr 1e-4 `
	--lambda_seg 0 `
	--num_workers 2 `
	--hr_dir .\dataset_triplet_x1\train\HR `
	--lr_dir .\dataset_triplet_x1\train\LR `
	--save_dir .\model `
	--experiment_name diffusion_x1_enhance `
	--save_best `
	--save_every 2
```

#### 5.1.3 多卡训练

```shell
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 train_diffusion.py \
  --ddp \
  --dist_backend nccl \
  --cond_mode concat \
  --batch_size 8 \
  --epochs 120 \
  --scale 1 \
  --hr_size 128 \
  --train_size 128 \
  --lr 1e-4 \
  --lambda_seg 0 \
  --num_workers 4 \
  --hr_dir ./dataset_triplet_x1/train/HR \
  --lr_dir ./dataset_triplet_x1/train/LR \
  --save_dir ./model \
  --experiment_name diffusion_x1_enhance_ddp \
  --save_best \
  --save_every 2
```

### 5.2 测试

#### 5.2.1 单模型推理测试

```shell
CUDA_VISIBLE_DEVICES=0 python inference_diffusion.py \
  -i ./eval_inputs \
  -o ./eval_outputs/enhance_x1_final \
  --model_path ./model/diffusion_x1_enhance_best.pth \
  --outscale 1 \
  --timesteps 120 \
  --target_min_side 256 \
  --strict_color_lock \
  --luma_strength 1.0 \
  --max_luma_delta 28 \
  --enhance_strength 1.0
```

#### 5.2.2 统一对比评测（画廊 + 指标）

```shell
CUDA_VISIBLE_DEVICES=0 python ./tools/evaluate_text_models.py \
  --input_dir ./eval_inputs \
  --output_dir ./eval_outputs/cmp_x1 \
  --methods bicubic,diffusion \
  --outscale 1 \
  --diffusion_model_path ./model/diffusion_x1_enhance_best.pth \
  --diffusion_outscale 1 \
  --diffusion_steps 120 \
  --diffusion_min_side 256 \
  --diffusion_fallback_min_side 192 \
  --diffusion_strict_color_lock \
  --diffusion_luma_strength 1.0 \
  --diffusion_max_luma_delta 28 \
  --diffusion_enhance_strength 1.0
```

#### 5.2.3 OCR指标评测（Acc/CER/WER）

```shell
conda activate paddleocr
export PPOCR_ROOT=/data/dachuang/TEST/PPOCRv5

python ./tools/evaluate_ocr_metrics.py \
  --pred_dir ./eval_outputs/enhance_x1 \
  --gt_csv ./eval_inputs/labels.csv \
  --image_col image \
  --text_col text \
  --suffix _diffusion \
  --ocr_backend paddleocr \
  --lang ch \
  --device gpu \
  --output_csv ./eval_outputs/ocr_metrics_detail.csv \
  --output_json ./eval_outputs/ocr_metrics_summary.json
```

---

参数说明：

- `-i`：输入图片目录
- `-o`：输出目录
- `--model_path`：训练得到的 checkpoint
- `--outscale`：输出倍率；`1` 表示同分辨率清晰化
- `--timesteps`：采样步数（更高通常更慢但可能更好）
- `--decoder_attn`：启用 decoder 注意力（显存开销极大，默认关闭）
- `--preserve_color`：保留原图色彩（仅增强亮度细节）
- `--strict_color_lock`：严格保色（锁定色彩通道，仅做受控亮度增强）
- `--luma_strength`：严格保色下亮度增强强度（`0~1`）
- `--max_luma_delta`：严格保色下每像素最大亮度变化（越小越接近原图色彩观感）
- `--enhance_strength`：增强强度（`0~1`，推荐 `0.7~1.0`）

---





---

## 6. 常见问题

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


### Q4: 如何从当前实验选最佳

优先级建议：

1. `metrics.csv`（若有 GT）：按 `SSIM`、`PSNR` 选 Top-2 checkpoint
2. 人工目检：重点看文字边缘、断笔、重影、可读性
3. 最终保留：`*_best.pth` + 对应训练命令（可复现）

### Q5：统一对比评测参数

说明：

- `--methods bicubic,diffusion`：只跑双三次 + 扩散，不依赖 `basicsr`
- `--diffusion_min_side`：扩散推理尺度，过大可能 OOM
- `--diffusion_fallback_min_side`：OOM 自动回退尺度


### Q6: 本次大创成员使用服务器的可能问题

1. 由于权限问题以及默认入口问题，启动不了conda
> 输入如下命令临时修改PATH、HOME
```bash
export PATH="/data/dachuang/TEST/miniconda3/bin:$PATH"
export HOME=/data/dachuang/TEST
```

2. conda 显示装不了pytorch或使用不了pytorch等问题
> 输入如下命令使用已配置好的env
```bash
echo 'export HOME=/data/dachuang' >> ~/.bashrc

echo 'export PATH=/data/dachuang/.local/bin:$PATH' >> ~/.bashrc

echo 'export LD_LIBRARY_PATH=/data/dachuang/envs/realesrgan/lib/python3.8/site-packages/nvidia/nvjitlink/lib:/data/dachuang/envs/realesrgan/lib/python3.8/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# 激活 base（可能是无pytorch的）
source ~/.bashrc

# 激活 realesrgan
source /data/dachuang/envs/realesrgan/bin/activate
```

3. 由于疏忽以及权限问题，导致可能出现同时激活两个环境如`(realesrgan)(base)$`
只需要输入 `conda deactivate` 即可关闭 `(base)` 环境。

已配置好的pytorch等工具是在环境 `realesrgan` 中的。

---

## 7. 服务器环境一键检查清单（GPU）

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


## 8. 关于 OCR 指标评测的说明

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

如果你的服务器目录结构是 `/data/dachuang/TEST`，推荐先激活环境并设置模型根目录：

```bash
cd /data/dachuang/TEST/dachuang-dachuang-text-enhancement
conda activate paddleocr
export PPOCR_ROOT=/data/dachuang/TEST/PPOCRv5
python -c "import sys; print(sys.executable)"
python -c "from paddleocr import PaddleOCR; print('paddleocr ok')"
```

> `tools/evaluate_ocr_metrics.py` 在未传 `--ppocr_root` 时会自动尝试：
> 1) `$PPOCR_ROOT`  2) 仓库根目录 `PPOCRv5`  3) 当前目录 `PPOCRv5`  4) `/data/dachuang/TEST/PPOCRv5`。

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

Linux 服务器示例（GPU）：

```bash
cd /data/dachuang/TEST/dachuang-dachuang-text-enhancement
conda activate paddleocr
export PPOCR_ROOT=/data/dachuang/TEST/PPOCRv5
python ./tools/evaluate_ocr_metrics.py \
	--pred_dir ./eval_outputs/diffusion_local_1to3h \
	--gt_csv ./eval_inputs/labels.csv \
	--image_col image \
	--text_col text \
	--suffix _diffusion \
	--ocr_backend paddleocr \
	--lang ch \
	--device gpu \
	--output_csv ./eval_outputs/ocr_metrics_detail.csv \
	--output_json ./eval_outputs/ocr_metrics_summary.json
```

参数说明：

- `--pred_dir`：预测图片目录
- `--gt_csv`：GT 文本标注 CSV
- `--image_col` / `--text_col`：CSV 中图片列和文本列
- `--suffix`：预测文件名后缀（例如 `eval_001_diffusion.png` 的 `_diffusion`）
- `--ocr_backend`：OCR后端（默认 `paddleocr`，兼容 `tesseract`）
- `--ppocr_root`：PP-OCR 模型根目录（可不填，脚本会自动探测：`$PPOCR_ROOT` → 仓库根目录 `PPOCRv5` → 当前目录 `PPOCRv5` → `/data/dachuang/TEST/PPOCRv5`）
- `--det_model_dir` / `--rec_model_dir` / `--cls_model_dir`：显式指定模型目录（可选）
- `--lang`：OCR语言（如 `ch`、`en`，也兼容 `eng`、`chi_sim`）
- `--use_angle_cls`：启用方向分类器（推荐开启）
- `--use_gpu`：使用 GPU 推理（需安装 `paddlepaddle-gpu`）

输出文件：

- `ocr_metrics_detail.csv`：每张图的 `pred_text / gt_text / CER / WER / exact_match`
- `ocr_metrics_summary.json`：整体 `accuracy / CER / WER / 样本数`

