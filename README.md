# 基于扩散模型的文本图像超分辨率

> 超越 Final2x 的扩散模型文本图像增强方案

本项目是一个基于扩散模型的文本图像超分辨率（Text Image Super-Resolution）系统，专门针对文本图像进行优化，在整体增强的前提下**优先保证文字区域的边缘、笔画与可读性**。与传统的插值方法（如双三次插值）和现有工具（如 Final2x）相比，本方案利用扩散模型生成更清晰、更自然的文本细节，显著提升 OCR 准确率和视觉质量。

## ✨ 主要特性

- **扩散模型驱动**: 使用条件扩散模型进行图像超分，生成高质量细节
- **文本优先优化**: 在整体增强的基础上，特别优化文字区域的可读性
- **灵活的推理预设**: 提供 `fast`、`balanced`、`best` 以及专为文本设计的 `text-*` 预设
- **完整的评估体系**: 支持 PSNR、SSIM、LPIPS 等图像质量指标，以及 OCR CER/WER 等文本可读性指标
- **易用的统一入口**: `run_all.py` 脚本提供训练、推理、评估、报告生成等一站式功能
- **多 GPU 训练支持**: 支持 DDP 分布式训练，充分利用服务器硬件资源
- **可视化报告**: 自动生成 HTML 报告，包含对比画廊和详细指标分析

## 🆚 与 Final2x 对比

| 特性 | 本项目 | Final2x |
|------|--------|---------|
| 核心算法 | 扩散模型（生成式） | 传统超分模型（如 Real-ESRGAN） |
| 文本优化 | 专门针对文字区域优化 | 通用图像增强 |
| 细节生成 | 生成更自然的细节和纹理 | 可能产生伪影或过度平滑 |
| 可读性提升 | 显著改善文字边缘和笔画连续性 | 有限改进 |
| 评估指标 | 同时关注图像质量和 OCR 准确率 | 主要关注视觉质量 |
| 灵活性 | 多档预设，可平衡速度与质量 | 固定模型和参数 |

## 🚀 快速开始

### 安装依赖

```bash
# 创建并激活 Conda 环境（推荐）
conda create -n text-enhance python=3.9
conda activate text-enhance

# 安装 PyTorch（根据你的 CUDA 版本）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装项目依赖
pip install -r requirements.txt
```

### 基本使用（使用预训练模型）

1. **单图像增强**：
```bash
python run_all.py enhance \
  -i eval_inputs \
  -o eval_outputs/enhanced \
  --preset text-balanced
```

2. **批量对比评估**：
```bash
python run_all.py compare \
  --input_dir eval_inputs \
  --output_dir eval_outputs/comparison \
  --preset text-balanced
```

3. **生成评估报告**：
```bash
python run_all.py report \
  --output_dir eval_outputs/comparison \
  --summary_json comparison_summary.json \
  --report_html report.html
```

## 📦 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.7+（支持 CUDA）
- GPU 内存 ≥ 8GB（训练），≥ 4GB（推理）

### 完整安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/dachuang-dachuang-text-enhancement.git
cd dachuang-dachuang-text-enhancement

# 2. 创建 Conda 环境
conda create -n text-enhance python=3.9
conda activate text-enhance

# 3. 安装 PyTorch（示例为 CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 安装 PaddleOCR（用于 OCR 评估，可选）
pip install paddleocr
# 注意：如需 GPU 支持，需额外安装 paddlepaddle-gpu
```

## 📊 数据准备

### TextZoom 数据集

本项目使用 TextZoom 数据集进行训练。如果你有 TextZoom 数据集的 LMDB 文件，可以按以下步骤提取：

```bash
# 提取 train1 的 HR 图像
python tools/extract_lmdb_images_generic.py \
  --lmdb_dir TextZoom/train1 \
  --out_dir dataset/HR \
  --prefix train1 \
  --only_hr

# 提取 train2 的 HR 图像
python tools/extract_lmdb_images_generic.py \
  --lmdb_dir TextZoom/train2 \
  --out_dir dataset/HR \
  --prefix train2 \
  --only_hr
```

### 生成训练三联数据（HR/LR/masks）

```bash
# 生成 x4 超分训练数据
python tools/make_triplet_from_hr.py \
  --hr_dir dataset/HR \
  --out_root dataset_triplet/train \
  --scale 4

# 如果有关联的 mask 数据
python tools/make_triplet_from_hr.py \
  --hr_dir dataset/HR \
  --mask_dir dataset/masks \
  --out_root dataset_triplet/train \
  --scale 4
```

输出目录结构：
```
dataset_triplet/
  train/
    HR/      # 高分辨率图像
    LR/      # 低分辨率图像（Real-ESRGAN 风格退化）
    masks/   # 文字区域掩码（可选）
```

## 🏋️ 训练指南

### 单 GPU 训练

```bash
python train_diffusion.py \
  --cond_mode concat \
  --batch_size 4 \
  --epochs 200 \
  --scale 4 \
  --hr_size 256 \
  --train_size 256 \
  --lr 1e-4 \
  --lambda_seg 0.2 \
  --num_workers 4 \
  --hr_dir dataset_triplet/train/HR \
  --lr_dir dataset_triplet/train/LR \
  --mask_dir dataset_triplet/train/masks \
  --save_dir model \
  --experiment_name diffusion_textzoom_bs8 \
  --save_best \
  --save_every 5
```

### 多 GPU 分布式训练（DDP）

```bash
# 2 卡训练示例
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
  --hr_dir dataset_triplet/train/HR \
  --lr_dir dataset_triplet/train/LR \
  --mask_dir dataset_triplet/train/masks \
  --save_dir model \
  --experiment_name diffusion_ddp_2gpu \
  --save_best \
  --save_every 5 \
  --archive_every 20
```

### 训练参数说明

| 参数 | 说明 |
|------|------|
| `--scale` | 超分倍率（4 表示 4 倍超分，1 表示同分辨率清晰化） |
| `--hr_size` | 训练时 HR 图像的裁剪尺寸 |
| `--batch_size` | 批大小（根据 GPU 内存调整） |
| `--lr` | 学习率 |
| `--lambda_seg` | 分割辅助损失的权重（0 表示禁用） |
| `--save_dir` | 模型保存目录 |
| `--experiment_name` | 实验名称，用于组织保存的模型文件 |

## 🔍 推理指南

### 使用训练好的模型进行推理

```bash
# 4 倍超分推理
python inference_diffusion.py \
  -i eval_inputs \
  -o eval_outputs/diffusion_results \
  --model_path model/diffusion_train_best.pth \
  --outscale 4 \
  --timesteps 180 \
  --target_min_side 352
```

### 使用统一入口脚本

```bash
# 使用预设（推荐）
python run_all.py enhance \
  -i eval_inputs \
  -o eval_outputs/enhanced \
  --preset text-balanced

# 自定义参数
python run_all.py enhance \
  -i eval_inputs \
  -o eval_outputs/enhanced \
  --steps 180 \
  --min_side 352 \
  --max_luma_delta 14.0 \
  --edge_sharpen_strength 0.35
```

### 预设说明

| 预设 | 适用场景 | 特点 |
|------|----------|------|
| `fast` | 速度优先 | 低时延，适合实时应用 |
| `balanced` | 平衡模式 | 速度与质量的平衡（推荐） |
| `best` | 质量优先 | 最高质量，速度较慢 |
| `text-fast` | 文本快速增强 | 轻锐化 + 强保色，适合批量处理 |
| `text-balanced` | 文本平衡模式 | 默认文本增强参数（推荐） |
| `text-best` | 文本最佳质量 | 强锐化，极致文本细节 |

## 📈 评估指南

### 图像质量评估（PSNR/SSIM）

```bash
python tools/evaluate_text_models.py \
  --input_dir eval_inputs \
  --output_dir eval_outputs/comparison \
  --methods bicubic,diffusion \
  --outscale 4 \
  --diffusion_model_path model/diffusion_train_best.pth \
  --diffusion_outscale 4 \
  --diffusion_steps 180 \
  --diffusion_min_side 352
```

### OCR 准确率评估

```bash
python tools/evaluate_ocr_metrics.py \
  --pred_dir eval_outputs/diffusion_results \
  --gt_csv eval_inputs/labels.csv \
  --image_col image \
  --text_col text \
  --suffix _diffusion \
  --ocr_backend paddleocr \
  --lang ch \
  --device gpu \
  --output_csv ocr_metrics_detail.csv \
  --output_json ocr_metrics_summary.json
```

### 完整评估流程（一键式）

```bash
python run_all.py full-eval \
  --input_dir eval_inputs \
  --output_dir eval_outputs/full_evaluation \
  --gt_dir eval_gt \
  --gt_csv eval_inputs/labels.csv \
  --methods bicubic,diffusion \
  --lpips \
  --report_html full_eval_report.html
```

## 🏗️ 高级功能

### 模型注册表

```bash
# 列出可用模型
python run_all.py model-registry --action list

# 验证模型完整性
python run_all.py model-registry --action verify --model text-priority

# 下载模型
python run_all.py model-registry --action download --model text-priority
```

### 预设管理

```bash
# 列出所有预设
python run_all.py preset --action list

# 创建自定义预设
python run_all.py preset --action set \
  --name my-text \
  --values_json '{"steps":180,"min_side":352,"edge_sharpen_strength":0.4}'

# 删除预设
python run_all.py preset --action delete --name my-text
```

### 任务队列

创建 `tasks.json`：
```json
{
  "tasks": [
    {
      "name": "compare-fast",
      "argv": ["compare", "--input_dir", "eval_inputs", "--output_dir", "eval_outputs/cmp_fast", "--preset", "text-fast"]
    },
    {
      "name": "report-fast",
      "argv": ["report", "--output_dir", "eval_outputs/cmp_fast", "--summary_json", "compare_summary.json"]
    }
  ]
}
```

执行任务队列：
```bash
python run_all.py queue \
  --queue_json tasks.json \
  --history_json queue_history.json \
  --stop_on_error
```

### GUI 界面

```bash
python run_all.py gui
```

启动桌面 GUI，提供可视化操作界面。

## 🖥️ 服务器训练指南

### 环境检查

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 PyTorch CUDA 支持
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())"

# 检查 PaddlePaddle GPU 支持
python -c "import paddle; print('Paddle compiled with CUDA:', paddle.is_compiled_with_cuda())"
```

### 多卡训练示例（4 张 GPU）

```bash
# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 启动 DDP 训练
torchrun --nproc_per_node=4 train_diffusion.py \
  --ddp \
  --dist_backend nccl \
  --cond_mode concat \
  --batch_size 32 \
  --epochs 200 \
  --scale 4 \
  --hr_size 256 \
  --train_size 256 \
  --lr 1e-4 \
  --lambda_seg 0.2 \
  --num_workers 16 \
  --hr_dir dataset_triplet/train/HR \
  --lr_dir dataset_triplet/train/LR \
  --mask_dir dataset_triplet/train/masks \
  --save_dir model \
  --experiment_name diffusion_ddp_4gpu \
  --save_best \
  --save_every 10 \
  --archive_every 50
```

### 服务器推理与评估

```bash
# 使用特定 GPU 推理
CUDA_VISIBLE_DEVICES=0 python inference_diffusion.py \
  -i eval_inputs \
  -o eval_outputs/server_results \
  --model_path model/diffusion_ddp_4gpu_best.pth \
  --outscale 4 \
  --timesteps 200 \
  --target_min_side 384

# 服务器端完整评估
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_text_models.py \
  --input_dir eval_inputs \
  --output_dir eval_outputs/server_eval \
  --methods bicubic,diffusion \
  --outscale 4 \
  --diffusion_model_path model/diffusion_ddp_4gpu_best.pth \
  --diffusion_outscale 4 \
  --diffusion_steps 200 \
  --diffusion_min_side 384 \
  --gt_dir eval_gt \
  --metrics_csv server_metrics.csv
```

## ❓ 常见问题

### Q1: CUDA 内存不足（OOM）

**解决方法**：
1. 减小批大小：`--batch_size 4 -> 2`
2. 减小训练尺寸：`--train_size 256 -> 128`
3. 减小推理尺寸：`--target_min_side 384 -> 256`
4. 减少采样步数：`--timesteps 200 -> 120`

### Q2: 训练效果不明显

**建议**：
1. 增加训练轮数：`--epochs 200 -> 400`
2. 增加采样步数：`--timesteps 120 -> 200`
3. 检查数据质量，确保 HR/LR 对应正确
4. 尝试调整学习率：`--lr 1e-4 -> 5e-5`

### Q3: 如何选择最佳模型

**优先级**：
1. 查看评估指标（如有 GT）：选择 PSNR/SSIM 最高的 checkpoint
2. 人工视觉检查：重点关注文字边缘、断笔、可读性
3. 保留 `*_best.pth` 和对应的训练命令以便复现

### Q4: OCR 评估失败

**检查步骤**：
1. 确认安装了 PaddleOCR：`pip show paddleocr`
2. 确认 PaddlePaddle GPU 版本与 CUDA 匹配
3. 检查 PP-OCRv5 模型文件是否存在
4. 确认 CSV 文件格式正确，包含 `image` 和 `text` 列

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进本项目。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交 Pull Request

## 🙏 致谢

- 感谢 TextZoom 数据集提供方
- 感谢 Real-ESRGAN 项目提供的退化方法
- 感谢 PaddleOCR 团队提供的 OCR 工具

---

**如有问题，请查看 [Issues](https://github.com/your-username/dachuang-dachuang-text-enhancement/issues) 或提交新 Issue。**