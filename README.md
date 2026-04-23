# 基于扩散模型的文本图像超分辨率

本项目是一个面向文字图像增强的扩散超分原型系统。当前主线已经具备完整闭环：

- 可训练的图像条件扩散模型
- 面向文字结构的增强损失与先验
- 真实手机拍照文字图像的 OCR 评测流程
- 简单可交互的演示页面与实验看板

当前最稳定的主模型是 `text_prior_v2`。在我们现有的 24 张真实手机文字图像评测集上，它相对 baseline 取得了更好的 OCR 指标。

## 当前状态

当前仓库已经完成：

- 图像空间条件扩散文字超分主干
- `baseline / text_prior_v2 / text_prior_v3` 多版本实验
- OCR 评测脚本与真实集评测闭环
- 实验 dashboard 与本地演示页面

当前仓库尚未完整完成申请书中的重型路线：

- VAE / latent diffusion
- 文本扩散 Transformer 分支
- 完整的 SAM 微调闭环
- 大规模高质量 `HR-mask-LR` 三联数据主训练线

更完整的现状说明见：

- [CURRENT_STATUS_CONFIRMATION_CN.md](./CURRENT_STATUS_CONFIRMATION_CN.md)
- [MODEL_IMPLEMENTATION_ROUTE_CN.md](./MODEL_IMPLEMENTATION_ROUTE_CN.md)
- [SAM_ASSISTED_MASK_EXPERIMENT_CN.md](./SAM_ASSISTED_MASK_EXPERIMENT_CN.md)

## 项目结构

核心训练与推理：

- `train_diffusion.py`：主训练脚本
- `inference_diffusion.py`：扩散推理脚本
- `model_unet.py`：当前主干模型 `SimpleUNet`
- `dataloader.py`：数据加载
- `run_all.py`：统一入口，封装增强、评测、OCR、报告

评测与工具：

- `tools/evaluate_text_models.py`：图像质量与对比评测
- `tools/evaluate_ocr_metrics.py`：OCR 指标评测
- `tools/init_eval_labels_csv.py`：初始化真实评测集标签
- `tools/build_experiment_dashboard.py`：生成实验看板
- `tools/prepare_samassist_smallset.py`：准备小规模 SAM-assisted 数据子集

演示与可视化：

- `app_textsr_demo.py`：本地交互式演示页面
- `dashboard/experiment_dashboard.html`：实验看板静态页面

文档：

- `CURRENT_STATUS_CONFIRMATION_CN.md`
- `MODEL_IMPLEMENTATION_ROUTE_CN.md`
- `NEXT_AI_MODEL_TASK_CN.md`
- `PROJECT_ASSESSMENT_CN.md`
- `RESEARCH_UPGRADE_PLAN_CN.md`

## 环境安装

建议使用 Conda。

```bash
conda create -n dachuang python=3.9
conda activate dachuang
pip install -r requirements.txt
```

如果你需要 GPU 版 PyTorch，请按你的 CUDA 版本自行安装匹配版本。

## 数据准备

### 1. 训练数据

当前训练主线使用成对的 `HR/LR` 数据，可选 `mask`：

```text
dataset_triplet/
  train/
    HR/
    LR/
    masks/   # 可选
```

如果你手头有 HR 图像，可使用工具生成训练三联数据：

```bash
python tools/make_triplet_from_hr.py \
  --hr_dir dataset/HR \
  --out_root dataset_triplet/train \
  --scale 4
```

### 2. 真实评测集

真实手机拍照评测集建议单独维护，例如：

```text
eval_inputs_v2/
  eval_001.png
  eval_002.png
  ...
  labels.csv
```

其中 `labels.csv` 需要填写每张图的真实文字内容。

## 训练

一个典型的训练命令如下：

```bash
python train_diffusion.py \
  --cond_mode concat \
  --batch_size 8 \
  --epochs 30 \
  --hr_size 128 \
  --train_size 128 \
  --lr 1e-4 \
  --scale 4 \
  --lambda_seg 0 \
  --lambda_focus 0 \
  --lambda_structure 0.03 \
  --lambda_recog 0.02 \
  --use_structure_prior \
  --structure_prior_strength 1.0 \
  --num_workers 4 \
  --hr_dir dataset_triplet/train/HR \
  --lr_dir dataset_triplet/train/LR \
  --save_dir model \
  --experiment_name text_prior_v2 \
  --save_best
```

常用能力包括：

- `--lambda_seg`：分割辅助损失
- `--lambda_focus`：文字区域关注损失
- `--lambda_structure`：结构一致性损失
- `--lambda_recog`：recognizer proxy loss
- `--use_structure_prior`：启用结构先验
- `--use_decoder_structure_gate`：启用 decoder 结构门控

## 推理

### 单图或目录增强

```bash
python run_all.py enhance \
  -i eval_inputs \
  -o eval_outputs/enhanced \
  --preset text-balanced \
  --model_path model/text_prior_v2_best.pth
```

### 真实 OCR 评测

```bash
python run_all.py real-ocr-eval \
  --input_dir eval_inputs_v2 \
  --output_dir eval_outputs/real_ocr_eval_v2_text_prior_v2 \
  --gt_csv eval_inputs_v2/labels.csv \
  --ocr_backend rapidocr \
  --preset text-balanced \
  --model_path model/text_prior_v2_best.pth
```

## 当前有效结果

在当前 24 张真实手机图评测集上：

- `baseline`
  - `accuracy = 0.2917`
  - `cer = 0.4951`
  - `wer = 1.3205`
- `text_prior_v2`
  - `accuracy = 0.6250`
  - `cer = 0.0932`
  - `wer = 0.4615`
- `text_prior_v3`
  - `accuracy = 0.6250`
  - `cer = 0.0964`
  - `wer = 0.4824`

所以当前建议：

- 主模型：`text_prior_v2`
- 对照模型：`baseline`
- 复杂结构分支：`text_prior_v3`

## 演示页面

启动本地演示系统：

```bash
python app_textsr_demo.py
```

然后浏览器打开：

```text
http://127.0.0.1:7860
```

当前演示支持：

- 上传 LR 文字图像
- 选择模型 checkpoint
- 生成增强 HR 结果
- 展示文本分割热力图与叠加图
- 输出 OCR 文本
- 导出结果图与 JSON

## 实验看板

重新生成实验看板：

```bash
python tools/build_experiment_dashboard.py --project_root .
```

输出文件：

- `dashboard/experiment_dashboard.html`

## GitHub 上传建议

建议上传：

- 源代码
- 文档
- 工具脚本
- `requirements.txt`
- `README.md`
- 示例性的少量轻量资源

不要上传：

- 大模型权重
- 大规模训练数据
- 大量评测输出
- 临时缓存
- 演示运行时导出目录

本仓库的 `.gitignore` 已按这个方向收过一版。

## 下一步

当前最合理的主线不是继续频繁改结构，而是：

1. 扩充更高质量训练数据
2. 清理和固定真实评测集
3. 继续围绕 `text_prior_v2` 做训练与验证
4. 小范围尝试 `SAM-assisted` mask 路线

---

如果你是第一次接手这个项目，建议先看：

1. [CURRENT_STATUS_CONFIRMATION_CN.md](./CURRENT_STATUS_CONFIRMATION_CN.md)
2. [MODEL_IMPLEMENTATION_ROUTE_CN.md](./MODEL_IMPLEMENTATION_ROUTE_CN.md)
3. [SAM_ASSISTED_MASK_EXPERIMENT_CN.md](./SAM_ASSISTED_MASK_EXPERIMENT_CN.md)
