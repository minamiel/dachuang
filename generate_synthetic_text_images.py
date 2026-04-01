#!/usr/bin/env python3
"""
生成合成带文本的 HR 图像及对应分割 mask
　　
用法示例:
python generate_synthetic_text_images.py \
    --backgrounds backgrounds/ \
    --texts texts.txt \
    --out_hr data/synthetic/hr_images \
    --out_mask data/synthetic/masks \
    --num 200

输出：
- data/synthetic/hr_images/    合成图像（保持背景原始分辨率）
- data/synthetic/masks/        对应分割 mask（文本区域=255，其他=0）

依赖： pillow, numpy, opencv-python

脚本要点（中文注释在代码中有更详细说明）：
- 随机选择背景、随机挑选文本行
- 随机字体/大小/颜色/位置，并进行换行以避免超出
- 在同一位置绘制二值 mask
"""

import os
import sys
import random
import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def list_files(folder, exts=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
    p = Path(folder)
    if not p.exists():
        return []
    return [str(x) for x in p.iterdir() if x.suffix.lower() in exts]


def find_system_fonts(fonts_dir=None):
    # 尝试使用用户指定字体目录，否则在 Windows 上使用 C:\Windows\Fonts
    candidates = []
    if fonts_dir:
        candidates = list_files(fonts_dir, exts=('.ttf', '.otf'))
    else:
        # 常见系统字体目录（Windows / Linux / macOS）
        sys_fonts = []
        if os.name == 'nt':
            sys_fonts = list_files('C:/Windows/Fonts', exts=('.ttf', '.otf'))
        else:
            # Linux / macOS 常见目录
            sys_fonts = list_files('/usr/share/fonts', exts=('.ttf', '.otf')) + list_files('/usr/local/share/fonts', exts=('.ttf', '.otf'))
        candidates = sys_fonts

    # 返回至少一个字体路径或空列表
    return candidates


def wrap_text_to_width(draw, text, font, max_width):
    # 使用 textwrap 来初步拆分，然后用 draw.textbbox 精确测量
    words = text.split()
    if not words:
        return ['']

    lines = []
    cur = words[0]
    for w in words[1:]:
        test = cur + ' ' + w
        bbox = draw.textbbox((0, 0), test, font=font)
        w_px = bbox[2] - bbox[0]
        if w_px <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def choose_text_color(bg_crop):
    # 简单策略：根据背景亮度选择黑或白，偶尔选择随机深色/亮色
    # bg_crop: numpy array HxWx3 (RGB)
    if bg_crop is None:
        return (255, 255, 255)
    mean = bg_crop.mean()
    # 如果背景很亮就选深色文本，反之选白色
    if mean > 180:
        return (0, 0, 0)
    if mean < 70:
        return (255, 255, 255)
    # 否则随机选择白/黑或一个颜色
    if random.random() < 0.6:
        return (255, 255, 255) if random.random() < 0.5 else (0, 0, 0)
    # 随机颜色（但避免太浅）
    return tuple(int(x) for x in np.random.randint(30, 230, size=3))


def generate_one(background_path, text, fonts, out_hr_path, out_mask_path, idx):
    # 打开背景
    bg = Image.open(background_path).convert('RGB')
    W, H = bg.size

    draw = ImageDraw.Draw(bg)

    # 选随机字体
    font_path = random.choice(fonts) if fonts else None

    # 根据图像大小决定字体尺寸范围
    min_dim = min(W, H)
    min_font = max(12, int(min_dim * 0.03))
    max_font = max(min_font + 1, int(min_dim * 0.12))
    font_size = random.randint(min_font, max_font)

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 最大文本宽度为图片宽度的 90%
    max_text_w = int(W * 0.9)

    # 尝试换行以适应宽度
    lines = wrap_text_to_width(draw, text, font, max_text_w)

    # 如果行数过多，逐步减小字体，最多尝试若干次
    attempts = 0
    while (len(lines) > H // (font_size // 2 + 1)) and attempts < 6:
        font_size = max(min_font, int(font_size * 0.85))
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        lines = wrap_text_to_width(draw, text, font, max_text_w)
        attempts += 1

    # 计算文本总高度
    line_heights = []
    line_widths = []
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        w_px = bbox[2] - bbox[0]
        h_px = bbox[3] - bbox[1]
        line_widths.append(w_px)
        line_heights.append(h_px)

    text_w = max(line_widths) if line_widths else 0
    text_h = sum(line_heights) + (len(lines) - 1) * int(font_size * 0.15)

    # 随机位置，确保文本不超出边界
    max_x = max(0, W - text_w - 1)
    max_y = max(0, H - text_h - 1)
    x = random.randint(0, max_x) if max_x > 0 else 0
    y = random.randint(0, max_y) if max_y > 0 else 0

    # 为了决定颜色，从背景中裁剪出相同区域并计算亮度
    try:
        bg_crop = np.array(bg.crop((x, y, x + text_w, y + text_h))).astype(np.uint8)
    except Exception:
        bg_crop = None

    color = choose_text_color(bg_crop)

    # 可选：添加描边以增强可读性
    stroke_width = max(1, int(font_size * 0.04))
    stroke_fill = (0, 0, 0) if sum(color) > 382 else (255, 255, 255)  # 若文字较亮用黑描边，反之用白描边

    # 在背景图上绘制文本（多行）
    cur_y = y
    for ln, lh in zip(lines, line_heights):
        # 居中每行到 text_w
        ln_w = draw.textbbox((0, 0), ln, font=font)[2]
        # 如果行宽小于 text_w，可以选择 left alignment; 这里我们 left-aligned
        draw.text((x, cur_y), ln, font=font, fill=color, stroke_width=stroke_width, stroke_fill=stroke_fill)
        cur_y += lh + int(font_size * 0.15)

    # 保存合成图像
    hr_fname = f"synth_{idx:06d}.png"
    out_hr_file = os.path.join(out_hr_path, hr_fname)
    bg.save(out_hr_file)

    # 创建 mask：空白黑图，文字区域为白色(255)
    mask = Image.new('L', (W, H), 0)
    md = ImageDraw.Draw(mask)
    cur_y = y
    for ln, lh in zip(lines, line_heights):
        md.text((x, cur_y), ln, font=font, fill=255)
        cur_y += lh + int(font_size * 0.15)

    mask_fname = f"synth_{idx:06d}.png"
    out_mask_file = os.path.join(out_mask_path, mask_fname)
    mask.save(out_mask_file)

    return out_hr_file, out_mask_file


def main(args):
    backgrounds = list_files(args.backgrounds, exts=('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    if not backgrounds:
        print('背景目录为空或不存在:', args.backgrounds)
        return

    # 读取文本文件
    with open(args.texts, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        print('文本文件为空或不存在:', args.texts)
        return

    # 字体列表
    fonts = find_system_fonts(args.fonts_dir)
    if not fonts:
        print('未找到系统字体，请通过 --fonts_dir 指定 ttf/otf 字体目录；将使用 PIL 默认字体（可能不支持中文）。')

    # 创建输出目录
    os.makedirs(args.out_hr, exist_ok=True)
    os.makedirs(args.out_mask, exist_ok=True)

    # 生成
    total = args.num
    for i in range(total):
        bg = random.choice(backgrounds)
        text = random.choice(lines)
        try:
            hr_file, mask_file = generate_one(bg, text, fonts, args.out_hr, args.out_mask, i)
            if (i + 1) % 10 == 0 or i < 5:
                print(f"[{i+1}/{total}] -> {hr_file} , {mask_file}")
        except Exception as e:
            print(f"生成第 {i} 张失败 (背景={bg}, text='{text}'): {e}")

    print('完成。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成合成文本图像及分割 mask')
    parser.add_argument('--backgrounds', type=str, required=True, help='背景图片文件夹')
    parser.add_argument('--texts', type=str, required=True, help='包含多行文本的 txt 文件，每行一句')
    parser.add_argument('--out_hr', type=str, default='data/synthetic/hr_images', help='输出 HR 合成图目录')
    parser.add_argument('--out_mask', type=str, default='data/synthetic/masks', help='输出 mask 目录')
    parser.add_argument('--fonts_dir', type=str, default=None, help='可选：字体目录，若不指定使用系统字体目录')
    parser.add_argument('--num', type=int, default=100, help='要生成的图像数量')

    args = parser.parse_args()
    main(args)

