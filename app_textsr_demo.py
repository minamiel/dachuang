import io
import json
import os
import time
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, abort, redirect, render_template_string, request, send_from_directory, url_for

from inference_diffusion import (
    DiffusionSampler,
    build_model,
    build_structure_prior_tensor,
    image_to_tensor,
    restore_image,
    smart_resize,
    upscale_input,
)
from run_all import merged_presets

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover
    RapidOCR = None


ROOT = Path(__file__).resolve().parent
DEMO_ROOT = ROOT / "dashboard" / "demo_runtime"
UPLOAD_ROOT = DEMO_ROOT / "uploads"
RESULT_ROOT = DEMO_ROOT / "results"
DEFAULT_MODEL = "model/text_prior_v2_latest.pth"
DEFAULT_PRESET = "text-balanced"


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>文字超分交互演示平台</title>
  <style>
    :root {
      --bg: #f5efe4;
      --panel: rgba(255,255,255,0.9);
      --line: rgba(26,34,46,0.12);
      --ink: #1a222e;
      --muted: #5d6775;
      --accent: #a3471f;
      --accent-2: #2f6b57;
      --shadow: 0 24px 60px rgba(41, 28, 13, 0.08);
      --radius: 24px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(163,71,31,0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(47,107,87,0.16), transparent 24%),
        linear-gradient(180deg, #fbf7f1, var(--bg));
    }
    .wrap {
      width: min(1260px, calc(100% - 28px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 28px;
      margin-bottom: 22px;
    }
    .eyebrow {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 10px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(32px, 4vw, 58px);
      line-height: 1.03;
      letter-spacing: -0.03em;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      max-width: 860px;
    }
    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }
    .panel {
      padding: 20px;
    }
    .panel h2 {
      margin: 0 0 14px;
      font-size: 22px;
    }
    label {
      display: block;
      font-weight: 600;
      margin: 14px 0 8px;
    }
    .hint {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    input[type="file"], select, button {
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      min-height: 46px;
      padding: 0 14px;
      font-size: 14px;
      background: white;
    }
    button {
      cursor: pointer;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), #c06e43);
      border: none;
      color: white;
      margin-top: 18px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 18px;
    }
    .meta div {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px;
      background: rgba(255,255,255,0.8);
    }
    .meta span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .result-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 22px;
      overflow: hidden;
      background: rgba(255,255,255,0.88);
    }
    .card img {
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #efe7da;
    }
    .card .body {
      padding: 14px;
    }
    .card h3 {
      margin: 0 0 6px;
      font-size: 18px;
    }
    .card p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }
    .text-box {
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px;
      background: rgba(255,255,255,0.88);
    }
    .text-box h3 {
      margin: 0 0 10px;
      font-size: 20px;
    }
    .ocr {
      white-space: pre-wrap;
      line-height: 1.7;
      font-size: 15px;
    }
    .downloads {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .downloads a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 14px;
      border-radius: 999px;
      text-decoration: none;
      color: white;
      background: var(--accent-2);
      font-weight: 600;
    }
    .empty {
      color: var(--muted);
      line-height: 1.8;
      font-size: 15px;
    }
    .nav {
      margin-bottom: 16px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .nav a {
      text-decoration: none;
      color: var(--ink);
      background: #f2dfd3;
      border-radius: 999px;
      min-height: 40px;
      padding: 0 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .result-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">阶段五 · 可交互可视化系统</div>
      <h1>文字超分交互演示平台</h1>
      <p>上传一张 LR 文本图像，系统会自动生成 HR 增强结果、文本分割可视化、OCR 文本输出，并提供导出功能。这个页面是通用原型，后面更换模型或升级评测流程仍然可以继续复用。</p>
    </section>

    <div class="nav">
      <a href="{{ url_for('index') }}">重新上传</a>
      <a href="{{ url_for('dashboard_page') }}" target="_blank">打开实验看板</a>
    </div>

    <div class="layout">
      <section class="panel">
        <h2>上传与推理设置</h2>
        <form method="post" action="{{ url_for('process') }}" enctype="multipart/form-data">
          <label>选择 LR 文本图像</label>
          <input type="file" name="image" accept=".png,.jpg,.jpeg,.bmp,.webp" required />
          <div class="hint">建议上传已经裁好的文字区域图，展示会更稳定。</div>

          <label>选择模型</label>
          <select name="model_path">
            {% for item in models %}
            <option value="{{ item.value }}" {% if item.value == active_model %}selected{% endif %}>{{ item.label }}</option>
            {% endfor %}
          </select>

          <label>选择预设</label>
          <select name="preset">
            {% for item in presets %}
            <option value="{{ item }}" {% if item == active_preset %}selected{% endif %}>{{ item }}</option>
            {% endfor %}
          </select>

          <button type="submit">开始增强并生成展示结果</button>
        </form>

        <div class="meta">
          <div><span>默认主模型</span><strong>{{ active_model_name }}</strong></div>
          <div><span>默认预设</span><strong>{{ active_preset }}</strong></div>
          <div><span>OCR 引擎</span><strong>{{ ocr_backend }}</strong></div>
          <div><span>分割图来源</span><strong>模型 mask 可视化</strong></div>
        </div>
      </section>

      <section class="panel">
        {% if result %}
          <h2>推理结果</h2>
          <div class="result-grid">
            <div class="card">
              <img src="{{ result.input_url }}" alt="输入图" />
              <div class="body">
                <h3>输入 LR 图</h3>
                <p>用户上传的原始文本图像。</p>
              </div>
            </div>
            <div class="card">
              <img src="{{ result.hr_url }}" alt="增强结果" />
              <div class="body">
                <h3>增强 HR 结果</h3>
                <p>基于扩散模型生成的增强图。</p>
              </div>
            </div>
            <div class="card">
              <img src="{{ result.mask_url }}" alt="分割图" />
              <div class="body">
                <h3>文本分割热力图</h3>
                <p>来自当前模型分割头的文字区域概率可视化。</p>
              </div>
            </div>
            <div class="card">
              <img src="{{ result.overlay_url }}" alt="分割叠加图" />
              <div class="body">
                <h3>文本区域叠加展示</h3>
                <p>把分割热力图叠加在增强结果上，方便观察模型关注区域。</p>
              </div>
            </div>
          </div>

          <div class="text-box">
            <h3>OCR 文本输出</h3>
            <div class="ocr">{{ result.ocr_text }}</div>
            <div class="downloads">
              <a href="{{ result.hr_url }}" download>导出增强图</a>
              <a href="{{ result.mask_url }}" download>导出分割图</a>
              <a href="{{ result.overlay_url }}" download>导出叠加图</a>
              <a href="{{ result.json_url }}" download>导出结果 JSON</a>
              <a href="{{ result.zip_url }}" download>打包导出全部结果</a>
            </div>
          </div>
        {% else %}
          <h2>结果展示区</h2>
          <div class="empty">
            这里会展示四类内容：输入 LR 图、增强 HR 图、文本分割图、OCR 文本输出。<br />
            这就是你申请书里“支持 LR 文本图像上传、生成 HR 结果与文本分割图展示及 OCR 文本输出、实现结果导出功能”的页面原型。
          </div>
        {% endif %}
      </section>
    </div>
  </div>
</body>
</html>
"""


def ensure_dirs() -> None:
    for path in [DEMO_ROOT, UPLOAD_ROOT, RESULT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def list_model_choices() -> List[Dict[str, str]]:
    model_dir = ROOT / "model"
    options: List[Dict[str, str]] = []
    if not model_dir.exists():
        return [{"label": DEFAULT_MODEL, "value": DEFAULT_MODEL}]
    for path in sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = path.relative_to(ROOT).as_posix()
        options.append({"label": path.name, "value": rel})
    return options or [{"label": DEFAULT_MODEL, "value": DEFAULT_MODEL}]


def resolve_model_path(model_value: str) -> Path:
    path = Path(model_value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def pick_defaults() -> Tuple[str, str]:
    model_choices = list_model_choices()
    active_model = DEFAULT_MODEL
    for item in model_choices:
        if item["value"].endswith("text_prior_v2_latest.pth"):
            active_model = item["value"]
            break
    else:
        active_model = model_choices[0]["value"]
    return active_model, DEFAULT_PRESET


def get_preset_values(preset_name: str) -> Dict[str, float]:
    presets = merged_presets()
    values = dict(presets.get(preset_name, presets.get(DEFAULT_PRESET, {})))
    return {
        "steps": int(values.get("steps", 180)),
        "min_side": int(values.get("min_side", 352)),
        "fallback_min_side": int(values.get("fallback_min_side", 256)),
        "outscale": float(values.get("outscale", 4.0)),
        "enhance_strength": float(values.get("enhance_strength", 1.0)),
        "luma_strength": float(values.get("luma_strength", 1.0)),
        "max_luma_delta": float(values.get("max_luma_delta", 14.0)),
        "color_lock_strength": float(values.get("color_lock_strength", 0.98)),
        "edge_sharpen_strength": float(values.get("edge_sharpen_strength", 0.35)),
    }


@lru_cache(maxsize=4)
def get_runtime(model_path_str: str, timesteps: int):
    model_path = resolve_model_path(model_path_str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(str(model_path), device, cond_mode="concat", use_decoder_attn=None)
    sampler = DiffusionSampler(timesteps, device)
    return device, model, sampler


@lru_cache(maxsize=1)
def get_ocr_engine():
    if RapidOCR is None:
        return None
    return RapidOCR()


def run_ocr_text(img_bgr: np.ndarray) -> str:
    engine = get_ocr_engine()
    if engine is None:
        return "当前环境未检测到 RapidOCR，OCR 输出不可用。"
    try:
        result, _ = engine(img_bgr)
        lines: List[str] = []
        if result:
            for item in result:
                if len(item) >= 2:
                    text = str(item[1]).strip()
                    if text:
                        lines.append(text)
        return "\n".join(lines) if lines else "OCR 未识别出明显文本。"
    except Exception as exc:  # pragma: no cover
        return f"OCR 执行失败：{exc}"


def build_mask_visualization(model, restored_bgr: np.ndarray, input_bgr: np.ndarray, device: str, target_min_side: int, outscale: float) -> Tuple[np.ndarray, np.ndarray]:
    sr_input = upscale_input(input_bgr, outscale=outscale)
    cond_resized = smart_resize(sr_input, target_min_side=target_min_side)
    restored_resized = cv2.resize(restored_bgr, (cond_resized.shape[1], cond_resized.shape[0]), interpolation=cv2.INTER_CUBIC)

    cond_tensor = image_to_tensor(cond_resized, device)
    restored_tensor = image_to_tensor(restored_resized, device)
    timestep = torch.zeros((1,), device=device, dtype=torch.long)

    structure_prior = None
    if getattr(model, "use_structure_prior", False) or getattr(model, "use_decoder_structure_gate", False):
        structure_prior = build_structure_prior_tensor(
            cond_tensor,
            strength=float(getattr(model, "structure_prior_strength", 1.0)),
        )

    with torch.no_grad():
        if getattr(model, "cond_mode", "concat") == "film":
            model_output = model(restored_tensor, timestep, cond=cond_tensor, structure_prior=structure_prior)
        else:
            model_input = torch.cat((restored_tensor, cond_tensor), dim=1)
            model_output = model(model_input, timestep, structure_prior=structure_prior)

    mask_pred = model_output.get("mask_pred")
    if mask_pred is None:
        blank = np.zeros(restored_bgr.shape[:2], dtype=np.uint8)
        mask_color = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
        return mask_color, restored_bgr.copy()

    mask_prob = torch.sigmoid(mask_pred)[0, 0].detach().cpu().numpy()
    mask_prob = cv2.resize(mask_prob, (restored_bgr.shape[1], restored_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask_uint8 = np.clip(mask_prob * 255.0, 0, 255).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(restored_bgr, 0.72, mask_color, 0.28, 0.0)
    return mask_color, overlay


def save_image(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr)


def make_zip(zip_path: Path, files: List[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                zf.write(path, arcname=path.name)


def process_one_image(image_path: Path, model_value: str, preset_name: str) -> Dict[str, str]:
    preset = get_preset_values(preset_name)
    device, model, sampler = get_runtime(model_value, preset["steps"])

    input_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if input_bgr is None:
        raise ValueError(f"无法读取图像：{image_path}")

    result_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    result_dir = RESULT_ROOT / result_id
    result_dir.mkdir(parents=True, exist_ok=True)

    hr_bgr = restore_image(
        model=model,
        sampler=sampler,
        img_bgr=input_bgr,
        device=device,
        target_min_side=preset["min_side"],
        timesteps=preset["steps"],
        outscale=preset["outscale"],
        preserve_color=False,
        enhance_strength=preset["enhance_strength"],
        strict_color_lock=True,
        luma_strength=preset["luma_strength"],
        max_luma_delta=preset["max_luma_delta"],
        match_luma_stats=True,
        color_lock_strength=preset["color_lock_strength"],
        edge_sharpen_strength=preset["edge_sharpen_strength"],
        tile_size=0,
        tile_overlap=32,
        tile_blend=True,
        init_mode="condition_noise",
        noise_strength=0.1,
    )
    mask_bgr, overlay_bgr = build_mask_visualization(
        model=model,
        restored_bgr=hr_bgr,
        input_bgr=input_bgr,
        device=device,
        target_min_side=preset["min_side"],
        outscale=preset["outscale"],
    )
    ocr_text = run_ocr_text(hr_bgr)

    input_path = result_dir / "input_lr.png"
    hr_path = result_dir / "output_hr.png"
    mask_path = result_dir / "mask_map.png"
    overlay_path = result_dir / "mask_overlay.png"
    json_path = result_dir / "result.json"
    zip_path = result_dir / "export_bundle.zip"

    save_image(input_path, input_bgr)
    save_image(hr_path, hr_bgr)
    save_image(mask_path, mask_bgr)
    save_image(overlay_path, overlay_bgr)

    payload = {
        "result_id": result_id,
        "model_path": model_value,
        "preset": preset_name,
        "ocr_text": ocr_text,
        "files": {
            "input_lr": input_path.name,
            "output_hr": hr_path.name,
            "mask_map": mask_path.name,
            "mask_overlay": overlay_path.name,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    make_zip(zip_path, [input_path, hr_path, mask_path, overlay_path, json_path])

    return {
        "input_url": url_for("serve_result_file", result_id=result_id, filename=input_path.name),
        "hr_url": url_for("serve_result_file", result_id=result_id, filename=hr_path.name),
        "mask_url": url_for("serve_result_file", result_id=result_id, filename=mask_path.name),
        "overlay_url": url_for("serve_result_file", result_id=result_id, filename=overlay_path.name),
        "json_url": url_for("serve_result_file", result_id=result_id, filename=json_path.name),
        "zip_url": url_for("serve_result_file", result_id=result_id, filename=zip_path.name),
        "ocr_text": ocr_text,
    }


def render_page(result: Dict[str, str] = None):
    models = list_model_choices()
    active_model, active_preset = pick_defaults()
    active_model_name = Path(active_model).name
    presets = sorted(merged_presets().keys())
    return render_template_string(
        PAGE_TEMPLATE,
        models=models,
        active_model=active_model,
        active_model_name=active_model_name,
        active_preset=active_preset,
        presets=presets,
        ocr_backend="rapidocr" if RapidOCR is not None else "不可用",
        result=result,
    )


@app.route("/", methods=["GET"])
def index():
    ensure_dirs()
    return render_page()


@app.route("/process", methods=["POST"])
def process():
    ensure_dirs()
    if "image" not in request.files:
        abort(400, "未检测到上传图片。")
    file = request.files["image"]
    if file.filename == "":
        abort(400, "未选择文件。")

    model_value = request.form.get("model_path") or pick_defaults()[0]
    preset_name = request.form.get("preset") or DEFAULT_PRESET

    suffix = Path(file.filename).suffix.lower() or ".png"
    upload_name = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + suffix
    upload_path = UPLOAD_ROOT / upload_name
    file.save(str(upload_path))

    result = process_one_image(upload_path, model_value, preset_name)
    return render_page(result=result)


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return redirect(url_for("serve_dashboard_file", filename="experiment_dashboard.html"))


@app.route("/dashboard/<path:filename>", methods=["GET"])
def serve_dashboard_file(filename: str):
    dashboard_dir = ROOT / "dashboard"
    return send_from_directory(dashboard_dir, filename)


@app.route("/demo_runtime/results/<result_id>/<path:filename>", methods=["GET"])
def serve_result_file(result_id: str, filename: str):
    result_dir = RESULT_ROOT / result_id
    if not result_dir.exists():
        abort(404)
    return send_from_directory(result_dir, filename)


if __name__ == "__main__":
    ensure_dirs()
    print("TextSR demo is starting at: http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
