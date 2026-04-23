import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def format_float(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:.2f}"
    return str(value)


def detect_report_path(exp_dir: Path) -> Optional[Path]:
    preferred = [
        "real_ocr_report.html",
        "full_eval_report.html",
        "report_curve_table_preview.html",
        "report_p2_preview.html",
    ]
    for name in preferred:
        path = exp_dir / name
        if path.exists():
            return path
    html_files = sorted(exp_dir.glob("*.html"))
    return html_files[0] if html_files else None


def detect_preview_path(exp_dir: Path) -> Optional[Path]:
    for sub in ["comparisons", "diffusion", "bicubic"]:
        subdir = exp_dir / sub
        if subdir.exists():
            images = sorted(
                [p for p in subdir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
            )
            if images:
                return images[0]
    return None


def collect_experiments(eval_root: Path, project_root: Path) -> List[Dict]:
    experiments: List[Dict] = []
    if not eval_root.exists():
        return experiments

    for exp_dir in sorted([p for p in eval_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        ocr_summary = load_json(exp_dir / "ocr_metrics_summary.json")
        real_eval_summary = load_json(exp_dir / "real_eval_summary.json")
        enhance_summary = load_json(exp_dir / "enhance_summary.json")
        summary = load_json(exp_dir / "summary.json")
        report_path = detect_report_path(exp_dir)
        preview_path = detect_preview_path(exp_dir)

        if not any([ocr_summary, real_eval_summary, enhance_summary, summary, report_path, preview_path]):
            continue

        record = {
            "name": exp_dir.name,
            "ocr_summary": ocr_summary,
            "real_eval_summary": real_eval_summary,
            "enhance_summary": enhance_summary,
            "summary": summary,
            "report_path": rel(report_path, project_root) if report_path else "",
            "preview_path": rel(preview_path, project_root) if preview_path else "",
            "path": rel(exp_dir, project_root),
        }
        experiments.append(record)

    return experiments


def collect_models(model_dir: Path, project_root: Path) -> List[Dict]:
    models: List[Dict] = []
    if not model_dir.exists():
        return models

    for path in sorted(model_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        models.append(
            {
                "name": path.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
                "path": rel(path, project_root),
            }
        )
    return models


def pick_best_real_ocr(experiments: List[Dict]) -> Optional[Dict]:
    candidates = []
    for exp in experiments:
        summary = exp.get("ocr_summary") or {}
        if not summary:
            continue
        samples = summary.get("samples")
        accuracy = summary.get("accuracy")
        cer = summary.get("cer")
        wer = summary.get("wer")
        if samples is None or accuracy is None:
            continue
        candidates.append((accuracy, -(cer or 999), -(wer or 999), samples, exp))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
    return best[-1]


def build_html(project_root: Path, experiments: List[Dict], models: List[Dict]) -> str:
    best_exp = pick_best_real_ocr(experiments)
    best_card = ""
    if best_exp:
        best = best_exp["ocr_summary"]
        best_card = f"""
        <section class="hero-card">
          <div class="eyebrow">当前最佳真实 OCR 结果</div>
          <h2>{html.escape(best_exp['name'])}</h2>
          <div class="metric-grid">
            <div><span>准确率</span><strong>{format_float(best.get('accuracy'))}</strong></div>
            <div><span>CER</span><strong>{format_float(best.get('cer'))}</strong></div>
            <div><span>WER</span><strong>{format_float(best.get('wer'))}</strong></div>
            <div><span>样本数</span><strong>{best.get('samples', '-')}</strong></div>
          </div>
        </section>
        """

    exp_cards = []
    for exp in experiments:
        ocr = exp.get("ocr_summary") or {}
        real_eval = exp.get("real_eval_summary") or {}
        enhance = exp.get("enhance_summary") or {}
        summary = real_eval or enhance or exp.get("summary") or {}
        method_stats = summary.get("method_stats") or {}
        diffusion_stats = method_stats.get("diffusion") or {}
        report_link = (
            f'<a class="btn" href="../{html.escape(exp["report_path"])}" target="_blank">打开报告</a>'
            if exp.get("report_path")
            else ""
        )
        folder_link = f'<a class="btn secondary" href="../{html.escape(exp["path"])}" target="_blank">打开目录</a>'
        preview = ""
        if exp.get("preview_path"):
            preview = f'<img class="preview" src="../{html.escape(exp["preview_path"])}" alt="{html.escape(exp["name"])} preview" />'

        exp_cards.append(
            f"""
            <article class="card">
              <div class="card-top">
                <div>
                  <div class="eyebrow">实验</div>
                  <h3>{html.escape(exp['name'])}</h3>
                </div>
                <div class="path">{html.escape(exp['path'])}</div>
              </div>
              {preview}
              <div class="mini-grid">
                <div><span>OCR 准确率</span><strong>{format_float(ocr.get('accuracy'))}</strong></div>
                <div><span>CER</span><strong>{format_float(ocr.get('cer'))}</strong></div>
                <div><span>WER</span><strong>{format_float(ocr.get('wer'))}</strong></div>
                <div><span>样本数</span><strong>{ocr.get('samples', '-')}</strong></div>
                <div><span>总耗时</span><strong>{format_float(summary.get('elapsed_sec'))}</strong></div>
                <div><span>扩散平均耗时</span><strong>{format_float(diffusion_stats.get('avg_sec'))}</strong></div>
              </div>
              <div class="actions">{report_link}{folder_link}</div>
            </article>
            """
        )

    rows = []
    for exp in experiments:
        ocr = exp.get("ocr_summary") or {}
        rows.append(
            f"""
            <tr>
              <td>{html.escape(exp['name'])}</td>
              <td>{ocr.get('samples', '-')}</td>
              <td>{format_float(ocr.get('accuracy'))}</td>
              <td>{format_float(ocr.get('cer'))}</td>
              <td>{format_float(ocr.get('wer'))}</td>
              <td>{html.escape(exp['path'])}</td>
            </tr>
            """
        )

    model_rows = []
    for model in models[:20]:
        model_rows.append(
            f"""
            <tr>
              <td>{html.escape(model['name'])}</td>
              <td>{model['size_mb']}</td>
              <td>{html.escape(model['path'])}</td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TextSR 实验看板</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255,255,255,0.88);
      --ink: #1d2330;
      --muted: #5f6877;
      --line: rgba(29,35,48,0.12);
      --accent: #b14d24;
      --accent-soft: #f0d6c9;
      --shadow: 0 20px 50px rgba(40, 29, 13, 0.08);
      --radius: 24px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(177,77,36,0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(52,112,92,0.12), transparent 24%),
        linear-gradient(180deg, #faf6ef, var(--bg));
    }}
    .wrap {{
      width: min(1220px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .masthead {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
    }}
    .title {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 30px;
      padding: 28px;
    }}
    .title h1 {{
      margin: 0 0 10px;
      font-size: clamp(30px, 4vw, 56px);
      line-height: 1.02;
      letter-spacing: -0.03em;
    }}
    .title p {{
      margin: 0;
      color: var(--muted);
      font-size: 16px;
    }}
    .hero-card {{
      background: linear-gradient(135deg, #fff6ef, #fffdf9);
      border: 1px solid rgba(177,77,36,0.18);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
      padding: 22px;
    }}
    .eyebrow {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent);
      margin-bottom: 8px;
      font-weight: 700;
    }}
    .hero-card h2 {{
      margin: 0 0 14px;
      font-size: 28px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }}
    .metric-grid div, .mini-grid div {{
      background: rgba(255,255,255,0.84);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px 14px;
    }}
    .metric-grid span, .mini-grid span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .metric-grid strong, .mini-grid strong {{
      font-size: 20px;
    }}
    .section {{
      margin-top: 28px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 24px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }}
    .card h3 {{
      margin: 0;
      font-size: 22px;
    }}
    .path {{
      color: var(--muted);
      font-size: 12px;
      max-width: 160px;
      text-align: right;
      word-break: break-word;
    }}
    .preview {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #eee7dc;
      margin-bottom: 12px;
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 0 14px;
      border-radius: 999px;
      text-decoration: none;
      color: white;
      background: var(--accent);
      font-weight: 600;
    }}
    .btn.secondary {{
      color: var(--ink);
      background: var(--accent-soft);
    }}
    .table-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: rgba(177,77,36,0.08);
      font-size: 13px;
    }}
    td {{
      font-size: 14px;
    }}
    @media (max-width: 720px) {{
      .wrap {{ width: min(100% - 20px, 1220px); }}
      .title {{ padding: 22px; }}
      .card {{ padding: 16px; }}
      .mini-grid {{ grid-template-columns: 1fr 1fr; }}
      .card-top {{ flex-direction: column; }}
      .path {{ text-align: left; max-width: none; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="masthead">
      <section class="title">
        <div class="eyebrow">可视化平台</div>
        <h1>TextSR 实验看板</h1>
        <p>这个页面会扫描 <code>eval_outputs</code> 和 <code>model</code>，汇总你们当前所有实验结果。后面继续换模型、补数据、升级评测，这个页面都还能复用。</p>
      </section>
      {best_card}
    </div>

    <section class="section">
      <h2>实验结果卡片</h2>
      <div class="card-grid">
        {''.join(exp_cards) if exp_cards else '<p>还没有扫描到实验结果。</p>'}
      </div>
    </section>

    <section class="section">
      <h2>OCR 汇总表</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>实验</th>
              <th>样本数</th>
              <th>准确率</th>
              <th>CER</th>
              <th>WER</th>
              <th>目录</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows) if rows else '<tr><td colspan="6">还没有检测到 OCR 汇总结果。</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>最近的模型权重</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>权重文件</th>
              <th>大小 (MB)</th>
              <th>路径</th>
            </tr>
          </thead>
          <tbody>
            {''.join(model_rows) if model_rows else '<tr><td colspan="3">还没有检测到权重文件。</td></tr>'}
          </tbody>
        </table>
      </div>
    </section>
  </div>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build a reusable HTML dashboard from experiment outputs.")
    parser.add_argument("--project_root", type=str, default=".", help="Project root directory")
    parser.add_argument("--eval_root", type=str, default="eval_outputs", help="Experiment output root")
    parser.add_argument("--model_dir", type=str, default="model", help="Checkpoint directory")
    parser.add_argument(
        "--output",
        type=str,
        default="dashboard/experiment_dashboard.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    eval_root = (project_root / args.eval_root).resolve() if not Path(args.eval_root).is_absolute() else Path(args.eval_root)
    model_dir = (project_root / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    output = (project_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    experiments = collect_experiments(eval_root, project_root)
    models = collect_models(model_dir, project_root)
    html_text = build_html(project_root, experiments, models)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    print(f"Dashboard written to: {output}")
    print(f"Experiments indexed: {len(experiments)}")
    print(f"Checkpoints indexed: {len(models)}")


if __name__ == "__main__":
    main()
