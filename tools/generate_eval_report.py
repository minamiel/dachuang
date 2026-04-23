import argparse
import csv
from datetime import datetime, timezone
import glob
import html
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


TEXT_PRESET_CURVES: List[Dict[str, object]] = [
    {
        "preset": "text-fast",
        "recommended": True,
        "quality_tier": "文本可读性优先（轻锐化）",
        "speed_tier": "高吞吐",
        "expected_cost_vs_balanced": "约 0.7x ~ 0.8x",
        "params": {
            "steps": 120,
            "min_side": 288,
            "luma_strength": 0.90,
            "max_luma_delta": 16.0,
            "color_lock_strength": 0.95,
            "edge_sharpen_strength": 0.20,
        },
        "notes": "推荐批量巡检、实时预览与低时延场景。",
    },
    {
        "preset": "text-balanced",
        "recommended": True,
        "quality_tier": "默认推荐（质量/速度平衡）",
        "speed_tier": "中等",
        "expected_cost_vs_balanced": "1.0x（基线）",
        "params": {
            "steps": 180,
            "min_side": 352,
            "luma_strength": 1.00,
            "max_luma_delta": 14.0,
            "color_lock_strength": 0.98,
            "edge_sharpen_strength": 0.35,
        },
        "notes": "推荐默认上线档，适合多数店招/票据/截图文本增强。",
    },
    {
        "preset": "text-best",
        "recommended": False,
        "quality_tier": "文本细节极致（强锐化）",
        "speed_tier": "较慢",
        "expected_cost_vs_balanced": "约 1.3x ~ 1.5x",
        "params": {
            "steps": 260,
            "min_side": 384,
            "luma_strength": 1.00,
            "max_luma_delta": 12.0,
            "color_lock_strength": 1.00,
            "edge_sharpen_strength": 0.45,
        },
        "notes": "推荐离线高质量重建或关键样本复核。",
    },
]


def render_text_preset_curve_section(summary: Optional[Dict]) -> str:
    current_profile = None
    current_cfg = {}
    if isinstance(summary, dict):
        current_profile = summary.get("diffusion_profile_name")
        cfg = summary.get("diffusion_config")
        if isinstance(cfg, dict):
            current_cfg = cfg

    rows = []
    for item in TEXT_PRESET_CURVES:
        preset_name = str(item.get("preset"))
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        mark = "✅ 当前运行" if current_profile == preset_name else ("⭐ 推荐" if item.get("recommended") else "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(preset_name)}</td>"
            f"<td>{html.escape(str(item.get('quality_tier', '-')))}</td>"
            f"<td>{html.escape(str(item.get('speed_tier', '-')))}</td>"
            f"<td>{html.escape(str(item.get('expected_cost_vs_balanced', '-')))}</td>"
            f"<td>{html.escape(str(params.get('steps', '-')))}</td>"
            f"<td>{html.escape(str(params.get('min_side', '-')))}</td>"
            f"<td>{html.escape(str(params.get('edge_sharpen_strength', '-')))}</td>"
            f"<td>{html.escape(str(params.get('max_luma_delta', '-')))}</td>"
            f"<td>{html.escape(str(params.get('color_lock_strength', '-')))}</td>"
            f"<td>{html.escape(str(item.get('notes', '-')))}</td>"
            f"<td>{html.escape(mark)}</td>"
            "</tr>"
        )

    current_html = "<p>当前运行未提供 diffusion 配置快照。</p>"
    if current_cfg:
        current_rows = []
        for key in [
            "steps",
            "min_side",
            "fallback_min_side",
            "outscale",
            "enhance_strength",
            "strict_color_lock",
            "luma_strength",
            "max_luma_delta",
            "color_lock_strength",
            "edge_sharpen_strength",
        ]:
            if key in current_cfg:
                current_rows.append(
                    f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(current_cfg.get(key)))}</td></tr>"
                )
        current_html = (
            f"<div class='muted'>current profile: {html.escape(str(current_profile or 'custom'))}</div>"
            + f"<table><tbody>{''.join(current_rows)}</tbody></table>"
        )

    table_html = (
        "<table><thead><tr>"
        "<th>preset</th><th>quality tier</th><th>speed tier</th><th>expected cost</th>"
        "<th>steps</th><th>min_side</th><th>edge_sharpen</th><th>max_luma_delta</th><th>color_lock</th><th>notes</th><th>tag</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    migration_html = (
        "<div class='card'>"
        "<h3>不可逆迁移建议</h3>"
        "<ul>"
        "<li>默认上线使用 <code>text-balanced</code>，作为组织内统一基线，避免后续口径漂移。</li>"
        "<li>高并发场景降级到 <code>text-fast</code>，并在报告中保留 expected cost 记录。</li>"
        "<li>关键样本复核使用 <code>text-best</code>，其结果可作为质量兜底版本归档。</li>"
        "</ul>"
        "</div>"
    )
    return table_html + current_html + migration_html


def resolve_summary_path(output_dir: str, explicit_summary: Optional[str]) -> Optional[str]:
    if explicit_summary:
        p = explicit_summary if os.path.isabs(explicit_summary) else os.path.join(output_dir, explicit_summary)
        return p if os.path.exists(p) else None

    candidates = [
        "compare_summary.json",
        "batch_summary.json",
        "enhance_summary.json",
        "summary.json",
    ]
    for name in candidates:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_json(path: Optional[str]) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_rel(path: str, base_dir: str) -> str:
    return os.path.relpath(path, base_dir).replace("\\", "/")


def collect_compare_images(output_dir: str, max_rows: int) -> List[str]:
    comparison_dir = os.path.join(output_dir, "comparisons")
    paths = sorted(glob.glob(os.path.join(comparison_dir, "*_compare.png")))
    return paths[:max_rows]


def collect_method_rows(output_dir: str, max_rows: int) -> List[Dict[str, Optional[str]]]:
    method_dirs = {
        "bicubic": os.path.join(output_dir, "bicubic"),
        "realesrgan": os.path.join(output_dir, "realesrgan"),
        "diffusion": os.path.join(output_dir, "diffusion"),
    }
    comparison_dir = os.path.join(output_dir, "comparisons")

    names = set()
    for method, method_dir in method_dirs.items():
        if not os.path.isdir(method_dir):
            continue
        for p in glob.glob(os.path.join(method_dir, "*.png")):
            names.add((method, os.path.splitext(os.path.basename(p))[0]))

    # 以 diffusion/bicubic 输出文件名为主键（通常一致）
    image_names = sorted({name for _, name in names})[:max_rows]
    rows: List[Dict[str, Optional[str]]] = []
    for name in image_names:
        row = {"image": name}
        for method, method_dir in method_dirs.items():
            candidate = os.path.join(method_dir, f"{name}.png")
            row[method] = candidate if os.path.exists(candidate) else None
        compare = os.path.join(comparison_dir, f"{name}_compare.png")
        row["comparison"] = compare if os.path.exists(compare) else None
        rows.append(row)
    return rows


def parse_failure_csv(path: Optional[str], max_rows: int) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_rows:
                break
    return rows


def resolve_metrics_path(output_dir: str, summary: Optional[Dict]) -> Optional[str]:
    if summary and summary.get("metrics_csv"):
        p = str(summary["metrics_csv"])
        if os.path.exists(p):
            return p
    p_default = os.path.join(output_dir, "metrics.csv")
    if os.path.exists(p_default):
        return p_default
    return None


def resolve_ocr_summary_path(output_dir: str, explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        p = explicit_path if os.path.isabs(explicit_path) else os.path.join(output_dir, explicit_path)
        return p if os.path.exists(p) else None

    p_default = os.path.join(output_dir, "ocr_metrics_summary.json")
    if os.path.exists(p_default):
        return p_default
    return None


def resolve_ocr_detail_path(output_dir: str, explicit_path: Optional[str], ocr_summary: Optional[Dict]) -> Optional[str]:
    if explicit_path:
        p = explicit_path if os.path.isabs(explicit_path) else os.path.join(output_dir, explicit_path)
        return p if os.path.exists(p) else None

    if isinstance(ocr_summary, dict) and ocr_summary.get("detail_csv"):
        p = str(ocr_summary["detail_csv"])
        if os.path.exists(p):
            return p

    p_default = os.path.join(output_dir, "ocr_metrics_detail.csv")
    if os.path.exists(p_default):
        return p_default
    return None


def resolve_baseline_report_path(output_dir: str, explicit_path: Optional[str]) -> Optional[str]:
    if not explicit_path:
        return None
    p = explicit_path if os.path.isabs(explicit_path) else os.path.join(output_dir, explicit_path)
    return p if os.path.exists(p) else None


def resolve_report_json_path(output_dir: str, explicit_path: str) -> str:
    if os.path.isabs(explicit_path):
        return explicit_path
    return os.path.join(output_dir, explicit_path)


def resolve_history_json_path(output_dir: str, explicit_path: str) -> str:
    if os.path.isabs(explicit_path):
        return explicit_path
    return os.path.join(output_dir, explicit_path)


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_leaderboard(
    summary: Optional[Dict],
    metrics_path: Optional[str],
    weights: Dict[str, float],
) -> List[Dict[str, object]]:
    bucket: Dict[str, Dict[str, object]] = {}

    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = str(row.get("method", "")).strip()
                if not method:
                    continue
                b = bucket.setdefault(method, {"count": 0, "psnr_sum": 0.0, "ssim_sum": 0.0, "lpips_sum": 0.0, "lpips_count": 0})
                b["count"] = int(b["count"]) + 1
                psnr = _to_float(row.get("psnr"))
                ssim = _to_float(row.get("ssim"))
                lpips = _to_float(row.get("lpips"))
                if psnr is not None:
                    b["psnr_sum"] = float(b["psnr_sum"]) + psnr
                if ssim is not None:
                    b["ssim_sum"] = float(b["ssim_sum"]) + ssim
                if lpips is not None:
                    b["lpips_sum"] = float(b["lpips_sum"]) + lpips
                    b["lpips_count"] = int(b["lpips_count"]) + 1

    method_stats = summary.get("method_stats") if isinstance(summary, dict) else None
    leaderboard = []
    all_methods = set(bucket.keys())
    if isinstance(method_stats, dict):
        all_methods.update(method_stats.keys())

    for method in sorted(all_methods):
        b = bucket.get(method, {})
        count = int(b.get("count", 0))
        psnr_avg = (float(b["psnr_sum"]) / count) if count > 0 and "psnr_sum" in b else None
        ssim_avg = (float(b["ssim_sum"]) / count) if count > 0 and "ssim_sum" in b else None
        lpips_count = int(b.get("lpips_count", 0))
        lpips_avg = (float(b["lpips_sum"]) / lpips_count) if lpips_count > 0 else None

        avg_sec = None
        if isinstance(method_stats, dict) and method in method_stats:
            avg_sec = _to_float(method_stats[method].get("avg_sec"))

        leaderboard.append(
            {
                "method": method,
                "samples": count,
                "psnr": psnr_avg,
                "ssim": ssim_avg,
                "lpips": lpips_avg,
                "avg_sec": avg_sec,
                "composite_score": None,
            }
        )

    def assign_composite_score(rows: List[Dict[str, object]]) -> None:
        def min_max(values: List[float]):
            return min(values), max(values)

        def normalize(v: Optional[float], lo: float, hi: float, larger_better: bool) -> Optional[float]:
            if v is None:
                return None
            if abs(hi - lo) < 1e-12:
                return 1.0
            x = (v - lo) / (hi - lo)
            return x if larger_better else (1.0 - x)

        metrics = {
            "ssim": {"larger_better": True, "values": [r["ssim"] for r in rows if r["ssim"] is not None]},
            "lpips": {"larger_better": False, "values": [r["lpips"] for r in rows if r["lpips"] is not None]},
            "psnr": {"larger_better": True, "values": [r["psnr"] for r in rows if r["psnr"] is not None]},
            "speed": {"larger_better": False, "values": [r["avg_sec"] for r in rows if r["avg_sec"] is not None]},
        }

        bounds = {}
        for key, info in metrics.items():
            vals = info["values"]
            if vals:
                bounds[key] = min_max(vals)

        for row in rows:
            weighted_sum = 0.0
            weighted_total = 0.0

            key_to_value = {
                "ssim": row.get("ssim"),
                "lpips": row.get("lpips"),
                "psnr": row.get("psnr"),
                "speed": row.get("avg_sec"),
            }
            key_to_weight = {
                "ssim": float(weights.get("ssim", 0.0)),
                "lpips": float(weights.get("lpips", 0.0)),
                "psnr": float(weights.get("psnr", 0.0)),
                "speed": float(weights.get("speed", 0.0)),
            }

            for key, value in key_to_value.items():
                weight = key_to_weight[key]
                if weight <= 0:
                    continue
                if key not in bounds:
                    continue
                lo, hi = bounds[key]
                norm = normalize(value, lo, hi, metrics[key]["larger_better"])
                if norm is None:
                    continue
                weighted_sum += weight * norm
                weighted_total += weight

            row["composite_score"] = (weighted_sum / weighted_total * 100.0) if weighted_total > 0 else None

    assign_composite_score(leaderboard)

    leaderboard.sort(
        key=lambda x: (
            -(x["composite_score"] if x["composite_score"] is not None else -1e9),
            -(x["ssim"] if x["ssim"] is not None else -1e9),
            (x["lpips"] if x["lpips"] is not None else 1e9),
            (x["avg_sec"] if x["avg_sec"] is not None else 1e9),
        )
    )
    return leaderboard


def build_baseline_comparison(
    leaderboard: List[Dict[str, object]],
    baseline_report: Optional[Dict],
) -> List[Dict[str, object]]:
    if not baseline_report or not isinstance(baseline_report, dict):
        return []

    baseline_rows = baseline_report.get("leaderboard")
    if not isinstance(baseline_rows, list):
        return []

    base_index = {str(row.get("method")): row for row in baseline_rows if isinstance(row, dict) and row.get("method")}
    deltas: List[Dict[str, object]] = []

    for row in leaderboard:
        method = str(row.get("method"))
        old = base_index.get(method)
        if not old:
            continue

        def delta(cur, prev):
            c = _to_float(cur)
            p = _to_float(prev)
            if c is None or p is None:
                return None
            return c - p

        deltas.append(
            {
                "method": method,
                "delta_score": delta(row.get("composite_score"), old.get("composite_score")),
                "delta_psnr": delta(row.get("psnr"), old.get("psnr")),
                "delta_ssim": delta(row.get("ssim"), old.get("ssim")),
                "delta_lpips": delta(row.get("lpips"), old.get("lpips")),
                "delta_avg_sec": delta(row.get("avg_sec"), old.get("avg_sec")),
            }
        )

    deltas.sort(key=lambda x: x["method"])
    return deltas


def render_summary_table(summary: Optional[Dict]) -> str:
    if not summary:
        return "<p>未找到 summary.json，以下仅展示对比图与失败样本（若存在）。</p>"

    keys = [
        "total_images",
        "success",
        "failed",
        "skipped_existing",
        "elapsed_sec",
        "avg_sec_per_success",
        "resume",
        "fail_fast",
    ]
    rows = []
    for k in keys:
        if k in summary:
            rows.append(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(summary[k]))}</td></tr>")

    method_stats = summary.get("method_stats")
    method_html = ""
    if isinstance(method_stats, dict) and method_stats:
        method_rows = []
        for method, stats in method_stats.items():
            method_rows.append(
                "<tr>"
                f"<td>{html.escape(str(method))}</td>"
                f"<td>{html.escape(str(stats.get('success')))}</td>"
                f"<td>{html.escape(str(stats.get('total_sec')))}</td>"
                f"<td>{html.escape(str(stats.get('avg_sec')))}</td>"
                "</tr>"
            )
        method_html = (
            "<h3>方法级统计</h3>"
            "<table><thead><tr><th>method</th><th>success</th><th>total_sec</th><th>avg_sec</th></tr></thead>"
            f"<tbody>{''.join(method_rows)}</tbody></table>"
        )

    err_counts = summary.get("error_type_counts")
    err_html = ""
    if isinstance(err_counts, dict) and err_counts:
        err_rows = [f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in err_counts.items()]
        err_html = (
            "<h3>错误类型分布</h3>"
            "<table><thead><tr><th>error_type</th><th>count</th></tr></thead>"
            f"<tbody>{''.join(err_rows)}</tbody></table>"
        )

    slow_samples = summary.get("top_slowest_samples")
    slow_html = ""
    if isinstance(slow_samples, list) and slow_samples:
        rows_s = []
        for item in slow_samples[:10]:
            rows_s.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('image')))}</td>"
                f"<td>{html.escape(str(item.get('elapsed_sec')))}</td>"
                "</tr>"
            )
        slow_html = (
            "<h3>最慢样本 Top10</h3>"
            "<table><thead><tr><th>image</th><th>elapsed_sec</th></tr></thead>"
            f"<tbody>{''.join(rows_s)}</tbody></table>"
        )

    fail_samples = summary.get("top_failure_samples")
    fail_html = ""
    if isinstance(fail_samples, list) and fail_samples:
        rows_f = []
        for item in fail_samples[:10]:
            rows_f.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('image')))}</td>"
                f"<td>{html.escape(str(item.get('error_type')))}</td>"
                f"<td>{html.escape(str(item.get('elapsed_sec')))}</td>"
                f"<td>{html.escape(str(item.get('error')))}</td>"
                "</tr>"
            )
        fail_html = (
            "<h3>失败样本摘要 Top10</h3>"
            "<table><thead><tr><th>image</th><th>error_type</th><th>elapsed_sec</th><th>error</th></tr></thead>"
            f"<tbody>{''.join(rows_f)}</tbody></table>"
        )

    return f"<table><tbody>{''.join(rows)}</tbody></table>{method_html}{err_html}{slow_html}{fail_html}"


def parse_ocr_detail_csv(path: Optional[str], max_rows: int) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def sort_key(row):
        cer_v = _to_float(row.get("cer"))
        return cer_v if cer_v is not None else -1.0

    rows.sort(key=sort_key, reverse=True)
    return rows[:max_rows]


def summarize_ocr_distribution(path: Optional[str]) -> Dict[str, object]:
    if not path or not os.path.exists(path):
        return {}

    cer_vals: List[float] = []
    wer_vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cer_v = _to_float(row.get("cer"))
            wer_v = _to_float(row.get("wer"))
            if cer_v is not None:
                cer_vals.append(cer_v)
            if wer_v is not None:
                wer_vals.append(wer_v)

    if not cer_vals and not wer_vals:
        return {}

    def stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "p50": None, "p90": None}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "mean": float(sum(sorted_vals) / n),
            "p50": float(sorted_vals[int(0.5 * (n - 1))]),
            "p90": float(sorted_vals[int(0.9 * (n - 1))]),
        }

    cer_hist = {
        "<=0.1": 0,
        "0.1-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-1.0": 0,
        ">1.0": 0,
    }
    for v in cer_vals:
        if v <= 0.1:
            cer_hist["<=0.1"] += 1
        elif v <= 0.2:
            cer_hist["0.1-0.2"] += 1
        elif v <= 0.4:
            cer_hist["0.2-0.4"] += 1
        elif v <= 0.6:
            cer_hist["0.4-0.6"] += 1
        elif v <= 1.0:
            cer_hist["0.6-1.0"] += 1
        else:
            cer_hist[">1.0"] += 1

    return {
        "samples": max(len(cer_vals), len(wer_vals)),
        "cer": stats(cer_vals),
        "wer": stats(wer_vals),
        "cer_hist": cer_hist,
    }


def load_history(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        return []
    return []


def write_history(path: str, history: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def build_svg_line_chart(values: List[float], width: int = 460, height: int = 130, color: str = "#2563eb") -> str:
    if not values:
        return "<p>-</p>"

    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1e-12

    points = []
    n = len(values)
    for i, value in enumerate(values):
        x = 10 + (width - 20) * (i / max(1, n - 1))
        y = 10 + (height - 20) * (1.0 - (value - lo) / (hi - lo))
        points.append(f"{x:.2f},{y:.2f}")

    polyline = " ".join(points)
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-label='trend chart'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='#e5e7eb'/>"
        f"<polyline points='{polyline}' fill='none' stroke='{color}' stroke-width='2'/>"
        "</svg>"
    )


def render_history_section(history: List[Dict[str, object]]) -> str:
    if not history:
        return "<p>暂无历史记录。</p>"

    tail = history[-20:]
    score_vals: List[float] = []
    acc_vals: List[float] = []
    elapsed_vals: List[float] = []
    rows = []

    for item in tail:
        score = _to_float(item.get("best_score"))
        acc = _to_float(item.get("ocr_accuracy"))
        elapsed = _to_float(item.get("elapsed_sec"))
        if score is not None:
            score_vals.append(score)
        if acc is not None:
            acc_vals.append(acc)
        if elapsed is not None:
            elapsed_vals.append(elapsed)

        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('generated_at', '-')))}</td>"
            f"<td>{html.escape(str(item.get('best_method', '-')))}</td>"
            f"<td>{html.escape(str(item.get('best_score', '-')))}</td>"
            f"<td>{html.escape(str(item.get('elapsed_sec', '-')))}</td>"
            f"<td>{html.escape(str(item.get('ocr_accuracy', '-')))}</td>"
            f"<td>{html.escape(str(item.get('ocr_cer', '-')))}</td>"
            "</tr>"
        )

    charts = (
        "<div class='trend-grid'>"
        "<div><h4>best score trend</h4>" + build_svg_line_chart(score_vals, color="#16a34a") + "</div>"
        "<div><h4>ocr accuracy trend</h4>" + build_svg_line_chart(acc_vals, color="#2563eb") + "</div>"
        "<div><h4>elapsed sec trend</h4>" + build_svg_line_chart(elapsed_vals, color="#dc2626") + "</div>"
        "</div>"
    )
    table = (
        "<table><thead><tr><th>time</th><th>best_method</th><th>best_score</th><th>elapsed_sec</th><th>ocr_acc</th><th>ocr_cer</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return charts + table


def extract_method_trends(history: List[Dict[str, object]], max_methods: int = 4, max_points: int = 20) -> List[Dict[str, object]]:
    if not history:
        return []

    tail = history[-max_points:]
    method_freq: Dict[str, int] = {}
    method_last_score: Dict[str, float] = {}

    for item in tail:
        method_scores = item.get("method_scores")
        if not isinstance(method_scores, dict):
            best_method = item.get("best_method")
            best_score = _to_float(item.get("best_score"))
            if isinstance(best_method, str) and best_method:
                method_scores = {best_method: best_score}
            else:
                method_scores = {}

        for method, raw_score in method_scores.items():
            m = str(method)
            score = _to_float(raw_score)
            method_freq[m] = method_freq.get(m, 0) + 1
            if score is not None:
                method_last_score[m] = score

    if not method_freq:
        return []

    methods = sorted(
        method_freq.keys(),
        key=lambda m: (
            -method_freq.get(m, 0),
            -(method_last_score.get(m, -1e9)),
            m,
        ),
    )[:max_methods]

    trends: List[Dict[str, object]] = []
    for method in methods:
        score_vals: List[float] = []
        sec_vals: List[float] = []
        rounds = 0
        for item in tail:
            method_scores = item.get("method_scores")
            method_avg_sec = item.get("method_avg_sec")

            if not isinstance(method_scores, dict):
                best_method = item.get("best_method")
                best_score = _to_float(item.get("best_score"))
                method_scores = {best_method: best_score} if isinstance(best_method, str) and best_method else {}
            if not isinstance(method_avg_sec, dict):
                method_avg_sec = {}

            score = _to_float(method_scores.get(method))
            sec = _to_float(method_avg_sec.get(method))
            if score is not None:
                score_vals.append(score)
                rounds += 1
            if sec is not None:
                sec_vals.append(sec)

        trends.append(
            {
                "method": method,
                "rounds": rounds,
                "score_values": score_vals,
                "avg_sec_values": sec_vals,
            }
        )

    return trends


def render_method_trends_section(history: List[Dict[str, object]]) -> str:
    trends = extract_method_trends(history)
    if not trends:
        return "<p>暂无可用方法级趋势数据（可通过新版本 history 累积后自动显示）。</p>"

    cards = []
    table_rows = []
    for item in trends:
        method = html.escape(str(item.get("method")))
        score_vals = item.get("score_values") if isinstance(item.get("score_values"), list) else []
        sec_vals = item.get("avg_sec_values") if isinstance(item.get("avg_sec_values"), list) else []
        rounds = int(item.get("rounds", 0))

        cards.append(
            "<div class='card'>"
            f"<h4>{method} - score trend</h4>"
            + build_svg_line_chart([float(v) for v in score_vals], color="#16a34a")
            + f"<div class='caption'>rounds={rounds}</div>"
            + "</div>"
        )
        cards.append(
            "<div class='card'>"
            f"<h4>{method} - avg sec trend</h4>"
            + build_svg_line_chart([float(v) for v in sec_vals], color="#dc2626")
            + f"<div class='caption'>points={len(sec_vals)}</div>"
            + "</div>"
        )

        latest_score = score_vals[-1] if score_vals else None
        latest_sec = sec_vals[-1] if sec_vals else None
        table_rows.append(
            "<tr>"
            f"<td>{method}</td>"
            f"<td>{rounds}</td>"
            f"<td>{'-' if latest_score is None else f'{float(latest_score):.2f}'}</td>"
            f"<td>{'-' if latest_sec is None else f'{float(latest_sec):.4f}'}</td>"
            "</tr>"
        )

    return (
        "<div class='trend-grid'>" + "".join(cards) + "</div>"
        + "<table><thead><tr><th>method</th><th>history rounds</th><th>latest score</th><th>latest avg sec</th></tr></thead>"
        + f"<tbody>{''.join(table_rows)}</tbody></table>"
    )


def generate_html(
    output_dir: str,
    report_path: str,
    title: str,
    summary: Optional[Dict],
    summary_path: Optional[str],
    max_rows: int,
    weights: Dict[str, float],
    ocr_summary_path: Optional[str],
    ocr_detail_path: Optional[str],
    baseline_report_path: Optional[str],
    history_tail: List[Dict[str, object]],
    ocr_distribution: Dict[str, object],
) -> str:
    compare_paths = collect_compare_images(output_dir, max_rows=max_rows)
    method_rows = collect_method_rows(output_dir, max_rows=max_rows)

    failure_csv = None
    if summary and summary.get("failure_csv"):
        failure_csv = summary["failure_csv"]
    failure_rows = parse_failure_csv(failure_csv, max_rows=max_rows)
    metrics_path = resolve_metrics_path(output_dir, summary)
    leaderboard = build_leaderboard(summary, metrics_path, weights=weights)
    baseline_report = load_json(baseline_report_path)
    baseline_deltas = build_baseline_comparison(leaderboard, baseline_report)
    ocr_summary = load_json(ocr_summary_path)
    ocr_detail_rows = parse_ocr_detail_csv(ocr_detail_path, max_rows=max_rows)

    compare_items = []
    for p in compare_paths:
        rel = to_rel(p, os.path.dirname(report_path))
        name = os.path.basename(p)
        compare_items.append(
            "<div class='card'>"
            f"<div class='caption'>{html.escape(name)}</div>"
            f"<img src='{html.escape(rel)}' loading='lazy' />"
            "</div>"
        )

    method_gallery_html = "<p>未找到方法输出目录（bicubic/realesrgan/diffusion）。</p>"
    if method_rows:
        m_rows = []
        base_dir = os.path.dirname(report_path)
        for row in method_rows:
            def td_img(path: Optional[str]) -> str:
                if not path:
                    return "<td>-</td>"
                rel = to_rel(path, base_dir)
                return f"<td><img class='thumb' src='{html.escape(rel)}' loading='lazy' /></td>"

            m_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('image', '')))}</td>"
                f"{td_img(row.get('bicubic'))}"
                f"{td_img(row.get('realesrgan'))}"
                f"{td_img(row.get('diffusion'))}"
                f"{td_img(row.get('comparison'))}"
                "</tr>"
            )
        method_gallery_html = (
            "<table><thead><tr><th>image</th><th>bicubic</th><th>realesrgan</th><th>diffusion</th><th>comparison</th></tr></thead>"
            f"<tbody>{''.join(m_rows)}</tbody></table>"
        )

    failure_html = "<p>无失败记录。</p>"
    if failure_rows:
        rows = []
        for row in failure_rows:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('image', '')))}</td>"
                f"<td>{html.escape(str(row.get('error_type', '')))}</td>"
                f"<td>{html.escape(str(row.get('error', '')))}</td>"
                "</tr>"
            )
        failure_html = (
            "<table><thead><tr><th>image</th><th>error_type</th><th>error</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    leaderboard_html = "<p>无可用排行榜数据（缺少 metrics.csv 或 method_stats）。</p>"
    if leaderboard:
        lb_rows = []
        for idx, row in enumerate(leaderboard, start=1):
            def fmt(v, nd=4):
                if v is None:
                    return "-"
                return f"{float(v):.{nd}f}"

            lb_rows.append(
                "<tr>"
                f"<td>{idx}</td>"
                f"<td>{html.escape(str(row['method']))}</td>"
                f"<td>{row['samples']}</td>"
                f"<td>{fmt(row['composite_score'], 2)}</td>"
                f"<td>{fmt(row['psnr'], 3)}</td>"
                f"<td>{fmt(row['ssim'], 4)}</td>"
                f"<td>{fmt(row['lpips'], 4)}</td>"
                f"<td>{fmt(row['avg_sec'], 4)}</td>"
                "</tr>"
            )
        leaderboard_html = (
            "<table><thead><tr><th>rank</th><th>method</th><th>samples</th><th>score↑</th><th>PSNR↑</th><th>SSIM↑</th><th>LPIPS↓</th><th>avg_sec↓</th></tr></thead>"
            f"<tbody>{''.join(lb_rows)}</tbody></table>"
        )

    baseline_html = "<p>无 baseline 对比数据。</p>"
    if baseline_deltas:
        def fmt(v, nd=4):
            if v is None:
                return "-"
            return f"{float(v):+.{nd}f}"

        b_rows = []
        for row in baseline_deltas:
            b_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row['method']))}</td>"
                f"<td>{fmt(row.get('delta_score'), 2)}</td>"
                f"<td>{fmt(row.get('delta_psnr'), 3)}</td>"
                f"<td>{fmt(row.get('delta_ssim'), 4)}</td>"
                f"<td>{fmt(row.get('delta_lpips'), 4)}</td>"
                f"<td>{fmt(row.get('delta_avg_sec'), 4)}</td>"
                "</tr>"
            )
        baseline_html = (
            "<table><thead><tr><th>method</th><th>Δscore</th><th>ΔPSNR</th><th>ΔSSIM</th><th>ΔLPIPS</th><th>Δavg_sec</th></tr></thead>"
            f"<tbody>{''.join(b_rows)}</tbody></table>"
        )

    ocr_summary_html = "<p>无 OCR summary 数据。</p>"
    if isinstance(ocr_summary, dict):
        ocr_rows = []
        for key in ["samples", "skipped", "accuracy", "cer", "wer", "ocr_backend", "lang"]:
            if key in ocr_summary:
                ocr_rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(ocr_summary[key]))}</td></tr>")
        if ocr_rows:
            ocr_summary_html = f"<table><tbody>{''.join(ocr_rows)}</tbody></table>"

    ocr_detail_html = "<p>无 OCR 细节数据。</p>"
    if ocr_detail_rows:
        detail_rows = []
        for row in ocr_detail_rows:
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('image', '')))}</td>"
                f"<td>{html.escape(str(row.get('cer', '')))}</td>"
                f"<td>{html.escape(str(row.get('wer', '')))}</td>"
                f"<td>{html.escape(str(row.get('gt_text', '')))}</td>"
                f"<td>{html.escape(str(row.get('pred_text', '')))}</td>"
                "</tr>"
            )
        ocr_detail_html = (
            "<table><thead><tr><th>image</th><th>cer</th><th>wer</th><th>gt_text</th><th>pred_text</th></tr></thead>"
            f"<tbody>{''.join(detail_rows)}</tbody></table>"
        )

    ocr_dist_html = "<p>无 OCR 分布统计。</p>"
    if ocr_distribution:
        cer_stats = ocr_distribution.get("cer") if isinstance(ocr_distribution, dict) else None
        wer_stats = ocr_distribution.get("wer") if isinstance(ocr_distribution, dict) else None
        cer_hist = ocr_distribution.get("cer_hist") if isinstance(ocr_distribution, dict) else None

        stat_rows = []
        if isinstance(cer_stats, dict):
            c_mean = _to_float(cer_stats.get("mean"))
            c_p50 = _to_float(cer_stats.get("p50"))
            c_p90 = _to_float(cer_stats.get("p90"))
            if c_mean is not None and c_p50 is not None and c_p90 is not None:
                stat_rows.append(f"<tr><th>CER mean / p50 / p90</th><td>{c_mean:.4f} / {c_p50:.4f} / {c_p90:.4f}</td></tr>")

        if isinstance(wer_stats, dict):
            w_mean = _to_float(wer_stats.get("mean"))
            w_p50 = _to_float(wer_stats.get("p50"))
            w_p90 = _to_float(wer_stats.get("p90"))
            if w_mean is not None and w_p50 is not None and w_p90 is not None:
                stat_rows.append(f"<tr><th>WER mean / p50 / p90</th><td>{w_mean:.4f} / {w_p50:.4f} / {w_p90:.4f}</td></tr>")

        hist_rows = []
        if isinstance(cer_hist, dict):
            for bin_name, count in cer_hist.items():
                hist_rows.append(f"<tr><td>{html.escape(str(bin_name))}</td><td>{html.escape(str(count))}</td></tr>")

        ocr_dist_html = (
            "<table><tbody>" + "".join(stat_rows) + "</tbody></table>"
            + "<table><thead><tr><th>CER bin</th><th>count</th></tr></thead><tbody>"
            + "".join(hist_rows)
            + "</tbody></table>"
        )

    summary_src = html.escape(summary_path or "(not found)")
    summary_block = render_summary_table(summary)
    ocr_summary_src = html.escape(ocr_summary_path or "(not found)")
    ocr_detail_src = html.escape(ocr_detail_path or "(not found)")
    baseline_src = html.escape(baseline_report_path or "(not found)")
    history_html = render_history_section(history_tail)
    method_history_html = render_method_trends_section(history_tail)
    preset_curve_html = render_text_preset_curve_section(summary)
    weight_text = (
        f"ssim={weights.get('ssim', 0):.3f}, "
        f"lpips={weights.get('lpips', 0):.3f}, "
        f"psnr={weights.get('psnr', 0):.3f}, "
        f"speed={weights.get('speed', 0):.3f}"
    )

    html_doc = f"""
<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    .muted {{ color: #6b7280; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px; background: #fff; }}
    .card img {{ width: 100%; border-radius: 6px; background: #f3f4f6; }}
    .caption {{ font-size: 12px; color: #4b5563; margin-bottom: 6px; word-break: break-all; }}
    .thumb {{ width: 180px; max-width: 100%; border-radius: 6px; display: block; background: #f3f4f6; }}
    .trend-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; margin: 10px 0 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class=\"muted\">summary source: {summary_src}</div>

  <h2>运行摘要</h2>
  {summary_block}

    <h2>方法排行榜（质量 + 速度）</h2>
    <div class="muted">score weights: {html.escape(weight_text)}</div>
    {leaderboard_html}

        <h2>与基线报告对比（涨跌）</h2>
        <div class="muted">baseline source: {baseline_src}</div>
        {baseline_html}

    <h2>OCR 指标摘要</h2>
    <div class="muted">ocr summary source: {ocr_summary_src}</div>
    {ocr_summary_html}

    <h2>OCR 错误样本（按 CER 降序，最多 {max_rows} 条）</h2>
    <div class="muted">ocr detail source: {ocr_detail_src}</div>
    {ocr_detail_html}

        <h2>OCR 分布统计（稳定性）</h2>
        {ocr_dist_html}

        <h2>多轮历史趋势（最近20轮）</h2>
        {history_html}

    <h2>方法级历史趋势（最近20轮）</h2>
    {method_history_html}

    <h2>文本 preset 速度/质量曲线（可解释）</h2>
    {preset_curve_html}

  <h2>失败样本（最多 {max_rows} 条）</h2>
  {failure_html}

    <h2>方法输出缩略图（最多 {max_rows} 条）</h2>
    {method_gallery_html}

  <h2>对比图画廊（最多 {max_rows} 张）</h2>
  <div class=\"grid\">{''.join(compare_items) if compare_items else '<p>未找到 comparisons/*_compare.png</p>'}</div>
</body>
</html>
""".strip()

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from evaluation outputs")
    parser.add_argument("--output_dir", type=str, required=True, help="Evaluation output directory")
    parser.add_argument("--summary_json", type=str, default=None, help="Optional summary json path")
    parser.add_argument("--report_html", type=str, default="report.html", help="Output HTML filename/path")
    parser.add_argument("--title", type=str, default="Text Enhancement Evaluation Report", help="HTML title")
    parser.add_argument("--max_rows", type=int, default=120, help="Max rows/images to render")
    parser.add_argument("--ocr_summary_json", type=str, default=None, help="Optional OCR summary json path")
    parser.add_argument("--ocr_detail_csv", type=str, default=None, help="Optional OCR detail csv path")
    parser.add_argument("--score_weight_ssim", type=float, default=0.45, help="Composite score weight for SSIM")
    parser.add_argument("--score_weight_lpips", type=float, default=0.25, help="Composite score weight for LPIPS")
    parser.add_argument("--score_weight_psnr", type=float, default=0.20, help="Composite score weight for PSNR")
    parser.add_argument("--score_weight_speed", type=float, default=0.10, help="Composite score weight for speed")
    parser.add_argument("--report_json", type=str, default="report_summary.json", help="Output report summary json path")
    parser.add_argument("--baseline_report_json", type=str, default=None, help="Baseline report summary json path")
    parser.add_argument("--no_writeback_summary", action="store_true", help="Do not write report analysis back to summary json")
    parser.add_argument("--history_json", type=str, default="report_history.json", help="History json path")
    parser.add_argument("--no_append_history", action="store_true", help="Do not append this run to history")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    summary_path = resolve_summary_path(output_dir, args.summary_json)
    summary = load_json(summary_path)
    baseline_report_path = resolve_baseline_report_path(output_dir, args.baseline_report_json)
    baseline_report = load_json(baseline_report_path)
    ocr_summary_path = resolve_ocr_summary_path(output_dir, args.ocr_summary_json)
    ocr_summary = load_json(ocr_summary_path)
    ocr_detail_path = resolve_ocr_detail_path(output_dir, args.ocr_detail_csv, ocr_summary)

    report_path = args.report_html
    if not os.path.isabs(report_path):
        report_path = os.path.join(output_dir, report_path)
    report_json_path = resolve_report_json_path(output_dir, args.report_json)
    history_json_path = resolve_history_json_path(output_dir, args.history_json)
    history = load_history(history_json_path)

    weight_total = args.score_weight_ssim + args.score_weight_lpips + args.score_weight_psnr + args.score_weight_speed
    if weight_total <= 0:
        weights = {"ssim": 0.45, "lpips": 0.25, "psnr": 0.20, "speed": 0.10}
    else:
        weights = {
            "ssim": args.score_weight_ssim / weight_total,
            "lpips": args.score_weight_lpips / weight_total,
            "psnr": args.score_weight_psnr / weight_total,
            "speed": args.score_weight_speed / weight_total,
        }

    generated = generate_html(
        output_dir=output_dir,
        report_path=report_path,
        title=args.title,
        summary=summary,
        summary_path=summary_path,
        max_rows=max(1, args.max_rows),
        weights=weights,
        ocr_summary_path=ocr_summary_path,
        ocr_detail_path=ocr_detail_path,
        baseline_report_path=baseline_report_path,
        history_tail=history[-20:],
        ocr_distribution=summarize_ocr_distribution(ocr_detail_path),
    )

    metrics_path = resolve_metrics_path(output_dir, summary)
    leaderboard = build_leaderboard(summary, metrics_path, weights=weights)
    baseline_deltas = build_baseline_comparison(leaderboard, baseline_report)

    report_payload = {
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "title": args.title,
        "output_dir": output_dir,
        "summary_json": summary_path,
        "report_html": os.path.abspath(generated),
        "weights": weights,
        "leaderboard": leaderboard,
        "method_scores": {
            str(row.get("method")): row.get("composite_score")
            for row in leaderboard
            if isinstance(row, dict) and row.get("method")
        },
        "method_avg_sec": {
            str(row.get("method")): row.get("avg_sec")
            for row in leaderboard
            if isinstance(row, dict) and row.get("method")
        },
        "baseline_report_json": baseline_report_path,
        "baseline_deltas": baseline_deltas,
        "ocr_summary_json": ocr_summary_path,
        "ocr_detail_csv": ocr_detail_path,
        "history_json": os.path.abspath(history_json_path),
        "text_preset_curves": TEXT_PRESET_CURVES,
        "diffusion_profile_name": summary.get("diffusion_profile_name") if isinstance(summary, dict) else None,
        "diffusion_config": summary.get("diffusion_config") if isinstance(summary, dict) else None,
    }
    os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    if not args.no_append_history:
        method_scores = {
            str(row.get("method")): row.get("composite_score")
            for row in leaderboard
            if isinstance(row, dict) and row.get("method")
        }
        method_avg_sec = {
            str(row.get("method")): row.get("avg_sec")
            for row in leaderboard
            if isinstance(row, dict) and row.get("method")
        }
        history.append(
            {
                "generated_at": report_payload["generated_at"],
                "title": args.title,
                "report_json": os.path.abspath(report_json_path),
                "best_method": leaderboard[0]["method"] if leaderboard else None,
                "best_score": leaderboard[0].get("composite_score") if leaderboard else None,
                "method_scores": method_scores,
                "method_avg_sec": method_avg_sec,
                "elapsed_sec": summary.get("elapsed_sec") if isinstance(summary, dict) else None,
                "ocr_accuracy": (ocr_summary or {}).get("accuracy") if isinstance(ocr_summary, dict) else None,
                "ocr_cer": (ocr_summary or {}).get("cer") if isinstance(ocr_summary, dict) else None,
            }
        )
        history = history[-200:]
        write_history(history_json_path, history)

    if summary is not None and summary_path and (not args.no_writeback_summary):
        summary["report_analysis"] = {
            "report_html": os.path.abspath(generated),
            "report_json": os.path.abspath(report_json_path),
            "weights": weights,
            "best_method": leaderboard[0]["method"] if leaderboard else None,
            "baseline_report_json": baseline_report_path,
            "baseline_deltas": baseline_deltas,
            "history_json": os.path.abspath(history_json_path),
            "diffusion_profile_name": summary.get("diffusion_profile_name") if isinstance(summary, dict) else None,
            "generated_at": report_payload["generated_at"],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Report summary json generated: {report_json_path}")
    print(f"Report generated: {generated}")


if __name__ == "__main__":
    main()
