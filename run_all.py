import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parent

PRESETS: Dict[str, Dict[str, object]] = {
	"fast": {
		"steps": 80,
		"min_side": 256,
		"fallback_min_side": 192,
		"outscale": 4.0,
		"enhance_strength": 0.95,
	},
	"balanced": {
		"steps": 160,
		"min_side": 320,
		"fallback_min_side": 256,
		"outscale": 4.0,
		"enhance_strength": 1.0,
	},
	"best": {
		"steps": 240,
		"min_side": 384,
		"fallback_min_side": 256,
		"outscale": 4.0,
		"enhance_strength": 1.0,
	},
	"text-fast": {
		"steps": 120,
		"min_side": 288,
		"fallback_min_side": 224,
		"outscale": 4.0,
		"enhance_strength": 1.0,
		"luma_strength": 0.90,
		"max_luma_delta": 16.0,
		"color_lock_strength": 0.95,
		"edge_sharpen_strength": 0.20,
	},
	"text-balanced": {
		"steps": 180,
		"min_side": 352,
		"fallback_min_side": 256,
		"outscale": 4.0,
		"enhance_strength": 1.0,
		"luma_strength": 1.00,
		"max_luma_delta": 14.0,
		"color_lock_strength": 0.98,
		"edge_sharpen_strength": 0.35,
	},
	"text-best": {
		"steps": 260,
		"min_side": 384,
		"fallback_min_side": 288,
		"outscale": 4.0,
		"enhance_strength": 1.0,
		"luma_strength": 1.00,
		"max_luma_delta": 12.0,
		"color_lock_strength": 1.00,
		"edge_sharpen_strength": 0.45,
	},
}

MODEL_PROFILES: Dict[str, Dict[str, str]] = {
	"text-priority": {
		"model_path": "model/diffusion_textzoom_bs8_latest.pth",
	},
	"natural-priority": {
		"model_path": "model/diffusion_natural_latest.pth",
	},
}


def load_custom_presets() -> Dict[str, Dict[str, object]]:
	config_path = ROOT_DIR / "config" / "custom_presets.json"
	if not config_path.exists():
		return {}
	try:
		with open(config_path, "r", encoding="utf-8") as f:
			payload = json.load(f)
		presets = payload.get("presets", {}) if isinstance(payload, dict) else {}
		if not isinstance(presets, dict):
			return {}

		result = {}
		for name, entry in presets.items():
			if not isinstance(entry, dict):
				continue
			values = entry.get("values", {})
			if isinstance(values, dict):
				result[str(name)] = values
		return result
	except Exception:
		return {}


def merged_presets() -> Dict[str, Dict[str, object]]:
	merged = dict(PRESETS)
	merged.update(load_custom_presets())
	return merged


def resolve_active_model_profile(default_profile: str = "text-priority") -> str:
	state_path = ROOT_DIR / "config" / "model_state.json"
	if not state_path.exists():
		return default_profile
	try:
		with open(state_path, "r", encoding="utf-8") as f:
			state = json.load(f)
		if isinstance(state, dict):
			p = state.get("active_profile")
			if isinstance(p, str) and p:
				return p
	except Exception:
		pass
	return default_profile


def resolve_model_path_from_registry(profile: str) -> Optional[str]:
	registry_path = ROOT_DIR / "config" / "model_registry.json"
	state_path = ROOT_DIR / "config" / "model_state.json"
	if not registry_path.exists():
		return None

	try:
		with open(registry_path, "r", encoding="utf-8") as f:
			registry = json.load(f)
		state = {}
		if state_path.exists():
			with open(state_path, "r", encoding="utf-8") as f:
				state = json.load(f)

		profiles = registry.get("profiles") if isinstance(registry, dict) else None
		models = registry.get("models") if isinstance(registry, dict) else None
		if not isinstance(profiles, dict) or not isinstance(models, dict):
			return None
		if profile not in profiles:
			return None

		pm = profiles[profile]
		if not isinstance(pm, dict):
			return None
		model_id = str(pm.get("model") or "")
		if model_id not in models:
			return None

		mm = models[model_id]
		versions = mm.get("versions") if isinstance(mm, dict) else None
		if not isinstance(versions, dict):
			return None

		overrides = state.get("profile_overrides") if isinstance(state, dict) else None
		override_version = overrides.get(profile) if isinstance(overrides, dict) else None
		version = str(override_version or pm.get("default_version") or "")
		if version not in versions:
			return None

		vm = versions[version]
		if not isinstance(vm, dict):
			return None
		path = vm.get("path")
		if isinstance(path, str) and path:
			if os.path.isabs(path):
				return path
			return str(ROOT_DIR / path)
	except Exception:
		return None

	return None


def run_python_script(script_path: Path, args: List[str], dry_run: bool = False) -> int:
	cmd = [sys.executable, str(script_path)] + args
	print("\n[run_all] Executing:")
	print(" ".join(cmd))
	if dry_run:
		print("[run_all] Dry run mode, command not executed.")
		return 0

	completed = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
	return int(completed.returncode)


def build_batch_script_args(args: argparse.Namespace) -> List[str]:
	script_args = [
		"--input_dir", args.input_dir,
		"--output_dir", args.output_dir,
		"--methods", args.methods,
		"--diffusion_model_profile", args.model_profile,
		"--diffusion_profile_name", args.preset,
		"--outscale", str(int(round(args.outscale))),
		"--diffusion_model_path", args.model_path,
		"--diffusion_steps", str(args.steps),
		"--diffusion_min_side", str(args.min_side),
		"--diffusion_fallback_min_side", str(args.fallback_min_side),
		"--diffusion_outscale", str(args.outscale),
		"--diffusion_enhance_strength", str(args.enhance_strength),
		"--diffusion_luma_strength", str(args.luma_strength),
		"--diffusion_max_luma_delta", str(args.max_luma_delta),
		"--diffusion_color_lock_strength", str(args.color_lock_strength),
		"--diffusion_edge_sharpen_strength", str(args.edge_sharpen_strength),
		"--diffusion_tile_size", str(args.tile_size),
		"--diffusion_tile_overlap", str(args.tile_overlap),
	]
	if args.no_tile_blend:
		script_args.append("--diffusion_no_tile_blend")
	if args.no_warmup:
		script_args.append("--diffusion_no_warmup")
	if getattr(args, "gt_dir", None):
		script_args.extend(["--gt_dir", args.gt_dir])
	if args.preserve_color:
		script_args.append("--diffusion_preserve_color")
	if args.strict_color_lock:
		script_args.append("--diffusion_strict_color_lock")
	if args.decoder_attn:
		script_args.append("--diffusion_decoder_attn")
	if getattr(args, "lpips", False):
		script_args.append("--lpips")
		script_args.extend(["--lpips_net", args.lpips_net])
	if args.resume:
		script_args.append("--resume")
	if args.fail_fast:
		script_args.append("--fail_fast")
	script_args.extend(["--failure_csv", args.failure_csv])
	script_args.extend(["--summary_json", args.summary_json])
	return script_args


def build_report_script_args(args: argparse.Namespace) -> List[str]:
	script_args = [
		"--output_dir", args.output_dir,
		"--report_html", args.report_html,
		"--report_json", args.report_json,
		"--history_json", args.history_json,
		"--title", args.title,
		"--max_rows", str(args.max_rows),
		"--score_weight_ssim", str(args.score_weight_ssim),
		"--score_weight_lpips", str(args.score_weight_lpips),
		"--score_weight_psnr", str(args.score_weight_psnr),
		"--score_weight_speed", str(args.score_weight_speed),
	]
	if args.summary_json:
		script_args.extend(["--summary_json", args.summary_json])
	if getattr(args, "baseline_report_json", None):
		script_args.extend(["--baseline_report_json", args.baseline_report_json])
	if getattr(args, "ocr_summary_json", None):
		script_args.extend(["--ocr_summary_json", args.ocr_summary_json])
	if getattr(args, "ocr_detail_csv", None):
		script_args.extend(["--ocr_detail_csv", args.ocr_detail_csv])
	if getattr(args, "no_writeback_summary", False):
		script_args.append("--no_writeback_summary")
	if getattr(args, "no_append_history", False):
		script_args.append("--no_append_history")
	return script_args


def add_report_scoring_args(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--score_weight_ssim", type=float, default=0.45, help="Composite score weight for SSIM")
	parser.add_argument("--score_weight_lpips", type=float, default=0.25, help="Composite score weight for LPIPS")
	parser.add_argument("--score_weight_psnr", type=float, default=0.20, help="Composite score weight for PSNR")
	parser.add_argument("--score_weight_speed", type=float, default=0.10, help="Composite score weight for speed")
	parser.add_argument("--report_json", type=str, default="report_summary.json", help="Output report summary json path")
	parser.add_argument("--history_json", type=str, default="report_history.json", help="Report history json path")
	parser.add_argument("--baseline_report_json", type=str, default=None, help="Baseline report summary json path")
	parser.add_argument("--no_writeback_summary", action="store_true", help="Do not write report analysis back to summary json")
	parser.add_argument("--no_append_history", action="store_true", help="Do not append report history")
	parser.add_argument("--ocr_summary_json", type=str, default=None, help="Optional OCR summary json path for report")
	parser.add_argument("--ocr_detail_csv", type=str, default=None, help="Optional OCR detail csv path for report")


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
	preset = merged_presets().get(args.preset)
	if not preset:
		return args

	for key, value in preset.items():
		if hasattr(args, key):
			current = getattr(args, key)
			if current is None:
				setattr(args, key, value)
	return args


def ensure_default_int(args: argparse.Namespace, field: str, default: int) -> None:
	if getattr(args, field) is None:
		setattr(args, field, default)


def ensure_default_float(args: argparse.Namespace, field: str, default: float) -> None:
	if getattr(args, field) is None:
		setattr(args, field, default)


def resolve_under_output_dir(output_dir: str, path: str) -> str:
	if os.path.isabs(path):
		return path
	if os.path.dirname(path):
		return path
	return os.path.join(output_dir, path)


def add_shared_diffusion_args(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--preset", type=str, default="balanced", help="Quality preset (built-in or custom preset name)")
	parser.add_argument("--model_profile", type=str, default="active", help="Model profile: text-priority/natural-priority/active")
	parser.add_argument("--model_path", type=str, default=None, help="Diffusion checkpoint path (override model profile default)")
	parser.add_argument("--outscale", type=float, default=None, help="Output scale factor")
	parser.add_argument("--steps", type=int, default=None, help="Diffusion sampling steps")
	parser.add_argument("--min_side", type=int, default=None, help="Minimum side length before diffusion")
	parser.add_argument("--fallback_min_side", type=int, default=None, help="Fallback min side on CUDA OOM")
	parser.add_argument("--enhance_strength", type=float, default=None, help="Blend strength in [0,1]")
	parser.add_argument("--luma_strength", type=float, default=None, help="Luma enhancement strength in strict color lock")
	parser.add_argument("--max_luma_delta", type=float, default=None, help="Max luma delta clamp in strict color lock")
	parser.add_argument("--color_lock_strength", type=float, default=None, help="Color lock strength in strict mode")
	parser.add_argument("--edge_sharpen_strength", type=float, default=None, help="Edge sharpen strength for text details")
	parser.add_argument("--tile_size", type=int, default=None, help="Diffusion tile size for large image inference")
	parser.add_argument("--tile_overlap", type=int, default=None, help="Diffusion tile overlap for seam smoothing")
	parser.add_argument("--no_tile_blend", action="store_true", help="Disable diffusion tile blending")
	parser.add_argument("--preserve_color", action="store_true", help="Preserve original chroma")
	parser.add_argument("--strict_color_lock", action="store_true", help="Use strict color lock")
	parser.add_argument("--decoder_attn", action="store_true", help="Enable decoder attention")
	parser.add_argument("--no_warmup", action="store_true", help="Disable diffusion warmup")


def add_guardrail_args(parser: argparse.ArgumentParser, default_failure_csv: str, default_summary_json: str) -> None:
	parser.add_argument("--resume", action="store_true", help="Skip samples with existing outputs")
	parser.add_argument("--fail_fast", action="store_true", help="Stop immediately on first failure")
	parser.add_argument("--failure_csv", type=str, default=default_failure_csv, help="Failure csv output path")
	parser.add_argument("--summary_json", type=str, default=default_summary_json, help="Summary json output path")


def normalize_diffusion_args(args: argparse.Namespace) -> argparse.Namespace:
	args = apply_preset(args)
	if str(getattr(args, "model_profile", "")).strip().lower() == "active":
		args.model_profile = resolve_active_model_profile(default_profile="text-priority")
	ensure_default_float(args, "outscale", 4.0)
	ensure_default_int(args, "steps", 160)
	ensure_default_int(args, "min_side", 320)
	ensure_default_int(args, "fallback_min_side", 256)
	ensure_default_float(args, "enhance_strength", 1.0)
	ensure_default_float(args, "luma_strength", 1.0)
	ensure_default_float(args, "max_luma_delta", 24.0)
	ensure_default_float(args, "color_lock_strength", 1.0)
	ensure_default_float(args, "edge_sharpen_strength", 0.0)
	ensure_default_int(args, "tile_size", 0)
	ensure_default_int(args, "tile_overlap", 32)
	if getattr(args, "model_path", None) is None:
		args.model_path = (
			resolve_model_path_from_registry(args.model_profile)
			or MODEL_PROFILES.get(args.model_profile, MODEL_PROFILES["text-priority"])["model_path"]
		)
	if str(getattr(args, "preset", "")).startswith("text-"):
		args.strict_color_lock = True
	return args


def handle_enhance(args: argparse.Namespace) -> int:
	args = normalize_diffusion_args(args)
	script = ROOT_DIR / "inference_diffusion.py"
	script_args = [
		"-i", args.input,
		"-o", args.output,
		"--model_path", args.model_path,
		"--model_profile", args.model_profile,
		"--timesteps", str(args.steps),
		"--target_min_side", str(args.min_side),
		"--outscale", str(args.outscale),
		"--enhance_strength", str(args.enhance_strength),
		"--luma_strength", str(args.luma_strength),
		"--max_luma_delta", str(args.max_luma_delta),
		"--color_lock_strength", str(args.color_lock_strength),
		"--edge_sharpen_strength", str(args.edge_sharpen_strength),
		"--tile_size", str(args.tile_size),
		"--tile_overlap", str(args.tile_overlap),
	]
	if args.preserve_color:
		script_args.append("--preserve_color")
	if args.strict_color_lock:
		script_args.append("--strict_color_lock")
	if args.decoder_attn:
		script_args.append("--decoder_attn")
	if args.save_comparison:
		script_args.append("--save_comparison")
	if args.resume:
		script_args.append("--resume")
	if args.fail_fast:
		script_args.append("--fail_fast")
	if args.no_warmup:
		script_args.append("--no_warmup")
	if args.no_tile_blend:
		script_args.append("--no_tile_blend")
	script_args.extend(["--failure_csv", args.failure_csv])
	script_args.extend(["--summary_json", args.summary_json])
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_queue(args: argparse.Namespace) -> int:
	script = ROOT_DIR / "tools" / "task_queue_runner.py"
	script_args = [
		"--queue_json", args.queue_json,
		"--history_json", args.history_json,
	]
	if args.stop_on_error:
		script_args.append("--stop_on_error")
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_preset_manager(args: argparse.Namespace) -> int:
	script = ROOT_DIR / "tools" / "preset_manager.py"
	script_args = ["--action", args.action]
	if args.name:
		script_args.extend(["--name", args.name])
	if args.values_json:
		script_args.extend(["--values_json", args.values_json])
	if args.description:
		script_args.extend(["--description", args.description])
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_gui(args: argparse.Namespace) -> int:
	script = ROOT_DIR / "tools" / "desktop_gui.py"
	script_args = []
	if args.queue_json:
		script_args.extend(["--queue_json", args.queue_json])
	if args.history_json:
		script_args.extend(["--history_json", args.history_json])
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_model_registry(args: argparse.Namespace) -> int:
	script = ROOT_DIR / "tools" / "model_registry.py"
	script_args = ["--action", args.action]
	if args.model:
		script_args.extend(["--model", args.model])
	if args.output_dir:
		script_args.extend(["--output_dir", args.output_dir])
	if args.model_path:
		script_args.extend(["--model_path", args.model_path])
	if args.version:
		script_args.extend(["--version", args.version])
	if args.force:
		script_args.append("--force")
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_batch(args: argparse.Namespace) -> int:
	args = normalize_diffusion_args(args)
	script = ROOT_DIR / "tools" / "evaluate_text_models.py"
	script_args = build_batch_script_args(args)
	rc = run_python_script(script, script_args, dry_run=args.dry_run)
	if rc != 0 or not args.auto_report:
		return rc

	report_script = ROOT_DIR / "tools" / "generate_eval_report.py"
	report_title = args.auto_report_title or f"Text Enhancement {args.command.capitalize()} Report"
	report_args = [
		"--output_dir", args.output_dir,
		"--summary_json", args.summary_json,
		"--report_html", args.auto_report_html,
		"--report_json", args.report_json,
		"--history_json", args.history_json,
		"--title", report_title,
		"--max_rows", str(args.auto_report_max_rows),
		"--score_weight_ssim", str(args.score_weight_ssim),
		"--score_weight_lpips", str(args.score_weight_lpips),
		"--score_weight_psnr", str(args.score_weight_psnr),
		"--score_weight_speed", str(args.score_weight_speed),
	]
	if args.baseline_report_json:
		report_args.extend(["--baseline_report_json", args.baseline_report_json])
	if args.no_writeback_summary:
		report_args.append("--no_writeback_summary")
	if args.no_append_history:
		report_args.append("--no_append_history")
	return run_python_script(report_script, report_args, dry_run=args.dry_run)


def handle_compare(args: argparse.Namespace) -> int:
	# compare 是 batch 的语义化别名：固定输出对比面板
	args.methods = "bicubic,diffusion" if args.methods is None else args.methods
	return handle_batch(args)


def handle_ocr_eval(args: argparse.Namespace) -> int:
	script = ROOT_DIR / "tools" / "evaluate_ocr_metrics.py"
	script_args = [
		"--pred_dir", args.pred_dir,
		"--gt_csv", args.gt_csv,
		"--image_col", args.image_col,
		"--text_col", args.text_col,
		"--suffix", args.suffix,
		"--ocr_backend", args.ocr_backend,
		"--lang", args.lang,
		"--output_csv", args.output_csv,
		"--output_json", args.output_json,
	]
	if args.ppocr_root:
		script_args.extend(["--ppocr_root", args.ppocr_root])
	if args.use_angle_cls:
		script_args.append("--use_angle_cls")
	if args.use_gpu:
		script_args.append("--use_gpu")
	if args.device:
		script_args.extend(["--device", args.device])
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_labels_init(args: argparse.Namespace) -> int:
	script = ROOT_DIR / "tools" / "init_eval_labels_csv.py"
	script_args = [
		"--input_dir", args.input_dir,
		"--output_csv", args.output_csv,
	]
	if args.manifest_csv:
		script_args.extend(["--manifest_csv", args.manifest_csv])
	if args.keep_missing:
		script_args.append("--keep_missing")
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_report(args: argparse.Namespace) -> int:
	script = ROOT_DIR / "tools" / "generate_eval_report.py"
	script_args = build_report_script_args(args)
	return run_python_script(script, script_args, dry_run=args.dry_run)


def handle_full_eval(args: argparse.Namespace) -> int:
	args = normalize_diffusion_args(args)
	if args.methods is None:
		args.methods = "bicubic,diffusion"
	ocr_output_csv = resolve_under_output_dir(args.output_dir, args.ocr_output_csv)
	ocr_output_json = resolve_under_output_dir(args.output_dir, args.ocr_output_json)

	batch_script = ROOT_DIR / "tools" / "evaluate_text_models.py"
	rc = run_python_script(batch_script, build_batch_script_args(args), dry_run=args.dry_run)
	if rc != 0:
		return rc

	ocr_script = ROOT_DIR / "tools" / "evaluate_ocr_metrics.py"
	ocr_args = [
		"--pred_dir", args.pred_dir,
		"--gt_csv", args.gt_csv,
		"--image_col", args.image_col,
		"--text_col", args.text_col,
		"--suffix", args.suffix,
		"--ocr_backend", args.ocr_backend,
		"--lang", args.lang,
		"--output_csv", ocr_output_csv,
		"--output_json", ocr_output_json,
	]
	if args.ppocr_root:
		ocr_args.extend(["--ppocr_root", args.ppocr_root])
	if args.use_angle_cls:
		ocr_args.append("--use_angle_cls")
	if args.use_gpu:
		ocr_args.append("--use_gpu")
	if args.device:
		ocr_args.extend(["--device", args.device])

	rc = run_python_script(ocr_script, ocr_args, dry_run=args.dry_run)
	if rc != 0:
		return rc

	report_script = ROOT_DIR / "tools" / "generate_eval_report.py"
	report_args = [
		"--output_dir", args.output_dir,
		"--summary_json", args.summary_json,
		"--report_html", args.report_html,
		"--report_json", args.report_json,
		"--history_json", args.history_json,
		"--title", args.title,
		"--max_rows", str(args.max_rows),
		"--score_weight_ssim", str(args.score_weight_ssim),
		"--score_weight_lpips", str(args.score_weight_lpips),
		"--score_weight_psnr", str(args.score_weight_psnr),
		"--score_weight_speed", str(args.score_weight_speed),
		"--ocr_summary_json", ocr_output_json,
		"--ocr_detail_csv", ocr_output_csv,
	]
	if args.baseline_report_json:
		report_args.extend(["--baseline_report_json", args.baseline_report_json])
	if args.no_writeback_summary:
		report_args.append("--no_writeback_summary")
	if args.no_append_history:
		report_args.append("--no_append_history")
	return run_python_script(report_script, report_args, dry_run=args.dry_run)


def handle_real_ocr_eval(args: argparse.Namespace) -> int:
	args = normalize_diffusion_args(args)
	if args.methods is None:
		args.methods = "bicubic,diffusion"
	ocr_output_csv = resolve_under_output_dir(args.output_dir, args.ocr_output_csv)
	ocr_output_json = resolve_under_output_dir(args.output_dir, args.ocr_output_json)

	batch_script = ROOT_DIR / "tools" / "evaluate_text_models.py"
	rc = run_python_script(batch_script, build_batch_script_args(args), dry_run=args.dry_run)
	if rc != 0:
		return rc

	ocr_script = ROOT_DIR / "tools" / "evaluate_ocr_metrics.py"
	pred_dir = args.pred_dir or os.path.join(args.output_dir, args.pred_method)
	ocr_args = [
		"--pred_dir", pred_dir,
		"--gt_csv", args.gt_csv,
		"--image_col", args.image_col,
		"--text_col", args.text_col,
		"--suffix", args.suffix,
		"--ocr_backend", args.ocr_backend,
		"--lang", args.lang,
		"--output_csv", ocr_output_csv,
		"--output_json", ocr_output_json,
	]
	if args.ppocr_root:
		ocr_args.extend(["--ppocr_root", args.ppocr_root])
	if args.use_angle_cls:
		ocr_args.append("--use_angle_cls")
	if args.use_gpu:
		ocr_args.append("--use_gpu")
	if args.device:
		ocr_args.extend(["--device", args.device])

	rc = run_python_script(ocr_script, ocr_args, dry_run=args.dry_run)
	if rc != 0:
		return rc

	report_script = ROOT_DIR / "tools" / "generate_eval_report.py"
	report_args = [
		"--output_dir", args.output_dir,
		"--summary_json", args.summary_json,
		"--report_html", args.report_html,
		"--report_json", args.report_json,
		"--history_json", args.history_json,
		"--title", args.title,
		"--max_rows", str(args.max_rows),
		"--score_weight_ssim", str(args.score_weight_ssim),
		"--score_weight_lpips", str(args.score_weight_lpips),
		"--score_weight_psnr", str(args.score_weight_psnr),
		"--score_weight_speed", str(args.score_weight_speed),
		"--ocr_summary_json", ocr_output_json,
		"--ocr_detail_csv", ocr_output_csv,
	]
	if args.baseline_report_json:
		report_args.extend(["--baseline_report_json", args.baseline_report_json])
	if args.no_writeback_summary:
		report_args.append("--no_writeback_summary")
	if args.no_append_history:
		report_args.append("--no_append_history")
	return run_python_script(report_script, report_args, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Unified entrypoint for text image enhancement workflows.",
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	p_enhance = subparsers.add_parser("enhance", help="Enhance a single image or directory with diffusion")
	p_enhance.add_argument("-i", "--input", type=str, required=True, help="Input image or folder")
	p_enhance.add_argument("-o", "--output", type=str, required=True, help="Output folder")
	add_shared_diffusion_args(p_enhance)
	p_enhance.add_argument("--save_comparison", action="store_true", help="Save input/output side-by-side image")
	add_guardrail_args(p_enhance, default_failure_csv="enhance_failures.csv", default_summary_json="enhance_summary.json")
	p_enhance.add_argument("--dry_run", action="store_true", help="Print command only")
	p_enhance.set_defaults(func=handle_enhance)

	p_batch = subparsers.add_parser("batch", help="Run unified multi-method batch evaluation")
	p_batch.add_argument("--input_dir", type=str, default="eval_inputs", help="Input image folder")
	p_batch.add_argument("--output_dir", type=str, default="eval_outputs/cmp_local", help="Output folder")
	p_batch.add_argument("--methods", type=str, default="bicubic,diffusion", help="Methods list")
	p_batch.add_argument("--gt_dir", type=str, default=None, help="Optional GT folder")
	p_batch.add_argument("--lpips", action="store_true", help="Enable LPIPS metric when GT is provided")
	p_batch.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS backbone")
	add_shared_diffusion_args(p_batch)
	add_guardrail_args(p_batch, default_failure_csv="batch_failures.csv", default_summary_json="batch_summary.json")
	p_batch.add_argument("--auto_report", action="store_true", help="Auto-generate HTML report after batch run")
	p_batch.add_argument("--auto_report_html", type=str, default="report.html", help="Auto report html filename/path")
	p_batch.add_argument("--auto_report_title", type=str, default=None, help="Optional auto report title")
	p_batch.add_argument("--auto_report_max_rows", type=int, default=120, help="Auto report max rows")
	add_report_scoring_args(p_batch)
	p_batch.add_argument("--dry_run", action="store_true", help="Print command only")
	p_batch.set_defaults(func=handle_batch)

	p_compare = subparsers.add_parser("compare", help="Generate bicubic vs diffusion comparison panels")
	p_compare.add_argument("--input_dir", type=str, default="eval_inputs", help="Input image folder")
	p_compare.add_argument("--output_dir", type=str, default="eval_outputs/cmp_local", help="Output folder")
	p_compare.add_argument("--methods", type=str, default=None, help="Optional methods override")
	p_compare.add_argument("--gt_dir", type=str, default=None, help="Optional GT folder")
	p_compare.add_argument("--lpips", action="store_true", help="Enable LPIPS metric when GT is provided")
	p_compare.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS backbone")
	add_shared_diffusion_args(p_compare)
	add_guardrail_args(p_compare, default_failure_csv="compare_failures.csv", default_summary_json="compare_summary.json")
	p_compare.add_argument("--auto_report", action="store_true", help="Auto-generate HTML report after compare run")
	p_compare.add_argument("--auto_report_html", type=str, default="report.html", help="Auto report html filename/path")
	p_compare.add_argument("--auto_report_title", type=str, default=None, help="Optional auto report title")
	p_compare.add_argument("--auto_report_max_rows", type=int, default=120, help="Auto report max rows")
	add_report_scoring_args(p_compare)
	p_compare.add_argument("--dry_run", action="store_true", help="Print command only")
	p_compare.set_defaults(func=handle_compare)

	p_ocr = subparsers.add_parser("ocr-eval", help="Run OCR metrics for prediction results")
	p_ocr.add_argument("--pred_dir", type=str, required=True, help="Predicted image folder")
	p_ocr.add_argument("--gt_csv", type=str, required=True, help="Ground-truth CSV")
	p_ocr.add_argument("--image_col", type=str, default="image", help="Image column in GT CSV")
	p_ocr.add_argument("--text_col", type=str, default="text", help="Text column in GT CSV")
	p_ocr.add_argument("--suffix", type=str, default="", help="Prediction filename suffix")
	p_ocr.add_argument("--ocr_backend", choices=["paddleocr", "tesseract", "rapidocr"], default="paddleocr", help="OCR backend")
	p_ocr.add_argument("--lang", type=str, default="ch", help="OCR language")
	p_ocr.add_argument("--ppocr_root", type=str, default=None, help="PaddleOCR model root")
	p_ocr.add_argument("--use_angle_cls", action="store_true", help="Enable angle classifier")
	p_ocr.add_argument("--use_gpu", action="store_true", help="Use GPU for OCR")
	p_ocr.add_argument("--device", choices=["cpu", "gpu"], default=None, help="OCR runtime device")
	p_ocr.add_argument("--output_csv", type=str, default="ocr_metrics_detail.csv", help="Output detail CSV")
	p_ocr.add_argument("--output_json", type=str, default="ocr_metrics_summary.json", help="Output summary JSON")
	p_ocr.add_argument("--dry_run", action="store_true", help="Print command only")
	p_ocr.set_defaults(func=handle_ocr_eval)

	p_labels = subparsers.add_parser("labels-init", help="Initialize or refresh eval_inputs/labels.csv for OCR")
	p_labels.add_argument("--input_dir", type=str, default="eval_inputs", help="Eval crop image folder")
	p_labels.add_argument("--output_csv", type=str, default="eval_inputs/labels.csv", help="Output labels.csv path")
	p_labels.add_argument("--manifest_csv", type=str, default=None, help="Optional manifest.csv path")
	p_labels.add_argument("--keep_missing", action="store_true", help="Keep existing rows whose image files are currently missing")
	p_labels.add_argument("--dry_run", action="store_true", help="Print command only")
	p_labels.set_defaults(func=handle_labels_init)

	p_report = subparsers.add_parser("report", help="Generate visual HTML report from evaluation outputs")
	p_report.add_argument("--output_dir", type=str, required=True, help="Evaluation output directory")
	p_report.add_argument("--summary_json", type=str, default=None, help="Optional summary json path")
	p_report.add_argument("--report_html", type=str, default="report.html", help="Report HTML output filename/path")
	p_report.add_argument("--title", type=str, default="Text Enhancement Evaluation Report", help="Report title")
	p_report.add_argument("--max_rows", type=int, default=120, help="Maximum table/gallery rows")
	add_report_scoring_args(p_report)
	p_report.add_argument("--dry_run", action="store_true", help="Print command only")
	p_report.set_defaults(func=handle_report)

	p_full = subparsers.add_parser("full-eval", help="Run image metrics + OCR metrics + HTML report in one command")
	p_full.add_argument("--input_dir", type=str, default="eval_inputs", help="Input image folder")
	p_full.add_argument("--output_dir", type=str, default="eval_outputs/cmp_local", help="Output folder")
	p_full.add_argument("--methods", type=str, default="bicubic,diffusion", help="Methods list")
	p_full.add_argument("--gt_dir", type=str, required=True, help="GT image directory for image metrics")
	p_full.add_argument("--lpips", action="store_true", help="Enable LPIPS metric when GT is provided")
	p_full.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS backbone")
	add_shared_diffusion_args(p_full)
	add_guardrail_args(p_full, default_failure_csv="full_eval_failures.csv", default_summary_json="full_eval_summary.json")
	p_full.add_argument("--pred_dir", type=str, default="eval_outputs/cmp_local/diffusion", help="Prediction folder for OCR eval")
	p_full.add_argument("--gt_csv", type=str, required=True, help="GT CSV for OCR eval")
	p_full.add_argument("--image_col", type=str, default="image", help="Image column in GT CSV")
	p_full.add_argument("--text_col", type=str, default="text", help="Text column in GT CSV")
	p_full.add_argument("--suffix", type=str, default="", help="Optional prediction filename suffix for OCR eval")
	p_full.add_argument("--ocr_backend", choices=["paddleocr", "tesseract", "rapidocr"], default="paddleocr", help="OCR backend")
	p_full.add_argument("--lang", type=str, default="ch", help="OCR language")
	p_full.add_argument("--ppocr_root", type=str, default=None, help="PaddleOCR model root")
	p_full.add_argument("--use_angle_cls", action="store_true", help="Enable angle classifier for OCR")
	p_full.add_argument("--use_gpu", action="store_true", help="Use GPU for OCR")
	p_full.add_argument("--device", choices=["cpu", "gpu"], default=None, help="OCR runtime device")
	p_full.add_argument("--ocr_output_csv", type=str, default="ocr_metrics_detail.csv", help="OCR detail csv path")
	p_full.add_argument("--ocr_output_json", type=str, default="ocr_metrics_summary.json", help="OCR summary json path")
	p_full.add_argument("--report_html", type=str, default="full_eval_report.html", help="Report HTML output filename/path")
	p_full.add_argument("--title", type=str, default="Text Enhancement Full Evaluation Report", help="Report title")
	p_full.add_argument("--max_rows", type=int, default=120, help="Maximum rows in report sections")
	add_report_scoring_args(p_full)
	p_full.add_argument("--dry_run", action="store_true", help="Print command only")
	p_full.set_defaults(func=handle_full_eval)

	p_real = subparsers.add_parser("real-ocr-eval", help="Run real-image enhancement + OCR metrics + HTML report without GT image folder")
	p_real.add_argument("--input_dir", type=str, default="eval_inputs", help="Input image folder")
	p_real.add_argument("--output_dir", type=str, default="eval_outputs/real_ocr_eval", help="Output folder")
	p_real.add_argument("--methods", type=str, default="bicubic,diffusion", help="Methods list")
	add_shared_diffusion_args(p_real)
	add_guardrail_args(p_real, default_failure_csv="real_eval_failures.csv", default_summary_json="real_eval_summary.json")
	p_real.add_argument("--pred_dir", type=str, default=None, help="Prediction folder for OCR eval; defaults to output_dir/pred_method")
	p_real.add_argument("--pred_method", type=str, default="diffusion", help="Method folder under output_dir used for OCR by default")
	p_real.add_argument("--gt_csv", type=str, default="eval_inputs/labels.csv", help="GT CSV for OCR eval")
	p_real.add_argument("--image_col", type=str, default="image", help="Image column in GT CSV")
	p_real.add_argument("--text_col", type=str, default="text", help="Text column in GT CSV")
	p_real.add_argument("--suffix", type=str, default="", help="Optional prediction filename suffix for OCR eval")
	p_real.add_argument("--ocr_backend", choices=["paddleocr", "tesseract", "rapidocr"], default="rapidocr", help="OCR backend")
	p_real.add_argument("--lang", type=str, default="ch", help="OCR language")
	p_real.add_argument("--ppocr_root", type=str, default=None, help="PaddleOCR model root")
	p_real.add_argument("--use_angle_cls", action="store_true", help="Enable angle classifier for OCR")
	p_real.add_argument("--use_gpu", action="store_true", help="Use GPU for OCR")
	p_real.add_argument("--device", choices=["cpu", "gpu"], default=None, help="OCR runtime device")
	p_real.add_argument("--ocr_output_csv", type=str, default="ocr_metrics_detail.csv", help="OCR detail csv path")
	p_real.add_argument("--ocr_output_json", type=str, default="ocr_metrics_summary.json", help="OCR summary json path")
	p_real.add_argument("--report_html", type=str, default="real_ocr_report.html", help="Report HTML output filename/path")
	p_real.add_argument("--title", type=str, default="Real Text OCR Evaluation Report", help="Report title")
	p_real.add_argument("--max_rows", type=int, default=120, help="Maximum rows in report sections")
	add_report_scoring_args(p_real)
	p_real.add_argument("--dry_run", action="store_true", help="Print command only")
	p_real.set_defaults(func=handle_real_ocr_eval)

	p_queue = subparsers.add_parser("queue", help="Run queued run_all tasks with history tracking")
	p_queue.add_argument("--queue_json", type=str, required=True, help="Queue json path")
	p_queue.add_argument("--history_json", type=str, default="queue_history.json", help="Queue history json path")
	p_queue.add_argument("--stop_on_error", action="store_true", help="Stop queue on first task failure")
	p_queue.add_argument("--dry_run", action="store_true", help="Print command only")
	p_queue.set_defaults(func=handle_queue)

	p_model = subparsers.add_parser("model-registry", help="Model registry operations (list/verify/download)")
	p_model.add_argument("--action", choices=["list", "verify", "download", "activate", "status"], required=True, help="Registry action")
	p_model.add_argument("--model", type=str, default=None, help="Model id")
	p_model.add_argument("--model_path", type=str, default=None, help="Local model path for verify")
	p_model.add_argument("--version", type=str, default=None, help="Model version override")
	p_model.add_argument("--output_dir", type=str, default="model", help="Output directory for download")
	p_model.add_argument("--force", action="store_true", help="Force redownload")
	p_model.add_argument("--dry_run", action="store_true", help="Print command only")
	p_model.set_defaults(func=handle_model_registry)

	p_preset = subparsers.add_parser("preset", help="Manage custom presets")
	p_preset.add_argument("--action", choices=["list", "set", "delete"], required=True, help="Preset action")
	p_preset.add_argument("--name", type=str, default=None, help="Preset name")
	p_preset.add_argument("--values_json", type=str, default=None, help="Preset values json object")
	p_preset.add_argument("--description", type=str, default="", help="Preset description")
	p_preset.add_argument("--dry_run", action="store_true", help="Print command only")
	p_preset.set_defaults(func=handle_preset_manager)

	p_gui = subparsers.add_parser("gui", help="Launch local desktop GUI")
	p_gui.add_argument("--queue_json", type=str, default="tasks.json", help="Default queue json path in GUI")
	p_gui.add_argument("--history_json", type=str, default="queue_history.json", help="Default queue history path in GUI")
	p_gui.add_argument("--dry_run", action="store_true", help="Print command only")
	p_gui.set_defaults(func=handle_gui)

	return parser


def main(argv: Optional[List[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	return int(args.func(args))


if __name__ == "__main__":
	raise SystemExit(main())
