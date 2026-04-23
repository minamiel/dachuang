import argparse
import csv
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2


@dataclass
class OCRResult:
    image: str
    gt_text: str
    pred_text: str
    cer: float
    wer: float
    exact_match: int


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev_diag = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            temp = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(
                dp[j] + 1,       # deletion
                dp[j - 1] + 1,   # insertion
                prev_diag + cost # substitution
            )
            prev_diag = temp
    return dp[-1]


def cer(gt: str, pred: str) -> float:
    gt = gt or ""
    pred = pred or ""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(gt, pred) / len(gt)


def wer(gt: str, pred: str) -> float:
    gt_tokens = (gt or "").split()
    pred_tokens = (pred or "").split()
    if len(gt_tokens) == 0:
        return 0.0 if len(pred_tokens) == 0 else 1.0
    return edit_distance("\u0001".join(gt_tokens), "\u0001".join(pred_tokens)) / len(gt_tokens)


def load_gt_map(gt_csv: str, image_col: str, text_col: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(gt_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if image_col not in reader.fieldnames or text_col not in reader.fieldnames:
            raise ValueError(
                f"CSV missing required columns. Found={reader.fieldnames}, need image_col={image_col}, text_col={text_col}"
            )
        for row in reader:
            image_name = os.path.basename(row[image_col]).strip()
            mapping[image_name] = row[text_col] or ""
    return mapping


def _resolve_model_dir(
    ppocr_root: str,
    user_dir: Optional[str],
    task_keyword: str,
    prefer_keyword: Optional[str] = None,
) -> str:
    if user_dir:
        if not os.path.isdir(user_dir):
            raise FileNotFoundError(f"Specified model dir does not exist: {user_dir}")
        return user_dir

    root = Path(ppocr_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"PPOCR root not found: {ppocr_root}. Please place PP-OCRv5 models under this folder or pass --det_model_dir/--rec_model_dir/--cls_model_dir explicitly."
        )

    candidates: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if task_keyword in name:
            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find '{task_keyword}' model folder under {ppocr_root}. "
            f"Expected folders like '*{task_keyword}*' (e.g. ch_PP-OCRv5_server_{task_keyword})."
        )

    if prefer_keyword:
        preferred = [p for p in candidates if prefer_keyword.lower() in p.name.lower()]
        if preferred:
            return str(sorted(preferred)[0])

    return str(sorted(candidates)[0])


def _map_lang_for_paddle(lang: str) -> str:
    # 兼容历史 tesseract 风格参数
    mapping = {
        "eng": "en",
        "en": "en",
        "chi_sim": "ch",
        "ch": "ch",
    }
    return mapping.get((lang or "").lower(), "ch")


def _infer_default_ppocr_root() -> str:
    """推断 PP-OCR 模型根目录。

    优先级：
    1) 环境变量 PPOCR_ROOT
    2) 仓库根目录/PPOCRv5
    3) 当前工作目录/PPOCRv5
    4) /data/dachuang/TEST/PPOCRv5（服务器约定路径）
    """
    env_root = os.environ.get("PPOCR_ROOT")
    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "PPOCRv5")
    candidates.append(Path.cwd() / "PPOCRv5")
    candidates.append(Path("/data/dachuang/TEST/PPOCRv5"))

    for c in candidates:
        if c.is_dir():
            return str(c)

    # 返回一个最常见默认值，后续由 _resolve_model_dir 报更详细错误
    return str(repo_root / "PPOCRv5")


def tesseract_ocr_text(image_bgr, lang: str) -> str:
    try:
        import pytesseract  # type: ignore
    except Exception as err:
        raise RuntimeError(
            "pytesseract is required for OCR metrics. Install with: pip install pytesseract "
            "and ensure Tesseract OCR executable is installed on system."
        ) from err

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb, lang=lang)
    return text or ""


def build_rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception as err:
        raise RuntimeError(
            "rapidocr_onnxruntime is required. Install with: pip install rapidocr_onnxruntime. "
            f"Current python: {sys.executable}."
        ) from err

    return RapidOCR()


def build_paddle_ocr(
    lang: str,
    ppocr_root: str,
    det_model_dir: Optional[str],
    rec_model_dir: Optional[str],
    cls_model_dir: Optional[str],
    use_angle_cls: bool,
    use_gpu: bool,
    device: Optional[str] = None,
):
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as err:
        raise RuntimeError(
            "paddleocr is required. Install with: pip install paddleocr. "
            f"Current python: {sys.executable}. "
            "Please ensure you run this script inside the same conda/venv where paddleocr is installed."
        ) from err

    resolved_det = _resolve_model_dir(
        ppocr_root=ppocr_root,
        user_dir=det_model_dir,
        task_keyword="det",
        prefer_keyword="pp-ocrv5_server",
    )
    resolved_rec = _resolve_model_dir(
        ppocr_root=ppocr_root,
        user_dir=rec_model_dir,
        task_keyword="rec",
        prefer_keyword="pp-ocrv5_server",
    )
    resolved_cls = _resolve_model_dir(
        ppocr_root=ppocr_root,
        user_dir=cls_model_dir,
        task_keyword="cls",
        prefer_keyword=None,
    )

    paddle_lang = _map_lang_for_paddle(lang)
    runtime_device = (device or ("gpu" if use_gpu else "cpu")).lower()

    # 优先尝试官网推荐的新参数风格；若当前 paddleocr 版本不支持则自动回退到旧参数。
    modern_kwargs = dict(
        lang=paddle_lang,
        device=runtime_device,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_detection_model_dir=resolved_det,
        text_recognition_model_dir=resolved_rec,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        show_log=False,
    )

    legacy_kwargs = dict(
        use_gpu=(runtime_device == "gpu"),
        lang=paddle_lang,
        det_model_dir=resolved_det,
        rec_model_dir=resolved_rec,
        cls_model_dir=resolved_cls,
        use_angle_cls=use_angle_cls,
        show_log=False,
    )

    try:
        return PaddleOCR(**modern_kwargs)
    except TypeError:
        return PaddleOCR(**legacy_kwargs)


def rapidocr_ocr_text(ocr_engine, image_bgr) -> str:
    parts: List[str] = []
    result, _ = ocr_engine(image_bgr)
    if not result:
        return ""

    for item in result:
        if not item or len(item) < 2:
            continue
        txt = str(item[1]).strip()
        if txt:
            parts.append(txt)
    return " ".join(parts)


def paddle_ocr_text(ocr_engine, image_bgr) -> str:
    parts: List[str] = []

    # 新版推荐接口：predict
    if hasattr(ocr_engine, "predict"):
        try:
            result = ocr_engine.predict(image_bgr)
            for res in result or []:
                rec_texts = getattr(res, "rec_texts", None)
                if rec_texts:
                    for txt in rec_texts:
                        s = str(txt).strip()
                        if s:
                            parts.append(s)
                elif isinstance(res, dict):
                    for txt in (res.get("rec_texts") or []):
                        s = str(txt).strip()
                        if s:
                            parts.append(s)
        except Exception:
            parts = []

    if parts:
        return " ".join(parts)

    # 旧版兼容接口：ocr
    result = ocr_engine.ocr(image_bgr, cls=True)
    if not result or not result[0]:
        return ""

    for item in result[0]:
        if not item or len(item) < 2:
            continue
        rec = item[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 1:
            txt = str(rec[0]).strip()
            if txt:
                parts.append(txt)
    return " ".join(parts)


def collect_prediction_paths(pred_dir: str, exts: Tuple[str, ...]) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(pred_dir):
        full = os.path.join(pred_dir, name)
        if os.path.isfile(full) and name.lower().endswith(exts):
            paths.append(full)
    paths.sort()
    return paths


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR metrics (Acc/CER/WER) from predicted images and GT text CSV.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted images")
    parser.add_argument("--gt_csv", type=str, required=True, help="CSV with ground-truth text labels")
    parser.add_argument("--image_col", type=str, default="image", help="Image column name in gt_csv")
    parser.add_argument("--text_col", type=str, default="text", help="Text label column name in gt_csv")
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix in prediction names, e.g. _diffusion")
    parser.add_argument("--lang", type=str, default="ch", help="OCR language, e.g. ch/en (also supports eng/chi_sim aliases)")
    parser.add_argument("--ocr_backend", type=str, default="paddleocr", choices=["paddleocr", "tesseract", "rapidocr"], help="OCR backend")
    parser.add_argument("--ppocr_root", type=str, default=None, help="PaddleOCR model root directory (auto-detected if omitted)")
    parser.add_argument("--det_model_dir", type=str, default=None, help="Explicit det model dir; overrides auto-discovery in --ppocr_root")
    parser.add_argument("--rec_model_dir", type=str, default=None, help="Explicit rec model dir; overrides auto-discovery in --ppocr_root")
    parser.add_argument("--cls_model_dir", type=str, default=None, help="Explicit cls model dir; overrides auto-discovery in --ppocr_root")
    parser.add_argument("--use_angle_cls", action="store_true", help="Enable angle classifier for PaddleOCR")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for PaddleOCR (requires paddlepaddle-gpu)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu"], help="PaddleOCR device, e.g. gpu; overrides --use_gpu")
    parser.add_argument("--output_csv", type=str, default="ocr_metrics_detail.csv", help="Per-sample output csv path")
    parser.add_argument("--output_json", type=str, default="ocr_metrics_summary.json", help="Summary output json path")
    args = parser.parse_args()
    if args.ocr_backend == "paddleocr" and not args.ppocr_root:
        args.ppocr_root = _infer_default_ppocr_root()

    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"pred_dir not found: {args.pred_dir}")
    if not os.path.isfile(args.gt_csv):
        raise FileNotFoundError(f"gt_csv not found: {args.gt_csv}")

    gt_map = load_gt_map(args.gt_csv, args.image_col, args.text_col)
    pred_paths = collect_prediction_paths(args.pred_dir, (".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    if not pred_paths:
        raise FileNotFoundError(f"No prediction images found under: {args.pred_dir}")

    ocr_engine = None
    if args.ocr_backend == "paddleocr":
        ocr_engine = build_paddle_ocr(
            lang=args.lang,
            ppocr_root=args.ppocr_root,
            det_model_dir=args.det_model_dir,
            rec_model_dir=args.rec_model_dir,
            cls_model_dir=args.cls_model_dir,
            use_angle_cls=args.use_angle_cls,
            use_gpu=args.use_gpu,
            device=args.device,
        )
    elif args.ocr_backend == "rapidocr":
        ocr_engine = build_rapidocr()

    results: List[OCRResult] = []
    skipped = 0

    for p in pred_paths:
        name = os.path.basename(p)
        stem, ext = os.path.splitext(name)

        if args.suffix and stem.endswith(args.suffix):
            gt_name = f"{stem[: -len(args.suffix)]}{ext}"
        else:
            gt_name = name

        gt_text_raw = gt_map.get(gt_name)
        if gt_text_raw is None:
            skipped += 1
            continue

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        if args.ocr_backend == "paddleocr":
            pred_text_raw = paddle_ocr_text(ocr_engine, img)
        elif args.ocr_backend == "rapidocr":
            pred_text_raw = rapidocr_ocr_text(ocr_engine, img)
        else:
            pred_text_raw = tesseract_ocr_text(img, lang=args.lang)

        gt_text = normalize_text(gt_text_raw)
        pred_text = normalize_text(pred_text_raw)
        c = cer(gt_text, pred_text)
        w = wer(gt_text, pred_text)
        em = 1 if gt_text == pred_text else 0
        results.append(OCRResult(image=name, gt_text=gt_text, pred_text=pred_text, cer=c, wer=w, exact_match=em))

    if not results:
        raise RuntimeError("No valid matched samples for OCR metrics. Please check --suffix and gt_csv columns.")

    avg_cer = sum(r.cer for r in results) / len(results)
    avg_wer = sum(r.wer for r in results) / len(results)
    acc = sum(r.exact_match for r in results) / len(results)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt_text", "pred_text", "cer", "wer", "exact_match"])
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "image": r.image,
                    "gt_text": r.gt_text,
                    "pred_text": r.pred_text,
                    "cer": f"{r.cer:.6f}",
                    "wer": f"{r.wer:.6f}",
                    "exact_match": r.exact_match,
                }
            )

    summary = {
        "samples": len(results),
        "skipped": skipped,
        "accuracy": acc,
        "cer": avg_cer,
        "wer": avg_wer,
        "pred_dir": args.pred_dir,
        "gt_csv": args.gt_csv,
        "suffix": args.suffix,
        "lang": args.lang,
        "ocr_backend": args.ocr_backend,
        "ppocr_root": os.path.abspath(args.ppocr_root) if args.ocr_backend == "paddleocr" else None,
        "detail_csv": os.path.abspath(args.output_csv),
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("OCR metrics done")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
