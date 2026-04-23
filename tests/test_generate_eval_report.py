import csv
import os
import sys
import tempfile
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.generate_eval_report import extract_method_trends, summarize_ocr_distribution, render_text_preset_curve_section


class TestGenerateEvalReportHelpers(unittest.TestCase):
    def test_extract_method_trends_with_method_scores(self):
        history = [
            {
                "generated_at": "t1",
                "method_scores": {"bicubic": 30.0, "diffusion": 65.0},
                "method_avg_sec": {"bicubic": 0.02, "diffusion": 0.15},
            },
            {
                "generated_at": "t2",
                "method_scores": {"bicubic": 31.0, "diffusion": 67.0},
                "method_avg_sec": {"bicubic": 0.02, "diffusion": 0.16},
            },
        ]

        trends = extract_method_trends(history, max_methods=2, max_points=20)
        methods = [t["method"] for t in trends]

        self.assertIn("bicubic", methods)
        self.assertIn("diffusion", methods)

        diffusion = [t for t in trends if t["method"] == "diffusion"][0]
        self.assertEqual(diffusion["rounds"], 2)
        self.assertEqual(len(diffusion["score_values"]), 2)
        self.assertEqual(len(diffusion["avg_sec_values"]), 2)

    def test_extract_method_trends_backward_compatible_best_method(self):
        history = [
            {"generated_at": "t1", "best_method": "diffusion", "best_score": 70.0},
            {"generated_at": "t2", "best_method": "diffusion", "best_score": 72.0},
        ]

        trends = extract_method_trends(history, max_methods=2, max_points=20)
        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0]["method"], "diffusion")
        self.assertEqual(trends[0]["rounds"], 2)

    def test_summarize_ocr_distribution(self):
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "ocr_metrics_detail.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "cer", "wer"])
                writer.writeheader()
                writer.writerow({"image": "a.png", "cer": "0.05", "wer": "0.10"})
                writer.writerow({"image": "b.png", "cer": "0.20", "wer": "0.30"})
                writer.writerow({"image": "c.png", "cer": "0.50", "wer": "0.60"})

            dist = summarize_ocr_distribution(csv_path)
            self.assertEqual(dist["samples"], 3)
            self.assertIn("cer", dist)
            self.assertIn("wer", dist)
            self.assertIn("cer_hist", dist)
            self.assertEqual(dist["cer_hist"]["<=0.1"], 1)
            self.assertEqual(dist["cer_hist"]["0.1-0.2"], 1)

    def test_render_text_preset_curve_section(self):
        summary = {
            "diffusion_profile_name": "text-balanced",
            "diffusion_config": {
                "steps": 180,
                "min_side": 352,
                "edge_sharpen_strength": 0.35,
                "color_lock_strength": 0.98,
            },
        }
        html_text = render_text_preset_curve_section(summary)
        self.assertIn("text-fast", html_text)
        self.assertIn("text-balanced", html_text)
        self.assertIn("当前运行", html_text)
        self.assertIn("不可逆迁移建议", html_text)


if __name__ == "__main__":
    unittest.main()
