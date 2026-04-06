import json
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestE2ESmokeEval(unittest.TestCase):
    def test_bicubic_pipeline_smoke_with_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "inputs")
            gt_dir = os.path.join(tmpdir, "gt")
            out_dir = os.path.join(tmpdir, "out")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)

            # 构造轻量文本样本
            sample = np.full((24, 48, 3), 255, dtype=np.uint8)
            cv2.putText(sample, "AB", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            inp_path = os.path.join(input_dir, "sample.png")
            cv2.imwrite(inp_path, sample)

            # GT 采用同样 bicubic x2 生成，便于稳定验证
            gt = cv2.resize(sample, (sample.shape[1] * 2, sample.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
            gt_path = os.path.join(gt_dir, "sample.png")
            cv2.imwrite(gt_path, gt)

            cmd = [
                sys.executable,
                os.path.join(ROOT_DIR, "tools", "evaluate_text_models.py"),
                "--input_dir",
                input_dir,
                "--output_dir",
                out_dir,
                "--methods",
                "bicubic",
                "--outscale",
                "2",
                "--gt_dir",
                gt_dir,
                "--summary_json",
                "smoke_summary.json",
                "--metrics_csv",
                "smoke_metrics.csv",
            ]

            completed = subprocess.run(cmd, cwd=ROOT_DIR, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                self.fail(f"smoke command failed: {completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}")

            summary_path = os.path.join(out_dir, "smoke_summary.json")
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            self.assertEqual(summary.get("success"), 1)
            self.assertEqual(summary.get("failed"), 0)
            self.assertEqual(summary.get("metric_rows"), 1)
            self.assertIn("bicubic", summary.get("method_stats", {}))

            metrics_path = os.path.join(out_dir, "smoke_metrics.csv")
            self.assertTrue(os.path.exists(metrics_path))
            compare_path = os.path.join(out_dir, "comparisons", "sample_compare.png")
            self.assertTrue(os.path.exists(compare_path))


if __name__ == "__main__":
    unittest.main()
