import os
import sys
import unittest
from unittest.mock import patch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import run_all


class TestRunAllCLI(unittest.TestCase):
    def test_compare_defaults_to_bicubic_diffusion_methods(self):
        parser = run_all.build_parser()
        args = parser.parse_args([
            "compare",
            "--input_dir",
            "eval_inputs",
            "--output_dir",
            "eval_outputs/cmp_local",
            "--dry_run",
        ])

        self.assertEqual(args.command, "compare")
        self.assertIsNone(args.methods)

    def test_preset_applies_when_not_explicit(self):
        parser = run_all.build_parser()
        args = parser.parse_args([
            "batch",
            "--preset",
            "fast",
            "--dry_run",
        ])

        normalized = run_all.normalize_diffusion_args(args)
        self.assertEqual(normalized.steps, 80)
        self.assertEqual(normalized.min_side, 256)
        self.assertEqual(normalized.fallback_min_side, 192)
        self.assertEqual(normalized.outscale, 4.0)

    def test_explicit_values_override_preset(self):
        parser = run_all.build_parser()
        args = parser.parse_args([
            "batch",
            "--preset",
            "best",
            "--steps",
            "111",
            "--min_side",
            "300",
            "--dry_run",
        ])

        normalized = run_all.normalize_diffusion_args(args)
        self.assertEqual(normalized.steps, 111)
        self.assertEqual(normalized.min_side, 300)

    def test_batch_guardrail_defaults_parse(self):
        parser = run_all.build_parser()
        args = parser.parse_args([
            "batch",
            "--dry_run",
        ])

        self.assertFalse(args.resume)
        self.assertFalse(args.fail_fast)
        self.assertEqual(args.failure_csv, "batch_failures.csv")
        self.assertEqual(args.summary_json, "batch_summary.json")
        self.assertFalse(args.auto_report)
        self.assertEqual(args.auto_report_html, "report.html")
        self.assertFalse(args.lpips)
        self.assertEqual(args.lpips_net, "alex")

    def test_text_preset_applies_text_defaults(self):
        parser = run_all.build_parser()
        args = parser.parse_args([
            "batch",
            "--preset",
            "text-balanced",
            "--dry_run",
        ])

        normalized = run_all.normalize_diffusion_args(args)
        self.assertEqual(normalized.steps, 180)
        self.assertEqual(normalized.min_side, 352)
        self.assertAlmostEqual(normalized.edge_sharpen_strength, 0.35)
        self.assertAlmostEqual(normalized.color_lock_strength, 0.98)
        self.assertTrue(normalized.strict_color_lock)

    def test_enhance_forwards_guardrail_flags(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "enhance",
                "-i",
                "eval_inputs",
                "-o",
                "eval_outputs/enhance_balanced",
                "--resume",
                "--fail_fast",
                "--failure_csv",
                "custom_failures.csv",
                "--summary_json",
                "custom_summary.json",
            ])

        self.assertEqual(code, 0)
        self.assertTrue(mocked.called)
        _, forwarded_args = mocked.call_args[0]
        self.assertIn("--resume", forwarded_args)
        self.assertIn("--fail_fast", forwarded_args)
        self.assertIn("custom_failures.csv", forwarded_args)
        self.assertIn("custom_summary.json", forwarded_args)

    def test_report_subcommand_parse_defaults(self):
        parser = run_all.build_parser()
        args = parser.parse_args([
            "report",
            "--output_dir",
            "eval_outputs/cmp_local",
            "--dry_run",
        ])

        self.assertEqual(args.command, "report")
        self.assertEqual(args.report_html, "report.html")
        self.assertEqual(args.max_rows, 120)
        self.assertAlmostEqual(args.score_weight_ssim, 0.45)
        self.assertAlmostEqual(args.score_weight_lpips, 0.25)
        self.assertAlmostEqual(args.score_weight_psnr, 0.20)
        self.assertAlmostEqual(args.score_weight_speed, 0.10)
        self.assertEqual(args.report_json, "report_summary.json")
        self.assertEqual(args.history_json, "report_history.json")
        self.assertIsNone(args.baseline_report_json)
        self.assertFalse(args.no_writeback_summary)
        self.assertFalse(args.no_append_history)

    def test_report_forwards_summary_and_title(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "report",
                "--output_dir",
                "eval_outputs/cmp_local",
                "--summary_json",
                "batch_summary.json",
                "--report_json",
                "current_report.json",
                "--baseline_report_json",
                "baseline_report.json",
                "--history_json",
                "history.json",
                "--no_writeback_summary",
                "--no_append_history",
                "--title",
                "My Report",
                "--report_html",
                "my_report.html",
                "--max_rows",
                "20",
            ])

        self.assertEqual(code, 0)
        self.assertTrue(mocked.called)
        _, forwarded_args = mocked.call_args[0]
        self.assertIn("--summary_json", forwarded_args)
        self.assertIn("batch_summary.json", forwarded_args)
        self.assertIn("--title", forwarded_args)
        self.assertIn("My Report", forwarded_args)
        self.assertIn("my_report.html", forwarded_args)
        self.assertIn("--report_json", forwarded_args)
        self.assertIn("current_report.json", forwarded_args)
        self.assertIn("--baseline_report_json", forwarded_args)
        self.assertIn("baseline_report.json", forwarded_args)
        self.assertIn("--history_json", forwarded_args)
        self.assertIn("history.json", forwarded_args)
        self.assertIn("--no_writeback_summary", forwarded_args)
        self.assertIn("--no_append_history", forwarded_args)

    def test_batch_auto_report_triggers_second_call(self):
        with patch("run_all.run_python_script", side_effect=[0, 0]) as mocked:
            code = run_all.main([
                "batch",
                "--dry_run",
                "--auto_report",
                "--auto_report_html",
                "auto_report.html",
                "--auto_report_title",
                "Batch Auto Report",
                "--auto_report_max_rows",
                "30",
            ])

        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_count, 2)
        report_call_args = mocked.call_args_list[1][0][1]
        self.assertIn("--report_html", report_call_args)
        self.assertIn("auto_report.html", report_call_args)
        self.assertIn("Batch Auto Report", report_call_args)

    def test_compare_auto_report_default_title(self):
        with patch("run_all.run_python_script", side_effect=[0, 0]) as mocked:
            code = run_all.main([
                "compare",
                "--dry_run",
                "--auto_report",
            ])

        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_count, 2)
        report_call_args = mocked.call_args_list[1][0][1]
        self.assertIn("Compare", " ".join(report_call_args))

    def test_batch_forwards_lpips_options(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "batch",
                "--dry_run",
                "--lpips",
                "--lpips_net",
                "vgg",
            ])

        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_count, 1)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--lpips", forwarded)
        self.assertIn("--lpips_net", forwarded)
        self.assertIn("vgg", forwarded)

    def test_enhance_forwards_text_postprocess_args(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "enhance",
                "-i",
                "eval_inputs",
                "-o",
                "eval_outputs/enhance_text",
                "--preset",
                "text-fast",
                "--edge_sharpen_strength",
                "0.5",
                "--color_lock_strength",
                "0.85",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--edge_sharpen_strength", forwarded)
        self.assertIn("0.5", forwarded)
        self.assertIn("--color_lock_strength", forwarded)
        self.assertIn("0.85", forwarded)

    def test_batch_forwards_text_postprocess_args(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "batch",
                "--dry_run",
                "--methods",
                "bicubic,diffusion",
                "--preset",
                "text-fast",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--diffusion_edge_sharpen_strength", forwarded)
        self.assertIn("--diffusion_color_lock_strength", forwarded)
        self.assertIn("--diffusion_luma_strength", forwarded)
        self.assertIn("--diffusion_profile_name", forwarded)
        self.assertIn("text-fast", forwarded)

    def test_batch_forwards_tile_args(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "batch",
                "--dry_run",
                "--methods",
                "bicubic,diffusion",
                "--tile_size",
                "384",
                "--tile_overlap",
                "48",
                "--no_tile_blend",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--diffusion_tile_size", forwarded)
        self.assertIn("384", forwarded)
        self.assertIn("--diffusion_tile_overlap", forwarded)
        self.assertIn("48", forwarded)
        self.assertIn("--diffusion_no_tile_blend", forwarded)

    def test_full_eval_runs_three_stages(self):
        with patch("run_all.run_python_script", side_effect=[0, 0, 0]) as mocked:
            code = run_all.main([
                "full-eval",
                "--dry_run",
                "--gt_dir",
                "eval_gt",
                "--gt_csv",
                "eval_inputs/labels.csv",
                "--pred_dir",
                "eval_outputs/cmp_local/diffusion",
                "--report_json",
                "full_eval_report_summary.json",
            ])

        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_count, 3)
        batch_args = mocked.call_args_list[0][0][1]
        ocr_args = mocked.call_args_list[1][0][1]
        report_args = mocked.call_args_list[2][0][1]

        self.assertIn("--gt_dir", batch_args)
        self.assertIn("eval_gt", batch_args)
        self.assertIn("--gt_csv", ocr_args)
        self.assertIn("eval_inputs/labels.csv", ocr_args)
        self.assertIn("--ocr_summary_json", report_args)
        self.assertIn("ocr_metrics_summary.json", report_args)
        self.assertIn("--report_json", report_args)
        self.assertIn("full_eval_report_summary.json", report_args)

    def test_report_forwards_score_weights(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "report",
                "--output_dir",
                "eval_outputs/cmp_local",
                "--score_weight_ssim",
                "0.5",
                "--score_weight_lpips",
                "0.1",
                "--score_weight_psnr",
                "0.3",
                "--score_weight_speed",
                "0.1",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--score_weight_ssim", forwarded)
        self.assertIn("0.5", forwarded)
        self.assertIn("--score_weight_lpips", forwarded)
        self.assertIn("0.1", forwarded)

    def test_queue_subcommand_forwards_args(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "queue",
                "--queue_json",
                "tasks.json",
                "--history_json",
                "queue_history.json",
                "--stop_on_error",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--queue_json", forwarded)
        self.assertIn("tasks.json", forwarded)
        self.assertIn("--history_json", forwarded)
        self.assertIn("queue_history.json", forwarded)
        self.assertIn("--stop_on_error", forwarded)

    def test_model_registry_subcommand_forwards_args(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "model-registry",
                "--action",
                "verify",
                "--model",
                "text-priority",
                "--model_path",
                "model/custom.pth",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--action", forwarded)
        self.assertIn("verify", forwarded)
        self.assertIn("--model", forwarded)
        self.assertIn("text-priority", forwarded)
        self.assertIn("--model_path", forwarded)
        self.assertIn("model/custom.pth", forwarded)

    def test_preset_subcommand_forwards_args(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "preset",
                "--action",
                "set",
                "--name",
                "demo",
                "--values_json",
                "{\"steps\": 150}",
                "--description",
                "demo preset",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--action", forwarded)
        self.assertIn("set", forwarded)
        self.assertIn("--name", forwarded)
        self.assertIn("demo", forwarded)

    def test_gui_subcommand_forwards_args(self):
        with patch("run_all.run_python_script", return_value=0) as mocked:
            code = run_all.main([
                "gui",
                "--queue_json",
                "tasks.json",
                "--history_json",
                "queue_history.json",
            ])

        self.assertEqual(code, 0)
        forwarded = mocked.call_args[0][1]
        self.assertIn("--queue_json", forwarded)
        self.assertIn("tasks.json", forwarded)
        self.assertIn("--history_json", forwarded)
        self.assertIn("queue_history.json", forwarded)


if __name__ == "__main__":
    unittest.main()
