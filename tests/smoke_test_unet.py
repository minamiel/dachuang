import os
import shutil
import sys
import unittest

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
	sys.path.insert(0, ROOT_DIR)

from inference_diffusion import build_model
from model_unet import SimpleUNet


class TestDiffusionUNetSmoke(unittest.TestCase):
	def test_concat_forward_shapes(self):
		model = SimpleUNet(cond_mode="concat")
		batch_size, h, w = 2, 64, 64
		x = torch.randn(batch_size, 6, h, w)
		t = torch.randint(0, 1000, (batch_size,))
		outputs = model(x, t)
		noise_pred = outputs["noise_pred"]
		mask_pred = outputs["mask_pred"]

		self.assertEqual(noise_pred.shape, (batch_size, 3, h, w))
		self.assertEqual(mask_pred.shape, (batch_size, 1, h, w))

	def test_structure_prior_forward_shapes(self):
		model = SimpleUNet(cond_mode="concat", use_structure_prior=True)
		batch_size, h, w = 2, 64, 64
		x = torch.randn(batch_size, 6, h, w)
		t = torch.randint(0, 1000, (batch_size,))
		structure_prior = torch.rand(batch_size, 1, h, w)
		outputs = model(x, t, structure_prior=structure_prior)

		self.assertEqual(outputs["noise_pred"].shape, (batch_size, 3, h, w))
		self.assertEqual(outputs["mask_pred"].shape, (batch_size, 1, h, w))

	def test_decoder_structure_gate_forward_shapes(self):
		model = SimpleUNet(
			cond_mode="concat",
			use_structure_prior=True,
			use_decoder_structure_gate=True,
			structure_gate_strength=1.2,
		)
		batch_size, h, w = 2, 64, 64
		x = torch.randn(batch_size, 6, h, w)
		t = torch.randint(0, 1000, (batch_size,))
		structure_prior = torch.rand(batch_size, 1, h, w)
		outputs = model(x, t, structure_prior=structure_prior)

		self.assertEqual(outputs["noise_pred"].shape, (batch_size, 3, h, w))
		self.assertEqual(outputs["mask_pred"].shape, (batch_size, 1, h, w))

	def test_build_model_with_checkpoint_payload(self):
		model = SimpleUNet(
			cond_mode="concat",
			use_structure_prior=True,
			use_decoder_structure_gate=True,
			structure_gate_strength=1.1,
			structure_gate_time_power=1.5,
		)
		payload = {
			"model_state": model.state_dict(),
			"optimizer_state": None,
			"epoch": 0,
			"avg_loss": 0.0,
			"config": {
				"cond_mode": "concat",
				"use_structure_prior": True,
				"structure_prior_strength": 1.25,
				"use_decoder_structure_gate": True,
				"structure_gate_strength": 1.1,
				"structure_gate_time_power": 1.5,
			},
		}

		tmpdir = os.path.join(ROOT_DIR, ".tmp_smoke_test_unet")
		os.makedirs(tmpdir, exist_ok=True)
		try:
			ckpt_path = os.path.join(tmpdir, "ckpt.pth")
			torch.save(payload, ckpt_path)
			loaded = build_model(ckpt_path, device="cpu")
		finally:
			shutil.rmtree(tmpdir, ignore_errors=True)

		self.assertIsInstance(loaded, SimpleUNet)
		self.assertEqual(getattr(loaded, "cond_mode", None), "concat")
		self.assertTrue(getattr(loaded, "use_structure_prior", False))
		self.assertTrue(getattr(loaded, "use_decoder_structure_gate", False))
		self.assertAlmostEqual(getattr(loaded, "structure_prior_strength", 0.0), 1.25)
		self.assertAlmostEqual(getattr(loaded, "structure_gate_strength", 0.0), 1.1)
		self.assertAlmostEqual(getattr(loaded, "structure_gate_time_power", 0.0), 1.5)


if __name__ == "__main__":
	unittest.main()
