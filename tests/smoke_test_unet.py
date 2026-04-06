import os
import sys
import tempfile
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
		noise_pred, mask_pred = model(x, t)

		self.assertEqual(noise_pred.shape, (batch_size, 3, h, w))
		self.assertEqual(mask_pred.shape, (batch_size, 1, h, w))

	def test_build_model_with_checkpoint_payload(self):
		model = SimpleUNet(cond_mode="concat")
		payload = {
			"model_state": model.state_dict(),
			"optimizer_state": None,
			"epoch": 0,
			"avg_loss": 0.0,
			"config": {"cond_mode": "concat"},
		}

		with tempfile.TemporaryDirectory() as tmpdir:
			ckpt_path = os.path.join(tmpdir, "ckpt.pth")
			torch.save(payload, ckpt_path)
			loaded = build_model(ckpt_path, device="cpu")

		self.assertIsInstance(loaded, SimpleUNet)
		self.assertEqual(getattr(loaded, "cond_mode", None), "concat")


if __name__ == "__main__":
	unittest.main()
