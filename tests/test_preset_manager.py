import os
import sys
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.preset_manager import delete_preset, list_presets, upsert_preset


class TestPresetManager(unittest.TestCase):
    def test_upsert_and_delete_preset(self):
        name = "test-custom-preset"
        values = {
            "steps": 123,
            "min_side": 300,
            "edge_sharpen_strength": 0.3,
        }
        saved = upsert_preset(name, values, description="test")
        self.assertTrue(saved.get("saved"))

        payload = list_presets()
        presets = payload.get("presets", {})
        self.assertIn(name, presets)
        self.assertEqual(presets[name]["values"]["steps"], 123)

        removed = delete_preset(name)
        self.assertTrue(removed.get("removed"))


if __name__ == "__main__":
    unittest.main()
