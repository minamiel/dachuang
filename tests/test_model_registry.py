import os
import sys
import tempfile
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.model_registry import activate_profile, list_models, model_status, verify_model


class TestModelRegistry(unittest.TestCase):
    def test_list_models_contains_profiles(self):
        payload = list_models()
        profile_ids = [m.get("profile") for m in payload.get("profiles", [])]
        self.assertIn("text-priority", profile_ids)
        self.assertIn("natural-priority", profile_ids)

    def test_verify_model_with_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "tmp_model.pth")
            with open(p, "wb") as f:
                f.write(b"abc")

            info = verify_model("text-priority", model_path=p)
            self.assertTrue(info.get("exists"))
            self.assertEqual(info.get("path"), os.path.abspath(p))
            self.assertIsNotNone(info.get("sha256_actual"))

    def test_activate_profile_and_status(self):
        res = activate_profile("text-priority")
        self.assertEqual(res.get("active_profile"), "text-priority")

        st = model_status()
        self.assertEqual(st.get("active_profile"), "text-priority")


if __name__ == "__main__":
    unittest.main()
