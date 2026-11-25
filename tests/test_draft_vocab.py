"""CPU-only launch policy tests, no model weights or CUDA needed."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from draft_vocab import configure

class VocabularyPolicy(unittest.TestCase):
    def test_unspecified_preserves_custom_values(self):
        env = {"QE_DRAFT_VOCAB": "123456", "QE_DF_HEAD": "full"}
        before = dict(env)
        configure(None, "qwen38-27b", env)
        self.assertEqual(env, before)

    def test_dense_round_trip_overrides_stale_settings(self):
        env = {"UNRELATED": "keep", "QE_DRAFT_VOCAB": "123456"}
        configure("full", "qwen38-27b", env)
        self.assertEqual((env["QE_DRAFT_VOCAB"], env["QE_MTP_VOCAB"], env["QE_DF_HEAD"]), ("0", "0", "full"))
        configure("restricted", "qwen38-27b", env)
        self.assertEqual((env["QE_DRAFT_VOCAB"], env["QE_MTP_VOCAB"], env["QE_DF_HEAD"]), ("98304", "65536", ""))
        self.assertEqual(env["UNRELATED"], "keep")

    def test_flash_both_modes(self):
        env = {}
        configure("restricted", "qwen38-flash-next", env)
        self.assertEqual(env, {"FLASH_DRAFT_V": "65536"})
        configure("full", "qwen38-flash-next", env)
        self.assertEqual(env, {"FLASH_DRAFT_V": "0"})

    def test_invalid_selection_has_no_side_effects(self):
        for mode, family in [("bad", "qwen38-27b"), ("full", "glm53-flash")]:
            env = {"UNRELATED": "keep"}
            with self.assertRaises(ValueError):
                configure(mode, family, env)
            self.assertEqual(env, {"UNRELATED": "keep"})

class LaunchFlag(unittest.TestCase):
    def test_actual_parser_applies_selection_before_native_load(self):
        import app
        import json
        import os
        import tempfile
        from unittest.mock import patch
        for model_type, family, key, restricted in [
            ("qwen3_5", "qwen38-27b", "QE_DRAFT_VOCAB", "98304"),
            ("qwen4_exp", "qwen38-flash-next", "FLASH_DRAFT_V", "65536"),
        ]:
            with tempfile.TemporaryDirectory() as model:
                Path(model, "config.json").write_text(json.dumps({"model_type": model_type}))
                for mode, expected in [("full", "0"), ("restricted", restricted)]:
                    with patch.dict(os.environ, {key: "77", "QE_DF_HEAD": "full"}):
                        args = app.parse_args(["--model", model, "--draft-vocab", mode])
                        self.assertEqual(args.profile.family, family)
                        self.assertEqual(os.environ[key], expected)
                        self.assertFalse(any("draft_vocab" in str(x) for x in args.ignored_start))
                        if family == "qwen38-27b" and mode == "restricted":
                            self.assertEqual(os.environ["QE_DF_HEAD"], "")

if __name__ == "__main__":
    unittest.main()
