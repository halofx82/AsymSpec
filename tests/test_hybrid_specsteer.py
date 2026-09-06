import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "vllm_specsteer" / "vllm_0_28" / "hybrid_specsteer.py"
SPEC = importlib.util.spec_from_file_location("hybrid_specsteer_test", MODULE)
hybrid = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(hybrid)


class HybridSpecSteerTests(unittest.TestCase):
    def test_auxiliary_state_layer_detection(self):
        self.assertTrue(hybrid.is_auxiliary_state_layer(
            "draft_model.model.layers.0.linear_attn"))
        self.assertTrue(hybrid.is_auxiliary_state_layer(
            "specsteer_base.model.layers.0.linear_attn"))
        self.assertFalse(hybrid.is_auxiliary_state_layer(
            "model.layers.0.linear_attn"))

    def test_capability_detection(self):
        self.assertTrue(hybrid.is_hybrid_model_config(type("C", (), {"is_hybrid": True})()))
        self.assertFalse(hybrid.is_hybrid_model_config(type("C", (), {})()))

    def test_split_hybrid_layer_names(self):
        full, gdn = hybrid.split_hybrid_layer_names({
            "draft_model.model.layers.0.linear_attn",
            "draft_model.model.layers.3.self_attn",
        })
        self.assertEqual(len(full), 1)
        self.assertEqual(len(gdn), 1)

    def test_requires_independent_trees(self):
        draft = {
            "draft_model.model.layers.0.linear_attn",
            "draft_model.model.layers.3.self_attn",
        }
        base = {
            "specsteer_base.model.layers.0.linear_attn",
            "specsteer_base.model.layers.3.self_attn",
        }
        hybrid.require_distinct_runtime_layer_sets(draft, base)
        with self.assertRaises(RuntimeError):
            hybrid.require_distinct_runtime_layer_sets(draft, draft)

    def test_text_mrope_shape(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        positions = torch.arange(4)
        actual = hybrid.text_mrope_positions(positions)
        self.assertEqual(tuple(actual.shape), (3, 4))
        self.assertTrue(torch.equal(actual[0], positions))
        self.assertTrue(torch.equal(actual[1], positions))


if __name__ == "__main__":
    unittest.main()
