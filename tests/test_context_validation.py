"""Request-level checks for AsymSpec's two independently sized contexts."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class ValidationError(ValueError):
    pass


class FakeInputProcessor:
    def process_inputs(self):
        return self.request


def installer():
    path = (Path(__file__).resolve().parents[1]
            / "vllm_specsteer/vllm_0_28/vllm.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_install_specsteer_context_validation")
    module = ast.Module(body=[node], type_ignores=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns[node.name]


class ContextValidationTest(unittest.TestCase):
    def setUp(self):
        FakeInputProcessor.process_inputs = lambda self: self.request
        exceptions = ModuleType("vllm.exceptions")
        exceptions.VLLMValidationError = ValidationError
        processor_mod = ModuleType("vllm.v1.engine.input_processor")
        processor_mod.InputProcessor = FakeInputProcessor
        self.modules = {
            "vllm": ModuleType("vllm"),
            "vllm.exceptions": exceptions,
            "vllm.v1": ModuleType("vllm.v1"),
            "vllm.v1.engine": ModuleType("vllm.v1.engine"),
            "vllm.v1.engine.input_processor": processor_mod,
        }

    def make_processor(self, main_len, full_len, max_tokens):
        proc = FakeInputProcessor()
        proc.speculative_config = SimpleNamespace(
            method="specsteer", specsteer_main_max_model_len=10)
        proc.model_config = SimpleNamespace(max_model_len=20)
        proc.request = SimpleNamespace(
            prompt_token_ids=[1] * main_len,
            prompt_embeds=None,
            sampling_params=SimpleNamespace(
                max_tokens=max_tokens,
                extra_args={"specsteer_aug_prompt_ids": [1] * full_len},
            ),
        )
        return proc

    def test_accepts_both_views_within_limits(self):
        with patch.dict(sys.modules, self.modules):
            installer()()
            request = self.make_processor(6, 15, 4).process_inputs()
        self.assertEqual(len(request.prompt_token_ids), 6)

    def test_rejects_compressed_view(self):
        with patch.dict(sys.modules, self.modules):
            installer()()
            with self.assertRaisesRegex(ValidationError, "compressed context"):
                self.make_processor(7, 10, 4).process_inputs()

    def test_rejects_full_view(self):
        with patch.dict(sys.modules, self.modules):
            installer()()
            with self.assertRaisesRegex(ValidationError, "full context"):
                self.make_processor(5, 18, 4).process_inputs()


if __name__ == "__main__":
    unittest.main()
