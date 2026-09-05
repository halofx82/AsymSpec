"""CPU checks for the checkpoint-free compressed SLM state rebinding."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


def sharing_helpers():
    path = (Path(__file__).resolve().parents[1]
            / "vllm_specsteer/vllm_0_28/specsteer_model.py")
    tree = ast.parse(path.read_text())
    wanted = {"_module_and_leaf", "_share_model_state"}
    nodes = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in wanted]
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *nodes], type_ignores=[])
    ns = {"nn": nn, "torch": torch,
          "logger": SimpleNamespace(info=lambda *args, **kwargs: None)}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns["_share_model_state"]


class TinyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embed = nn.Embedding(8, 4, device=device)
        self.head = nn.Linear(4, 8, bias=False, device=device)
        self.head.weight = self.embed.weight
        self.attn = nn.Linear(4, 4, bias=False, device=device)
        self.register_buffer("scale", torch.ones(1, device=device))

    def forward(self, token_ids):
        return self.head(self.attn(self.embed(token_ids))) * self.scale


class SharedStateTest(unittest.TestCase):
    def test_rebinds_alias_graph_without_sharing_modules(self):
        share = sharing_helpers()
        draft = TinyModel("cpu")
        base = TinyModel("meta")
        share(draft, base)

        draft_params = dict(draft.named_parameters(remove_duplicate=False))
        base_params = dict(base.named_parameters(remove_duplicate=False))
        self.assertEqual(draft_params.keys(), base_params.keys())
        for name in draft_params:
            self.assertIs(base_params[name], draft_params[name])
            self.assertFalse(base_params[name].is_meta)
        self.assertIs(base.embed.weight, base.head.weight)
        self.assertIs(base.scale, draft.scale)
        self.assertIsNot(base.attn, draft.attn)
        token_ids = torch.tensor([1, 3, 5])
        self.assertTrue(torch.equal(base(token_ids), draft(token_ids)))


if __name__ == "__main__":
    unittest.main()
