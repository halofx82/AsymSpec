"""Dependency-light contract tests for strict target verification."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SAMPLER = ROOT / "vllm_specsteer" / "vllm_0_28" / "specsteer_sampler.py"
_spec = importlib.util.spec_from_file_location("payload_specsteer_sampler", SAMPLER)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
strict_target_reference = _module.strict_target_reference


def logits_with_top1(tokens: list[int], vocab_size: int = 12) -> torch.Tensor:
    logits = torch.full((len(tokens), vocab_size), -10.0)
    for row, token in enumerate(tokens):
        logits[row, token] = 10.0
    return logits


class StrictTargetSamplerTest(unittest.TestCase):
    def run_reference(self, drafts, counts, top1, bonus, max_spec_len=2):
        return strict_target_reference(
            torch.tensor(drafts, dtype=torch.int32), counts, max_spec_len,
            logits_with_top1(top1), torch.tensor(bonus, dtype=torch.int32),
        ).tolist()

    def test_all_drafts_match_emits_target_bonus(self):
        self.assertEqual(self.run_reference([5, 7], [2], [5, 7], [9]),
                         [[5, 7, 9]])

    def test_first_mismatch_stops_without_bonus(self):
        self.assertEqual(self.run_reference([5, 7], [2], [6, 7], [9]),
                         [[6, -1, -1]])

    def test_second_mismatch_stops_without_bonus(self):
        self.assertEqual(self.run_reference([5, 7], [2], [5, 8], [9]),
                         [[5, 8, -1]])

    def test_ragged_batch_uses_each_request_bonus(self):
        self.assertEqual(
            self.run_reference([5, 7, 4], [2, 1], [5, 7, 6], [9, 10]),
            [[5, 7, 9], [6, -1, -1]],
        )

    def test_target_logits_are_the_only_decision_input(self):
        target = logits_with_top1([5, 8])
        expected = strict_target_reference(
            torch.tensor([5, 7], dtype=torch.int32), [2], 2, target,
            torch.tensor([9], dtype=torch.int32),
        )
        # These are deliberately extreme stand-ins for changing beta/gamma,
        # augmented logits, and base logits. The strict reference has no path
        # by which they can affect a target-only decision.
        aug = torch.full_like(target, 1e20)
        base = torch.full_like(target, -1e20)
        self.assertEqual(aug.argmax(dim=-1).tolist(), [0, 0])
        self.assertEqual(base.argmax(dim=-1).tolist(), [0, 0])
        actual = strict_target_reference(
            torch.tensor([5, 7], dtype=torch.int32), [2], 2, target,
            torch.tensor([9], dtype=torch.int32),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
