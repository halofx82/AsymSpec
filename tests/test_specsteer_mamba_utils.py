"""CPU-only invariants for verifier-only Mamba group selection."""
from types import SimpleNamespace
import unittest

import torch

from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.specsteer_mamba_utils import _validated_groups


def _group(spec):
    return SimpleNamespace(kv_cache_spec=spec)


class TestSpecsteerMambaGroups(unittest.TestCase):
    def setUp(self):
        self.spec = MambaSpec(16, ((4,),), (torch.float16,))
        self.config = SimpleNamespace(kv_cache_groups=[
            _group(object()), _group(self.spec), _group(self.spec),
        ])

    def test_explicit_verifier_groups_are_preserved(self):
        gids, spec = _validated_groups(self.config, [1, 2])
        self.assertEqual(gids, [1, 2])
        self.assertEqual(spec, self.spec)

    def test_auxiliary_or_empty_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            _validated_groups(self.config, [])
        with self.assertRaises(ValueError):
            _validated_groups(self.config, [0])
        with self.assertRaises(ValueError):
            _validated_groups(self.config, [1, 1])

