"""Test the deployed coordinator's actual accounting without CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Manager:
    def get_num_blocks_to_allocate(self, *args, **kwargs):
        self.predicted = args[1]
        return (args[1] + 15) // 16

    def allocate_new_blocks(self, *args):
        self.allocated = args[1]
        return [args[1]]


class CrossAttentionManager(Manager):
    pass


class NoPrefix:
    def __init__(self, config):
        self.kv_cache_config = config
        self.single_type_managers = [Manager(), Manager(), CrossAttentionManager()]

    def free(self, rid):
        self.freed = rid


def coordinator_class():
    path = Path(__file__).resolve().parents[1] / 'vllm_specsteer/vllm_0_28/kv_cache_coordinator.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'SpecSteerKVCacheCoordinator')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ns = {'KVCacheCoordinatorNoPrefixCache': NoPrefix,
          'CrossAttentionManager': CrossAttentionManager}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns['SpecSteerKVCacheCoordinator']


class CacheAccountingTest(unittest.TestCase):
    def test_prediction_matches_allocation_and_cleanup(self):
        cls = coordinator_class()
        for offset in (0, 17001, -8):
            with self.subTest(offset=offset):
                cfg = SimpleNamespace(kv_cache_groups=[
                    SimpleNamespace(layer_names=['model.layer', 'specsteer_base.layer']),
                    SimpleNamespace(layer_names=['draft_model.layer']),
                    SimpleNamespace(layer_names=['encoder.layer'])])
                coord = cls(cfg)
                coord.aug_offsets['a'] = offset
                n = coord.get_num_blocks_to_allocate('a', 600, ([], [], []), 20, 0, 0, 598)
                coord.allocate_new_blocks('a', 600, 598, 20)
                expected = [600, 600 + offset, 20]
                self.assertEqual([m.predicted for m in coord.single_type_managers], expected)
                self.assertEqual([m.allocated for m in coord.single_type_managers], expected)
                self.assertEqual(n, sum((x + 15) // 16 for x in expected))
                coord.free('a')
                self.assertNotIn('a', coord.aug_offsets)
                self.assertEqual(coord._view_tokens(1, 'a', 600), 600)


if __name__ == '__main__':
    unittest.main()
