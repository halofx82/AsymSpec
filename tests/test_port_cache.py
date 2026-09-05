"""Test the deployed coordinator's actual accounting without CUDA imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class Manager:
    def __init__(self, block_pool=None):
        self.block_pool = block_pool

    def get_num_blocks_to_allocate(self, *args, **kwargs):
        self.predicted = args[1]
        return (args[1] + 15) // 16

    def allocate_new_blocks(self, *args):
        self.allocated = args[1]
        return [args[1]]


class CrossAttentionManager(Manager):
    pass


class NoPrefix:
    def __init__(self, config, *args, **kwargs):
        self.kv_cache_config = config
        self.single_type_managers = [Manager(), Manager(), CrossAttentionManager()]
        self.enable_caching = False
        self.scheduler_block_size = kwargs.get('scheduler_block_size', 16)

    def free(self, rid):
        self.freed = rid


def coordinator_class():
    path = Path(__file__).resolve().parents[1] / 'vllm_specsteer/vllm_0_28/kv_cache_coordinator.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'SpecSteerKVCacheCoordinator')
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    class Pool:
        def __init__(self, num_gpu_blocks, **kwargs):
            self.free = num_gpu_blocks - 1

        def get_num_free_blocks(self):
            return self.free

        def get_usage(self):
            return 0.0

    def manager_factory(*, block_pool, **kwargs):
        return Manager(block_pool)

    ns = {'KVCacheCoordinatorNoPrefixCache': NoPrefix,
          'CrossAttentionManager': CrossAttentionManager,
          'BlockPool': Pool,
          'get_manager_for_kv_cache_spec': manager_factory,
          'logger': SimpleNamespace(info=lambda *args, **kwargs: None)}
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

    def test_asymmetric_pools_have_group_local_capacity(self):
        cls = coordinator_class()
        cfg = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=['specsteer_base.layer'],
                                kv_cache_spec=SimpleNamespace()),
                SimpleNamespace(layer_names=['draft_model.layer'],
                                kv_cache_spec=SimpleNamespace()),
            ],
            num_blocks_per_group=[513, 2561],
            max_model_len_per_group=[8192, 40960],
            needs_kv_cache_zeroing=False,
        )
        coord = cls(cfg, 40960, 40960, False, False,
                    dcp_world_size=1, pcp_world_size=1,
                    scheduler_block_size=16, hash_block_size=16)
        self.assertTrue(coord.has_asymmetric_pools)
        self.assertEqual([p.get_num_free_blocks() for p in coord.block_pools],
                         [512, 2560])
        coord.aug_offsets['r'] = 32000
        required = coord.get_num_blocks_to_allocate_per_group(
            'r', 1000, ([], []), 0, 0, 0, 1000)
        self.assertEqual(required, [63, 2063])


if __name__ == '__main__':
    unittest.main()
