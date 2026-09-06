"""Run explicitly on CUDA: python tests/gpu_sampler_check.py."""
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # isolate compilation caches before importing the runtime
import torch
from vllm.v1.sample.specsteer_sampler import specsteer_greedy_sample


def reference(drafts, counts, target, aug, base, bonus, method, source, gamma):
    if method == 'strict_target':
        out = torch.full((len(counts), max(counts)+1), -1, dtype=torch.int32)
        top1 = target.argmax(-1)
        start = 0
        for req, count in enumerate(counts):
            for pos in range(count):
                j = start + pos
                out[req, pos] = int(drafts[j]) if drafts[j] == top1[j] else top1[j]
                if drafts[j] != top1[j]:
                    break
            else:
                out[req, count] = bonus[req]
            start += count
        return out
    t, a, b = [torch.log_softmax(x.double().cpu(), -1) for x in (target, aug, base)]
    delta = {'ours': a-b, 'raw_aug': a, 'scd': t-b}[source]
    fused = (t + delta).argmax(-1)
    out = torch.full((len(counts), max(counts)+1), -1, dtype=torch.int32)
    start = 0
    for req, count in enumerate(counts):
        for pos in range(count):
            j = start + pos
            pa, pb = a[j].exp(), b[j].exp()
            kl = (pa * (a[j]-b[j])).sum().clamp_min(0)
            mid = ((pa+pb)/2).log()
            jsd = ((pa*(a[j]-mid)).sum() + (pb*(b[j]-mid)).sum()).clamp_min(0)/2
            divergence = {'gamma_rule': 0., 'cma': kl, 'jsd': jsd,
                          'jsd_pos': jsd/(pos+1), 'cma_vnorm': kl/math.log(t.shape[-1]),
                          'cma_hbase': kl/(-(pb*b[j]).sum()).clamp_min(1e-4)}[method]
            token = int(drafts[j])
            if t[j, token].exp() > gamma * math.exp(-float(divergence)) * pb[token]:
                out[req, pos] = token
            else:
                out[req, pos] = fused[j]
                break
        else:
            out[req, count] = bonus[req]
        start += count
    return out


def main():
    assert torch.cuda.is_available(), 'CUDA is required; this is not a static check'
    generator = torch.Generator().manual_seed(260826004)
    checked = 0
    for counts in ([1, 1], [2, 2], [4, 2, 0]):
        n, k = sum(counts), max(counts)
        target, aug, base = [torch.randn(n, 17, generator=generator).cuda() for _ in range(3)]
        drafts = aug.argmax(-1)
        cu = torch.tensor(counts, device='cuda', dtype=torch.int32).cumsum(0).int()
        for method in ('gamma_rule', 'cma', 'jsd', 'jsd_pos', 'cma_vnorm', 'cma_hbase',
                       'strict_target'):
            for source in ('ours', 'raw_aug', 'scd'):
                for gamma in (0., .5, 100.):
                    for bonus_value in (3, -1):
                        bonus = torch.full((len(counts), 1), bonus_value, device='cuda', dtype=torch.int32)
                        with patch.dict(os.environ, {'ASYMSPEC_METHOD': method,
                            'ASYMSPEC_DELTA_SRC': source, 'CMA_LAMBDA': '1', 'JSD_LAMBDA': '1'}):
                            actual = specsteer_greedy_sample(drafts, counts, k, cu,
                                target, aug, base, bonus, beta=1., gamma=gamma)
                        expected = reference(drafts.cpu(), counts, target, aug, base,
                                             bonus.cpu().flatten(), method, source, gamma)
                        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                        checked += 1
    # Equality rejects: p_target == gamma * p_base, including identical views.
    same = torch.zeros((1, 4), device='cuda')
    with patch.dict(os.environ, {'ASYMSPEC_METHOD': 'jsd', 'ASYMSPEC_DELTA_SRC': 'ours'}):
        out = specsteer_greedy_sample(torch.tensor([1], device='cuda'), [1], 1,
            torch.tensor([1], dtype=torch.int32, device='cuda'), same, same, same,
            torch.tensor([[2]], dtype=torch.int32, device='cuda'), gamma=1.)
        assert out.cpu().tolist() == [[0, -1]]
    # strict_target must be completely independent of beta/gamma and of the
    # auxiliary 4B views. This invokes the Triton kernel, unlike the CPU unit
    # reference tests above.
    target = torch.tensor([[0., 9., 1.], [2., 0., 7.]], device='cuda')
    drafts = torch.tensor([1, 1], device='cuda', dtype=torch.int32)
    cu = torch.tensor([2], device='cuda', dtype=torch.int32)
    bonus = torch.tensor([[2]], device='cuda', dtype=torch.int32)
    expected = [[1, 2, -1]]
    for beta, gamma, aux in ((-99., -10., torch.full_like(target, -1e4)),
                             (99., 1e6, torch.full_like(target, 1e4))):
        with patch.dict(os.environ, {'ASYMSPEC_METHOD': 'strict_target',
                                     'ASYMSPEC_DELTA_SRC': 'scd'}):
            out = specsteer_greedy_sample(drafts, [2], 2, cu, target,
                aux, -aux, bonus, beta=beta, gamma=gamma)
        assert out.cpu().tolist() == expected
    print(f'PASS: {checked} sampler reference cases, strict-threshold equality, '
          'and strict-target isolation')


if __name__ == '__main__':
    main()
