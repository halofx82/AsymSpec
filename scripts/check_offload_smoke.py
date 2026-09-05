#!/usr/bin/env python3
"""Validate the exact 4x3090 acceptance artifacts and compare an optional baseline."""
import argparse
import json
from pathlib import Path


def validate(metrics_path, responses_path):
    metrics = json.loads(metrics_path.read_text())
    records = [json.loads(line) for line in responses_path.read_text().splitlines() if line]
    assert metrics['n_kept'] == len(records) == 3, 'Missing/failed benchmark requests'
    assert set(metrics['out_by_idx']) == {'0', '1', '2'}
    assert all(len(x) > 0 and len(x) <= 1024 for x in metrics['out_by_idx'].values())
    assert all(metrics['per_dataset'][t]['n'] == 1
               for t in ('hotpotqa', '2wikimqa', 'musique'))
    expected = {'mode': 'specsteer', 'K': 2, 'slm': '4B', 'main_context': 'summary',
                'asym_method': 'jsd', 'specsteer_beta': 1.0, 'specsteer_gamma': .5,
                'max_model_len': 24576, 'max_new': 1024, 'enforce_eager': True,
                'tensor_parallel_size': 4, 'cpu_offload_gb': 2.,
                'gpu_memory_utilization': .97}
    for key, value in expected.items():
        assert metrics['config'][key] == value, (key, metrics['config'].get(key), value)
    assert metrics['config']['kv_cache_memory_bytes'] == 5 * 1024**3
    assert metrics['spec_metrics']['num_draft_tokens'] > 0
    assert metrics['total'] == sum(map(len, metrics['out_by_idx'].values()))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path('outputs/longbench/4x3090/k2')
    stem = 'asymspec-24k-offload-smoke'
    parser.add_argument('--metrics', type=Path, default=directory / (stem+'_metrics.json'))
    parser.add_argument('--responses', type=Path, default=directory / (stem+'_responses.jsonl'))
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    metrics = validate(args.metrics, args.responses)
    report = {'complete': True, 'output_lengths': {k: len(v) for k,v in metrics['out_by_idx'].items()},
              'tokens_per_second': metrics['tps'], 'acceptance': metrics['spec_metrics']['draft_acceptance_rate']}
    if args.baseline:
        old = json.loads(args.baseline.read_text())
        comparisons = {}
        for key, tokens in metrics['out_by_idx'].items():
            previous = old['out_by_idx'][key]
            common = next((i for i,(a,b) in enumerate(zip(tokens, previous)) if a != b),
                          min(len(tokens), len(previous)))
            comparisons[key] = {'identical': tokens == previous, 'common_prefix_tokens': common}
        report['baseline_comparison'] = comparisons
        report['baseline_tokens_per_second'] = old['tps']
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
