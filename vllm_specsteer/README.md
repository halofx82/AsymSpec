# AsymSpec vLLM integration

This directory contains the implementation used by the AsymSpec experiments.
The internal `specsteer` module name predates the final paper title and is
retained to avoid a risky mass rename of the vLLM integration surface.

## Patch set

`vllm_0_28/` is the single release patch set for vLLM 0.28.0. It supports
both text and multimodal drafters. Use `scripts/deploy_specsteer.py`; do not
copy files manually.

```bash
python scripts/deploy_specsteer.py --check
python scripts/deploy_specsteer.py --apply
```

The helper locates the active vLLM installation dynamically, saves original
files under `.backups/`, and supports `--revert`.

The eight-file 0.28 payload adds the proposer and sampler and rebases six
upstream files: speculative configuration, engine configuration, the V1 GPU
runner, cache grouping, cache coordination, and cache management. Scheduler
allocation no longer depends on worker-side monkey patches. Multimodal
exceptions are confined to the AsymSpec proposer subclass.

## Per-step computation

AsymSpec evaluates:

1. the drafter on the full context to propose tokens and obtain `aug_logits`;
2. the same drafter on the compressed context to obtain `base_logits`;
3. the verifier on the compressed context to obtain `target_logits`.

The compressed-context drafter pass maintains a separate KV cache, reducing
its incremental cost to O(K). The sampler then applies CDA and delta fusion:

```text
gamma_eff = gamma * exp(-JSD(p_aug || p_base))
delta     = log p_aug - log p_base
fallback  = argmax(log p_target + beta * delta)
```

## Asymmetric context capacity

Set `specsteer_main_max_model_len` in `speculative_config` (or pass
`--specsteer-main-max-model-len` to `bench_lb.py`) to allocate the compressed
verifier/base path independently from the full-context drafter. For example,
global `max_model_len=24576` and a main limit of `8192` allocate 24K only to
the full 4B path and 8K to the 32B verifier plus compressed 4B path. The limit
includes prompt tokens, generated tokens, and speculative lookahead.

Omitting the option preserves the original symmetric cache allocation. The
asymmetric path intentionally requires synchronous scheduling, uniform KV
precision, no KV connector/offload, no KV-cache events, and no prefix caching.
It uses independent BlockPools whose numeric block IDs are local to each cache
group; attention block tables are already interpreted relative to their group.

The two 4B views remain distinct module/attention trees, but the compressed
view is structurally initialized on the meta device and shares every Parameter
and registered buffer with the checkpoint-loaded full-context view. Startup
therefore reads and materializes the 4B checkpoint once.

See [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for the KV-cache implementation
and equivalence argument.

## Safety checks

The implementation fails fast when the compressed-context drafter logits are
missing or shape-incompatible. A silent `base_logits = aug_logits` fallback
would make the context delta zero and invalidate the method.

The patches modify installed vLLM source files. Always run inside a dedicated
environment and keep the generated `.backups/` directory until validation is
complete.
