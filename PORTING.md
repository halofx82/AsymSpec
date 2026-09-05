# vLLM 0.28.0 port

Status: implementation and GPU validation in progress. Do not interpret
source deployment or CPU tests as a runtime-parity claim.

## Provenance and environment

The baseline is AsymSpec commit `15582b2` plus the local modifications to
`requirements.txt`, `scripts/bench_lb.py`, and `scripts/gen_lb_summaries.py`.
These include TP selection, CPU offload, and removal of a worker-local Path B
toggle. All five installed 0.19.0 payload files matched the original checkout
at the start of the port. Original source, environment, and outputs were not
modified.

The target is upstream vLLM tag `v0.28.0`, commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. The deployment manifest records
upstream file hashes. `requirements-0.28.lock.txt` records the resolved
environment; install it in a fresh Python 3.12 virtual environment. The
local environment uses PyTorch 2.13.0+cu130 and Triton 3.7.1.

The acceptance environment reused the original benchmark inputs through a
temporary local symlink. Dataset files are not versioned; set
`ASYMSPEC_DATA_DIR` or populate this checkout's `data` directory. Compilation
caches and output paths are isolated under this checkout.

## Integration decisions

- Public method names, parameters, ablations, and vocabulary-map artifacts
  retain their original meanings. The sampler is unchanged.
- The proposer subclasses the new shared draft-model implementation. Its
  context swap binds the current proposal signature before substituting
  inputs. Multimodal support stays in subclass overrides.
- Full-context and compressed-context cache groups are separated by registered
  layer names. Allocation prediction and allocation apply the same full-context
  offset in the scheduler process; workers reconstruct image/context metadata
  from transported request state. Completion and preemption invalidate state.
- The two drafter views retain separate module and attention objects but alias
  their identical frozen parameter storage. This preserves independent KV
  caches while recovering one drafter copy per TP rank for the 24 GiB budget.
- AsymSpec uses synchronous scheduling, no prefix caching, full draft logits,
  and its existing optional heterogeneous-vocabulary extension. Upstream
  automatic vocabulary mapping is disabled for AsymSpec.
- Offloading retains the upstream process-wide budget and lifecycle across
  all three model loads. The acceptance configuration uses equal TP=4 for
  verifier and drafter and eager execution.
- LongBench caps each scheduled chunk at 16,384 tokens. Path B evaluates the
  two contexts independently, so the old combined-forward 24K batch is no
  longer required; the total sequence and generation limits remain 24,576
  and 1,024.
- The exact 4x24 GiB acceptance configuration caps KV allocation at 5 GiB per
  rank. The three cache views require about 4.69 GiB at 24K; the remainder of
  the automatic 5.56 GiB profile is retained for augmented-prefill activations.
- There is no startup deletion of shared compilation caches.

## Hardware acceptance

Activate this checkout's `.venv`, deploy with
`python scripts/deploy_specsteer.py --apply`, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/bench_lb.py \
  --mode specsteer --slm 4B --tp 4 --K 2 --beta 1.0 --gamma 0.5 \
  --asym_method jsd --main_context summary --max_model_len 24576 \
  --gpu_memory_utilization 0.97 --cpu_offload_gb 2 --max_new 1024 \
  --n 3 --enforce_eager --cell asymspec-24k-offload-smoke \
  --out outputs/longbench/4x3090/k2/asymspec-24k-offload-smoke_metrics.json \
  --responses outputs/longbench/4x3090/k2/asymspec-24k-offload-smoke_responses.jsonl
python scripts/check_offload_smoke.py \
  --baseline ../AsymSpec/outputs/longbench/4x3090/k2/asymspec-24k-offload-smoke_metrics.json
```

The artifact checker requires all three tasks, unchanged configuration,
complete responses, and speculative statistics. Also inspect logs for active
Path B/CDA, worker failures, or fallbacks. The original baseline generated
3,072 tokens at 11.5766 tokens/s with 0.96994 draft acceptance. These are
comparison measurements, not required speed or acceptance thresholds.

The acceptance run on 2026-09-05 completed all three requests with prompt
lengths 565, 560, and 564, full-context lengths 11,750, 7,371, and 16,466,
and output lengths 165, 233, and 229. It produced 627 tokens at 8.52 tokens/s
with 0.8691 draft acceptance and no allocation failure. The earlier EOS and
token streams differ from the 0.19 baseline after two common-prefix tokens;
exact output identity is not an acceptance requirement across vLLM runtimes.

`24576` is the total sequence limit, including generation. The harness reserves
`max_new + 16` tokens when constructing inputs. The three actual prompt lengths
must be recorded; a configured limit alone does not demonstrate a full-window
prefill. Earlier EOS is valid; reducing the 1,024-token budget is not.

## Supporting checks

```bash
bash scripts/check_release.sh
python tests/gpu_sampler_check.py
```

CPU tests cover original release invariants, deployment/revert and modified-file
protection, heterogeneous patch anchors, and scheduler allocation prediction.
The GPU sampler check compares 324 cases across all acceptance/delta modes,
ragged draft counts, and bonus modes against an independent reference, plus
strict-threshold equality. Additional model-level checks must be reported
separately from this sampler check.
