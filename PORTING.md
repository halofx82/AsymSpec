# vLLM 0.28.0 port

Status: asymmetric KV allocation and single-load 4B sharing are implemented
and validated on four RTX 3090 GPUs.

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
- The full-context drafter is checkpoint-loaded once. The compressed view is
  constructed through vLLM's lower-level `initialize_model()` path on the meta
  device, then every Parameter and registered buffer is rebound by qualified
  name. Hard startup checks require identical layouts, object-identical state,
  no meta/uninitialized tensors, and distinct attention registrations.
- With `specsteer_main_max_model_len`, gid0 (32B verifier + compressed 4B) and
  gid1 (full 4B drafter) receive separate physical tensor sizes and separate
  BlockPools. Block IDs are group-local; worker block tables and attention
  metadata already select a group before interpreting them. Prefix caching,
  cache connectors, KV-cache events, deferred asynchronous frees,
  mixed-precision zeroing, and Mamba are deliberately outside this isolated
  path because their APIs flatten block IDs across groups. Omitting the option
  keeps vLLM's shared pool.
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

The optimized zero-offload run used
`--specsteer-main-max-model-len 8192`. It completed the same three requests
and produced byte-identical token arrays and speculative counts to that
pre-optimization 0.28 result: 627 tokens, AR 0.86910, MAL 2.73820, and
per-position acceptance 0.91416 / 0.82403. Throughput was 15.20 tokens/s
versus 8.54 tokens/s in the stored 2 GiB-offload run. Physical KV allocation
was 1.63 GiB/rank: gid0 had 513 blocks (8,192 compressed tokens) and gid1 had
1,537 blocks (24,576 full-context tokens).

A capacity test then configured the native Qwen3 maximum of 40,960 with the
compressed limit fixed at 8,192. The engine allocated 2.19 GiB/rank
(gid0=513 blocks, gid1=2,561 blocks) and completed a real request containing
40,928 full-context prompt tokens plus 16 generated tokens. No YaRN or other
RoPE scaling was used. `scripts/check_context_capacity.py` reproduces this
test; 40,960 is the maximum tested because it is the checkpoint's native
context limit.

The historical 0.19 artifact remains a cross-runtime comparison, not an exact
semantic oracle for the 0.28 engine. A paired short trace showed that 0.19 and
0.28 agree on the first two output tokens and initial model top-1 signals, then
diverge; the 0.19 sample text becomes repetitive/malformed while 0.28 reaches
EOS coherently. The optimization does not attempt to recreate that behavior:
its strict regression oracle is the stored pre-optimization 0.28 token stream,
which it preserves exactly. Consequently the older 0.19 figures (AR 0.96994,
MAL 2.93989, 3,072 tokens) should not be reported as achieved by this port.

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
