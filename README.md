# AsymSpec: Context-Asymmetric Speculative Decoding

This is a derived work based on the original **AsymSpec:
Context-Asymmetric Speculative Decoding for Agentic LLMs** (EMNLP 2026)
reference implementation of [AsymSpec](https://github.com/USTC-StarTeam/AsymSpec.git).
It ports that implementation from vLLM 0.19.0 to vLLM 0.28.0; it is not the
original or official release.
See [PORTING.md](PORTING.md) for provenance, and validation status.

Associated paper: [arXiv:2608.26004](https://arxiv.org/abs/2608.26004)

AsymSpec lets a lightweight drafter read the full context while the large
verifier operates on a compressed view. A same-model cross-context signal,
`delta = logits(full) - logits(compressed)`, steers the verifier after a
rejection. Context-Divergence Acceptance (CDA) uses
`gamma_eff = gamma * exp(-JSD)` to relax acceptance when the two context
views disagree.

## Derived-work status

This separate port is maintained independently of the original authors and
reference repository. It intentionally excludes private Git history, raw
outputs, cached web/tool responses, model weights, and datasets. See
[`RELEASE_PROVENANCE.md`](RELEASE_PROVENANCE.md) for attribution and source
provenance.

## Repository layout

```text
vllm_specsteer/              AsymSpec patches for vLLM 0.28.0
scripts/deploy_specsteer.py  Patch deployment and rollback helper
scripts/bench_*.py           Benchmark harnesses adapted from the reference repo
scripts/asym_smolagents/     GAIA and SimpleQA agentic harnesses
experiments/                 Compression and portability utilities
experiments/cross_family/    Qwen--Llama portability implementation
configs/paper.yaml           Configurations reproduced from the paper setup
```

`specsteer` is retained in a few internal file and class names as a legacy
implementation identifier. Public method names and reported results use
**AsymSpec**.

## Environment

The patches target **vLLM 0.28.0**. Newer versions may change the patched
interfaces; see [`VLLM_COMPATIBILITY.md`](VLLM_COMPATIBILITY.md) for the
source-level migration assessment.

Use an isolated research environment with trusted inputs and weights.
The port has not been audited for public serving. See [`SECURITY.md`](SECURITY.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the example configuration files only when a benchmark needs a gated
dataset or an LLM judge:

```bash
cp .env.example .env
cp conf.example.yaml conf.yaml
```

Download model and dataset assets:

```bash
bash scripts/download_models.sh
bash scripts/download_datasets.sh
```

Generate the offline compressed contexts used by LongBench and
MultiChallenge:

```bash
python scripts/gen_lb_summaries.py
python scripts/gen_mc_summaries.py
```

Check that the local benchmark inputs are ready before launching a GPU job:

```bash
python scripts/check_assets.py --benchmark all
```

MathVista additionally expects the official Bard captions and EasyOCR output
at the paths configured by `MV_CAPTIONS` and `MV_OCRS` in `paths.py`.

## Deploy the vLLM integration

The deployment helper backs up each overwritten vLLM file under `.backups/`.

```bash
python scripts/deploy_specsteer.py --check
python scripts/deploy_specsteer.py --apply
python scripts/deploy_specsteer.py --check
```

Restore the original vLLM files with:

```bash
python scripts/deploy_specsteer.py --revert
```

## Original-paper defaults

The original paper's camera-ready defaults are greedy decoding, `beta=1.0`,
`gamma=0.5`, and CDA via `--asym_method jsd`. Text and agentic benchmarks use
`K=2`; MathVista is the cross-modal exception and uses `K=4`. The port's
reproduced settings are in [`configs/paper.yaml`](configs/paper.yaml).

Representative commands:

```bash
# LongBench: Qwen3-4B -> Qwen3-32B
python scripts/bench_lb.py --mode specsteer --slm 4B --K 2 \
  --beta 1.0 --gamma 0.5 --asym_method jsd --main_context summary \
  --cell asymspec --out outputs/longbench/metrics.json \
  --responses outputs/longbench/responses.jsonl

# Asymmetric KV capacity: 24K full context, 8K compressed context, no offload
python scripts/bench_lb.py --mode specsteer --slm 4B --tp 4 --K 2 \
  --max_model_len 24576 --specsteer-main-max-model-len 8192 \
  --cpu_offload_gb 0 --max_new 1024 --enforce_eager \
  --cell asymspec-24k-asymmetric --out outputs/longbench/asymmetric.json \
  --responses outputs/longbench/asymmetric.jsonl

# Exercise the full-context pool with a real near-limit native Qwen3 request
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/check_context_capacity.py \
  --max-model-len 40960 --specsteer-main-max-model-len 8192 --max-new 16

# MultiChallenge
python scripts/bench_multichallenge.py --mode specsteer --slm 4B --K 2 \
  --beta 1.0 --gamma 0.5 --asym_method jsd \
  --main_context summary_last_k --last_k 1 \
  --cell asymspec --out outputs/multichallenge/metrics.json \
  --responses outputs/multichallenge/responses.jsonl

# API-Bank Method A
python scripts/bench_apibank.py --mode specsteer --slm 1.7B --K 2 \
  --beta 1.0 --gamma 0.5 --asym_method jsd \
  --main_compression name_sig --max_new 256 \
  --cell asymspec --out outputs/apibank/metrics.json \
  --responses outputs/apibank/responses.jsonl

# MathVista cross-modal configuration
python scripts/bench_mathvista.py --cfg ss --K 4 --beta 1.0 \
  --gamma 0.5 --asym_method jsd --n 0 --tag paper
```

The adapted agentic entry points are
`scripts/asym_smolagents/run_gaia_web.py` and
`scripts/asym_smolagents/run_simpleqa.py`. Their defaults reproduce the
paper setup: CDA via `jsd`, LLMLingua-2 ratio 0.3, two recent turns retained,
and `K=2`.

## Interactive chat

Start a local terminal REPL (not a benchmark) after deploying the patches:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/chat_specsteer.py \
  --tp 4 --max-model-len 40960 --specsteer-main-max-model-len 8192 \
  --gpu-memory-utilization 0.97 --cpu-offload-gb 0 --enforce-eager
```

`--full-context-file` adds a document only to the full-context drafter;
provide its corresponding concise summary with `--main-context-file` for the
verifier/compressed path. Without files, the full path retains all chat turns,
while the compressed path retains the newest complete turns that fit. Use
`/reset` to clear history and `/quit` to exit.

### OpenAI-compatible server

After deployment, ordinary `vllm serve` transparently derives the full drafter
prompt and a recent-history compressed prompt from a standard Chat Completions
request. The compressed prompt's usage accounting is reported by vLLM as
`usage.prompt_tokens`.

### Hybrid Qwen3.5 / Qwen3.8 models

The text-only hybrid pair uses vLLM's GDN/Mamba state machinery in addition to
ordinary full-attention KV caches. Keep V1 enabled and prefix caching disabled:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export ASYMSPEC_METHOD=jsd
export ASYMSPEC_DELTA_SRC=ours

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3.8-27B \
  --served-model-name qwen38-asymspec \
  --language-model-only --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 40960 --gpu-memory-utilization 0.97 \
  --no-enable-prefix-caching --enforce-eager --generation-config vllm \
  --speculative-config '{
    "method": "specsteer", "model": "Qwen/Qwen3.5-4B",
    "num_speculative_tokens": 2, "draft_tensor_parallel_size": 4,
    "specsteer_beta": 1.0, "specsteer_gamma": 0.5,
    "specsteer_main_max_model_len": 8192
  }'
```

The same language-only setting is propagated to the 4B draft ModelConfig. The
full drafter groups use `max_model_len`; compressed verifier/base groups use
`specsteer_main_max_model_len`. Hybrid prefix caching is rejected explicitly.

#### Verifier strict-target diagnostic

Before interpreting hybrid-pair output quality, validate the 27B verifier with
target-only greedy speculative decoding. It retains 4B proposals for scheduling
but commits a proposal only when it equals the 27B verifier's top-1 token;
otherwise it commits that verifier token and stops the speculative block.

```bash
export ASYMSPEC_METHOD=strict_target
export ASYMSPEC_STRICT_DIAG_LOG="$PWD/outputs/strict-target.jsonl"
```

For a text-chat capture, put the same messages in `messages.json`, then run the
two commands separately (never concurrently):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/capture_strict_target.py \
  --mode target --messages messages.json --out outputs/target-greedy.json \
  --max-tokens 256

CUDA_VISIBLE_DEVICES=0,1,2,3 ASYMSPEC_STRICT_DIAG_LOG="$PWD/outputs/strict-target.jsonl" \
  python scripts/capture_strict_target.py \
  --mode strict-target --messages messages.json --out outputs/strict-target.json \
  --max-tokens 256
```

The capture tool renders the same Qwen chat template and deliberately supplies
the same prompt IDs to strict target's compressed and full paths, isolating
verifier correctness from compression. It writes one JSON object per run:

```json
{"generated_token_ids": [123, 456], "generated_text": "..."}
```

Token IDs—not decoded text—are the oracle:

```bash
python scripts/check_strict_target.py \
  --target outputs/target-greedy.json \
  --asymspec outputs/strict-target.json
```

The command reports the common prefix and first divergent token, and exits
nonzero on any difference. `ASYMSPEC_STRICT_DIAG_LOG` appends one record per
request/speculative block without prompt contents, e.g.

```json
{"method":"strict_target","request_index":0,"num_draft_tokens":2,"num_target_top1_matches":1,"first_mismatch_position":1,"all_drafts_match":false,"bonus_emitted":false,"bonus_token_id":null,"positions":[{"pos":0,"draft":123,"target_top1":123,"match":true,"emitted":123},{"pos":1,"draft":456,"target_top1":789,"match":false,"emitted":789}]}
```

An A-vs-B mismatch is evidence to investigate the verifier/cache path; do not
attribute it to decoded-text differences or to the JSD sampler. Deterministic
execution settings (V1, eager mode, fixed seed, no concurrent batching) should
be held constant. The normal `jsd` mode remains a separate quality comparison.

```bash
export ASYMSPEC_METHOD=jsd
export ASYMSPEC_DELTA_SRC=ours

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-32B \
  --served-model-name qwen3-32b-asymspec --dtype bfloat16 --trust-remote-code \
  --tensor-parallel-size 4 --max-model-len 40960 --gpu-memory-utilization 0.97 \
  --max-num-batched-tokens 16384 --no-enable-prefix-caching --enforce-eager \
  --generation-config vllm --speculative-config '{
    "method": "specsteer", "model": "Qwen/Qwen3-4B",
    "num_speculative_tokens": 2, "draft_tensor_parallel_size": 4,
    "specsteer_beta": 1.0, "specsteer_gamma": 0.5,
    "specsteer_main_max_model_len": 8192,
    "specsteer_context_strategy": "recent"
  }'
```

Use the normal OpenAI endpoint; no `vllm_xargs` or AsymSpec request fields are
needed. `system` and `developer` messages at the start remain pinned, and the
newest complete user/tool interactions that fit are kept for the compressed
view. Multimodal requests are currently rejected only for SpecSteer serving.
The server owns `specsteer_aug_prompt_ids`; clients may not supply that key.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-32b-asymspec",
    "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
    "temperature": 0,
    "max_tokens": 256
  }'
```

Run the dependency-light release checks with:

```bash
bash scripts/check_release.sh
```

## Cross-family portability

Build a vocabulary map, deploy the standard patches, apply the heterogeneous
vocabulary extension, and run the cross-family LongBench harness:

```bash
python experiments/cross_family/build_hetero_map.py \
  --output artifacts/hetero_llama3b_qwen32b.pt
python experiments/cross_family/apply_hetero_patch.py
python experiments/cross_family/bench_lb_crossfamily.py \
  --mode specsteer --hetero \
  --drafter_path meta-llama/Llama-3.2-3B-Instruct \
  --verifier_path Qwen/Qwen3-32B \
  --hetero_map artifacts/hetero_llama3b_qwen32b.pt \
  --K 2 --beta 1.0 --gamma 0.5 --asym_method jsd \
  --cell llama_to_qwen --out outputs/cross_family/metrics.json \
  --responses outputs/cross_family/responses.jsonl
```

## Data and credentials

This derived checkout contains no model weights, benchmark datasets, generated
responses, API keys, or cached tool results. Users must accept and follow each
upstream model and dataset license. Do not commit `.env` or `conf.yaml`.

## Citation

Citation information for the associated paper is in
[`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
