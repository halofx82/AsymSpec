# AsymSpec: Context-Asymmetric Speculative Decoding

This is a derived work based on the original **AsymSpec:
Context-Asymmetric Speculative Decoding for Agentic LLMs** (EMNLP 2026)
reference repository in the sibling [`../AsymSpec`](../AsymSpec) checkout.
It ports that implementation from vLLM 0.19.0 to vLLM 0.28.0; it is not the
original or official release.
See [PORTING.md](PORTING.md) for provenance, validation status, and the
4×RTX 3090 acceptance command. The original 0.19.0 checkout is unchanged.

Paper: [arXiv:2608.26004](https://arxiv.org/abs/2608.26004)

AsymSpec lets a lightweight drafter read the full context while the large
verifier operates on a compressed view. A same-model cross-context signal,
`delta = logits(full) - logits(compressed)`, steers the verifier after a
rejection. Context-Divergence Acceptance (CDA) uses
`gamma_eff = gamma * exp(-JSD)` to relax acceptance when the two context
views disagree.

## Release status

This repository is a clean release candidate prepared from the authors'
collaborative research code. It intentionally excludes private Git history,
raw outputs, cached web/tool responses, model weights, and datasets. See
[`RELEASE_PROVENANCE.md`](RELEASE_PROVENANCE.md) for attribution and source
provenance.

## Repository layout

```text
vllm_specsteer/              AsymSpec patches for vLLM 0.28.0
scripts/deploy_specsteer.py  Patch deployment and rollback helper
scripts/bench_*.py           Paper benchmark harnesses
scripts/asym_smolagents/     GAIA and SimpleQA agentic harnesses
experiments/                 Compression and portability utilities
experiments/cross_family/    Qwen--Llama portability implementation
configs/paper.yaml           Camera-ready default configurations
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

## Paper defaults

The camera-ready defaults are greedy decoding, `beta=1.0`, `gamma=0.5`, and
CDA via `--asym_method jsd`. Text and agentic benchmarks use `K=2`;
MathVista is the cross-modal exception and uses `K=4`. Exact settings are in
[`configs/paper.yaml`](configs/paper.yaml).

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

The agentic entry points are
`scripts/asym_smolagents/run_gaia_web.py` and
`scripts/asym_smolagents/run_simpleqa.py`. Their defaults match
the paper: CDA via `jsd`, LLMLingua-2 ratio 0.3, two recent turns retained,
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
`/reset` to clear history and `/quit` to exit. This is a terminal interface:
stock vLLM's OpenAI server cannot carry the required second prompt stream.

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

No model weights, benchmark datasets, generated responses, API keys, or
cached tool results are distributed in this repository. Users must accept
and follow each upstream model and dataset license. Do not commit `.env` or
`conf.yaml`.

## Citation

Please cite the paper using [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
