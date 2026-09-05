"""MultiChallenge benchmark harness for AsymSpec and its baselines.

The harness supports configurable draft length and batch size, records the
active integration hashes for traceability, and writes both metrics and
decoded responses for downstream LLM-judge evaluation.

Modes:
  b1_aug             — LLM only on full conversation.
  b1_main            — LLM only on compressed main (--main_context).
  classical_sps_aug  — LLM + draft_model SpS, both see aug (full).
  specsteer          — LLM on main + drafter on aug + Gate A fusion.

K is ignored for b1_aug / b1_main (no spec decode).
"""
import argparse, hashlib, json, os, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from paths import LLM_PATH, slm_model, vllm_specsteer_targets, REPO_ROOT, MC_QUESTIONS, MC_SUMMARIES

MC_PATH = str(MC_QUESTIONS)
SUMMARY_PATH = str(MC_SUMMARIES)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True,
                choices=["b1_aug", "b1_main", "specsteer", "classical_sps_aug",
                         "classical_sps_main",
                         "b1_drafter_aug", "b1_drafter_main", "scd"],
                help="b1_drafter_aug=drafter(SLM) alone on the full "
                     "conversation (B1); "
                     "classical_sps_main=classical SD with verifier+drafter "
                     "both on the COMPRESSED context (fair-cost "
                     "compressed-SD baseline); "
                     "scd=Speculative Contrastive Decoding baseline, "
                     "expert(32B)/amateur(SLM) on the SAME main context (B3)")
ap.add_argument("--asym_method", default="jsd",
                choices=["gamma_rule", "cma", "jsd", "jsd_pos", "cma_vnorm", "cma_hbase"])
ap.add_argument("--delta_src", default="ours",
                choices=["ours", "raw_aug", "scd"],
                help="δ-source ablation (specsteer mode); forced to 'scd' "
                     "when --mode scd")
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--K", type=int, default=2,
                help="num_speculative_tokens (drafts/step); ignored for b1_*")
ap.add_argument("--max_new", type=int, default=2000)
ap.add_argument("--temperature", type=float, default=0.0,
                help="Sampling temperature; 0.0 = greedy.")
ap.add_argument("--top_p", type=float, default=1.0,
                help="Top-p nucleus sampling; 1.0 = disabled.")
ap.add_argument("--seed", type=int, default=None,
                help="Sampling seed (only meaningful when temperature>0).")
ap.add_argument("--n", type=int, default=271)
ap.add_argument("--slm", default="4B", choices=["0.6B", "1.7B", "4B"])
ap.add_argument("--main_context", default="summary_last_k",
                choices=["last_user", "summary_last_k", "llmlingua_last_k"],
                help="CANONICAL paper config: summary_last_k with last_k=1 (the 'sum1' setup). "
                     "llmlingua_last_k: same shape as summary_last_k but compressed via LLMLingua-2.")
ap.add_argument("--last_k", type=int, default=1,
                help="k for summary_last_k mode (cache supports 1, 2, 4)")
ap.add_argument("--cell", required=True, help="Cell id, used in logs only")
ap.add_argument("--out", required=True, help="Output JSON path (metrics + tokens)")
ap.add_argument("--responses", required=True, help="Output JSONL path (decoded text)")
ap.add_argument("--bs", type=int, default=1,
                help="Batch size: number of prompts per llm.generate() call.")
ap.add_argument("--enforce_eager", action="store_true",
                help="Disable CUDA graph capture entirely (last-resort).")
ap.add_argument("--max_cudagraph_size", type=int, default=128,
                help="Cap cudagraph_capture_sizes to <= this batch size. "
                     "Default 128 is the validated vLLM 0.28 cap across all "
                     "(K, drafter) combos: custom_ops=['+rms_norm'] bypasses "
                     "the RMSNorm fusion bug, but Qwen3-1.7B (head_dim=128, "
                     "hidden=2048) has additional inductor kernel bugs at "
                     "batch>=144 that need cap. Verified: K=6 1.7B passes with "
                     "cap=128. Zero effect on BS=1 inference — those large-batch "
                     "graphs are never dispatched in our single-stream workload.")
args = ap.parse_args()

SLM_PATH = slm_model(args.slm)

# Traceability: hash the live specsteer_model.py we're about to run against.
_specsteer_model_path, _specsteer_sampler_path = vllm_specsteer_targets()
LIVE_SPECSTEER = str(_specsteer_model_path)
LIVE_SAMPLER = str(_specsteer_sampler_path)
def _sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except FileNotFoundError:
        return "MISSING"
SHA_MODEL = _sha(LIVE_SPECSTEER)
SHA_SAMPLER = _sha(LIVE_SAMPLER)
print(f"[trace] cell={args.cell} K={args.K} live specsteer_model.sha256={SHA_MODEL}", flush=True)
print(f"[trace] cell={args.cell} K={args.K} live specsteer_sampler.sha256={SHA_SAMPLER}", flush=True)

import vllm.config.speculative as _sc
_sc.SpeculativeConfig.verify_equal_vocab_size_if_draft_model = lambda self: None

# Monkey-patch BEFORE LLM instantiation so we capture spec decode stats from
# the scheduler. With disable_log_stats=True (default for offline
# LLM) skips LoggingStatLogger entirely → patching SpecDecodingLogging.observe
# does nothing. Patching SpecDecodingStats.observe_draft instead — that's the
# source of truth, called by scheduler.make_spec_decoding_stats every step.
import vllm.v1.spec_decode.metrics as _sdm
_SPEC_TOTALS = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0,
                "per_pos_accepted": None, "K_seen": None}
_orig_observe_draft = _sdm.SpecDecodingStats.observe_draft
def _capture_observe_draft(self, num_draft_tokens, num_accepted_tokens):
    _SPEC_TOTALS["drafts"] += 1
    _SPEC_TOTALS["draft_tokens"] += num_draft_tokens
    _SPEC_TOTALS["accepted_tokens"] += num_accepted_tokens
    if _SPEC_TOTALS["per_pos_accepted"] is None:
        _SPEC_TOTALS["per_pos_accepted"] = [0] * self.num_spec_tokens
        _SPEC_TOTALS["K_seen"] = self.num_spec_tokens
    for i in range(num_accepted_tokens):
        if i < len(_SPEC_TOTALS["per_pos_accepted"]):
            _SPEC_TOTALS["per_pos_accepted"][i] += 1
    return _orig_observe_draft(self, num_draft_tokens, num_accepted_tokens)
_sdm.SpecDecodingStats.observe_draft = _capture_observe_draft

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# B1 (drafter-alone) runs the SLM as the only model → use its tokenizer.
TOK_PATH = SLM_PATH if args.mode in ("b1_drafter_aug", "b1_drafter_main") else LLM_PATH
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

SUMMARY_LOOKUP = None
if args.main_context == "summary_last_k":
    sum_d = json.load(open(SUMMARY_PATH))
    SUMMARY_LOOKUP = sum_d["summaries"]
    print(f"[setup] loaded summary_lookup n={len(SUMMARY_LOOKUP)} k={args.last_k}", flush=True)

LLMLINGUA_LOOKUP = None
if args.main_context == "llmlingua_last_k":
    _llm_path = f"{REPO_ROOT}/experiments/cache/mc_llmlingua.json"
    if not os.path.exists(_llm_path):
        raise SystemExit(f"mc llmlingua needs {_llm_path} (run gen_mc_llmlingua.py)")
    LLMLINGUA_LOOKUP = json.load(open(_llm_path))["compressions"]
    print(f"[setup] loaded llmlingua_lookup n={len(LLMLINGUA_LOOKUP)} k={args.last_k}", flush=True)


def _build_main_summary_last_k(conv, qid, k):
    """Main = [system: summary of earlier turns] + last k turns (must start with user)."""
    if qid not in SUMMARY_LOOKUP:
        return None
    summary = SUMMARY_LOOKUP[qid].get(str(k))
    if not summary:
        return None
    msgs = list(conv[-k:])
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    if not msgs:
        return None
    return [{"role": "system", "content": summary}] + msgs


def build_prompts(mode):
    """Returns list of (idx, qid, prompt_ids, aug_ids_or_None)."""
    entries = []
    with open(MC_PATH) as f:
        for i, line in enumerate(f):
            if i >= args.n: break
            entries.append(json.loads(line))
    out = []
    MAX_L = 8192 - args.max_new - 16
    skipped = 0
    for i, e in enumerate(entries):
        conv = e["CONVERSATION"]
        qid = e["QUESTION_ID"]
        last_user = next((m for m in reversed(conv) if m["role"] == "user"), None)
        if last_user is None: continue

        if args.main_context == "last_user":
            msg_main = [last_user]
        elif args.main_context == "llmlingua_last_k":
            comp = (LLMLINGUA_LOOKUP.get(qid) or {}).get(str(args.last_k))
            if not comp:
                skipped += 1
                continue
            _msgs = list(conv[-args.last_k:])
            while _msgs and _msgs[0]["role"] != "user":
                _msgs = _msgs[1:]
            if not _msgs:
                skipped += 1
                continue
            msg_main = [{"role": "system", "content": comp}] + _msgs
        else:
            msg_main = _build_main_summary_last_k(conv, qid, args.last_k)
            if msg_main is None:
                skipped += 1
                continue

        if mode in ("b1_aug", "classical_sps_aug", "b1_drafter_aug"):
            # b1_drafter_aug: SLM is the only model, run it on full conv.
            p = tok.apply_chat_template(conv, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
            ids = tok.encode(p)
            aug = None
        elif mode in ("b1_main", "b1_drafter_main", "classical_sps_main"):
            # b1_drafter_main: SLM alone on the COMPRESSED context (B4).
            # classical_sps_main: 32B + SLM both on compressed context.
            p = tok.apply_chat_template(msg_main, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
            ids = tok.encode(p)
            aug = None
        elif mode == "scd":
            # SCD baseline: expert(32B)/amateur(SLM) on the SAME main
            # context (aug==main); SCD-style δ via ASYMSPEC_DELTA_SRC=scd.
            main_p = tok.apply_chat_template(msg_main, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
            ids = tok.encode(main_p)
            aug = list(ids)
        else:  # specsteer
            main_p = tok.apply_chat_template(msg_main, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
            aug_p = tok.apply_chat_template(conv, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
            ids = tok.encode(main_p)
            aug = tok.encode(aug_p)
        if len(ids) > MAX_L or (aug is not None and len(aug) > MAX_L):
            skipped += 1
            continue
        out.append((i, qid, ids, aug))
    print(f"[setup] mode={mode}  kept {len(out)}/{len(entries)}  skipped={skipped}", flush=True)
    return out


prompts = build_prompts(args.mode)

BASE_MODEL = SLM_PATH if args.mode in ("b1_drafter_aug", "b1_drafter_main") else LLM_PATH
kwargs = dict(model=BASE_MODEL, dtype="bfloat16", trust_remote_code=True,
              max_model_len=8192, gpu_memory_utilization=0.65,
              enforce_eager=args.enforce_eager,
              # disable_log_stats=False enables vLLM scheduler to construct
              # SpecDecodingStats and call observe_draft; without it our patch
              # never fires (scheduler.py:1938 short-circuits on log_stats=False).
              disable_log_stats=False)
# Root-cause fix for "CUDA illegal memory access" during graph capture:
# Inductor's auto-fusion of Qwen3's PyTorch-native RMSNorm produces a triton
# kernel `triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0` with
# pointer-arithmetic OOB (`tl.load(ptr + stride*xindex, mask)` where xindex
# is computed before mask). Triggers cudaErrorIllegalAddress at specific
# (hidden_size, K, batch_size) combos.
#
# Fix: enable vLLM's hand-tuned CUDA `rms_norm` CustomOp via
# custom_ops=['none', '+rms_norm']. This puts rms_norm in vLLM's
# splitting_ops list, so inductor treats it as an opaque black box and
# never generates the buggy fusion. Side benefit: vLLM's hand-tuned CUDA
# RMSNorm is FASTER than inductor's triton (~+40% tps observed on K=2
# 0.6B compared to default config). All other inductor optimizations
# (combo_kernels, autotune, etc.) keep their defaults.
#
# Verified: K=6 0.6B (worst case in our matrix) passes full 48-size
# graph capture with this config alone, no cap needed.
compilation_cfg = {
    "custom_ops": ["none", "+rms_norm"],
    # Explicitly disable vLLM's pass_config fusions. When custom_ops includes
    # rms_norm, vLLM auto-enables `fuse_norm_quant=True` which tries to fuse
    # RMSNorm + quantize. Without quantization in our setup, this fusion
    # produces buggy kernels: K=4 0.6B → illegal memory crash, classical_1.7B
    # all K → silent corruption (AR<0.05). Explicitly setting all fuse_* False
    # bypasses these passes; rms_norm CustomOp still runs as hand-tuned CUDA.
    "pass_config": {
        "fuse_norm_quant": False, "fuse_act_quant": False,
        "fuse_attn_quant": False, "enable_sp": False,
        "fuse_gemm_comms": False, "fuse_allreduce_rms": False,
    },
}
# --max_cudagraph_size kept as escape hatch / diagnostic; default 512 = no cap.
if args.max_cudagraph_size < 512:
    _DEFAULT_CGS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
                    112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                    208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336,
                    352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
    _capped = [s for s in _DEFAULT_CGS if s <= args.max_cudagraph_size]
    compilation_cfg["cudagraph_capture_sizes"] = _capped
    print(f"[setup] cudagraph_capture_sizes capped to <={args.max_cudagraph_size} "
          f"({len(_capped)} sizes; max={_capped[-1]})", flush=True)
else:
    print(f"[setup] cudagraph_capture_sizes: default (no cap)", flush=True)
print(f"[setup] custom_ops=['none', '+rms_norm'] (root-cause fix; "
      f"bypasses inductor RMSNorm fusion bug, hand-tuned CUDA kernel)",
      flush=True)
kwargs["compilation_config"] = compilation_cfg
if args.mode in ("specsteer", "scd"):
    kwargs["speculative_config"] = {
        "method": "specsteer", "model": SLM_PATH,
        "num_speculative_tokens": args.K,
        "specsteer_beta": args.beta, "specsteer_gamma": args.gamma,
    }
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _asym_method_util import setup_asym_method
    setup_asym_method(args.asym_method, beta=args.beta)
    # scd mode forces SCD-style two-model δ; specsteer honors --delta_src.
    os.environ["ASYMSPEC_DELTA_SRC"] = (
        "scd" if args.mode == "scd" else args.delta_src)
    print(f"[setup] ASYMSPEC_METHOD={os.environ.get('ASYMSPEC_METHOD')} "
          f"ASYMSPEC_DELTA_SRC={os.environ['ASYMSPEC_DELTA_SRC']}", flush=True)
elif args.mode in ("classical_sps_aug", "classical_sps_main"):
    # Same speculative config; the prompt content (full vs summary)
    # differs in build_prompts.
    kwargs["speculative_config"] = {
        "method": "draft_model", "model": SLM_PATH,
        "num_speculative_tokens": args.K,
    }
print(f"[setup] loading LLM (K={args.K})...", flush=True)
# OOM-init race retry: gpu_acquire returns when memory near-free, but
# gpu_occupy.py daemon can reoccupy in the gap before LLM claims memory.
# Retry up to 3× with re-poll on ValueError("Free memory ...").
import time as _t, subprocess as _sp
def _wait_gpu_free(min_free_gb=130.0, timeout_s=300):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd: return
    phys = cvd.split(",")[0]
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        try:
            out = _sp.check_output(
                ["nvidia-smi", f"--id={phys}",
                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True, timeout=5).strip()
            if int(out)/1024.0 >= min_free_gb: return
        except Exception: pass
        _t.sleep(2)

_wait_gpu_free()
llm = None
for _attempt in range(3):
    try:
        llm = LLM(**kwargs)
        break
    except ValueError as e:
        if "Free memory" not in str(e) and "memory utilization" not in str(e):
            raise
        print(f"[setup] LLM init OOM race ({_attempt+1}/3), sleep 60 + re-poll", flush=True)
        _t.sleep(60); _wait_gpu_free()
if llm is None:
    raise RuntimeError("LLM init failed 3× with OOM race")
if args.mode in ("specsteer", "scd"):
    print("[setup] SpecSteer Path B enabled", flush=True)

# Warmup
_ = llm.generate([{"prompt_token_ids": prompts[0][2]}],
                 SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)

outputs = {}            # idx → token_ids
qid_by_idx = {}         # idx → qid
prompt_lens = {}        # idx → prompt token count
per_prompt_elapsed = {} # idx → wall seconds for this generate() call
t0 = time.perf_counter()

# Iteration: chunk prompts by --bs. At bs=1, each chunk is a one-element list;
# vLLM applies the single SamplingParams object to that request.
def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

for k_chunk, prompt_chunk in enumerate(_chunk(prompts, args.bs)):
    # Build per-req inputs + sampling params for this chunk.
    reqs = [{"prompt_token_ids": ids} for _, _, ids, _ in prompt_chunk]
    sps = []
    for _, _, _, aug in prompt_chunk:
        sp_kwargs = dict(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new,
        )
        if args.seed is not None:
            sp_kwargs["seed"] = args.seed
        if aug is not None:
            sp_kwargs["extra_args"] = {"specsteer_aug_prompt_ids": aug}
        sps.append(SamplingParams(**sp_kwargs))
    # Track per-req prompt metadata.
    for idx, qid, ids, _ in prompt_chunk:
        qid_by_idx[idx] = qid
        prompt_lens[idx] = len(ids)
    pt0 = time.perf_counter()
    try:
        out_list = llm.generate(reqs, sps, use_tqdm=False)
        for (idx, _, _, _), out in zip(prompt_chunk, out_list):
            outputs[idx] = list(out.outputs[0].token_ids)
            per_prompt_elapsed[idx] = (time.perf_counter() - pt0) / len(prompt_chunk)
    except Exception as e:
        import traceback
        print(f"[fail] chunk {k_chunk}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if "dead" in str(e).lower() or "engine" in type(e).__name__.lower():
            print(f"[ABORT] engine dead at chunk {k_chunk} — stopping", flush=True)
            break
    # Periodic progress + checkpoint (every ~20 prompts equivalent).
    if (k_chunk + 1) * args.bs % 20 < args.bs:
        elapsed = time.perf_counter() - t0
        total = sum(len(v) for v in outputs.values())
        tps = total / elapsed if elapsed > 0 else 0
        ar_str = ""
        if _SPEC_TOTALS["draft_tokens"] > 0:
            ar = _SPEC_TOTALS["accepted_tokens"] / _SPEC_TOTALS["draft_tokens"]
            mal = 1 + _SPEC_TOTALS["accepted_tokens"] / max(_SPEC_TOTALS["drafts"], 1)
            ar_str = f" AR={ar:.3f} MAL={mal:.2f}"
        print(f"[{(k_chunk+1)*args.bs}/{len(prompts)}] tps={tps:.2f} "
              f"elapsed={elapsed:.1f}s bs={args.bs}{ar_str}", flush=True)
        try:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out + ".partial", "w") as _f:
                json.dump({
                    "mode": args.mode, "K": args.K, "cell": args.cell, "bs": args.bs,
                    "out_by_idx": {str(k_): v for k_, v in outputs.items()},
                    "tps": tps, "elapsed": elapsed, "total": total,
                    "n_kept": len(outputs),
                    "partial": True, "progress": f"{(k_chunk+1)*args.bs}/{len(prompts)}",
                    "_spec_totals": dict(_SPEC_TOTALS),
                }, _f)
        except Exception as _e:
            print(f"[warn] periodic checkpoint write failed: {_e}", flush=True)
elapsed = time.perf_counter() - t0
total = sum(len(v) for v in outputs.values())
tps = total / elapsed if elapsed > 0 else 0

# Derive spec-decode efficiency metrics (None for b1_aug / b1_main).
spec_metrics = None
if _SPEC_TOTALS["drafts"] > 0:
    n_drafts = _SPEC_TOTALS["drafts"]
    n_dtoks = _SPEC_TOTALS["draft_tokens"]
    n_acc = _SPEC_TOTALS["accepted_tokens"]
    per_pos = _SPEC_TOTALS["per_pos_accepted"] or []
    # Per-position acceptance rate: fraction of drafts where position i was accepted.
    per_pos_rate = [c / n_drafts for c in per_pos]
    spec_metrics = {
        "K": args.K,
        "num_drafts": int(n_drafts),
        "num_draft_tokens": int(n_dtoks),
        "num_accepted_tokens": int(n_acc),
        "draft_acceptance_rate": n_acc / n_dtoks if n_dtoks > 0 else None,
        "mean_acceptance_length": 1 + (n_acc / n_drafts),
        "per_position_acceptance_rate": per_pos_rate,
        "per_position_accepted_count": [int(c) for c in per_pos],
        "accepted_tokens_per_sec": n_acc / elapsed if elapsed > 0 else None,
    }

# Per-prompt distribution stats (mean / median / p90 / max) — useful for K-sweep.
import statistics as _st
def _dist(xs):
    if not xs: return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return {
        "n": n,
        "mean": _st.mean(xs_sorted),
        "median": xs_sorted[n // 2],
        "p10": xs_sorted[max(0, int(0.1 * n) - 1)],
        "p90": xs_sorted[min(n - 1, int(0.9 * n))],
        "min": xs_sorted[0],
        "max": xs_sorted[-1],
    }
out_lens = [len(v) for v in outputs.values()]
prompt_len_list = [prompt_lens[i] for i in outputs.keys()]
elapsed_list = [per_prompt_elapsed[i] for i in outputs.keys() if i in per_prompt_elapsed]
distributions = {
    "prompt_tokens": _dist(prompt_len_list),
    "output_tokens": _dist(out_lens),
    "per_prompt_elapsed_s": _dist(elapsed_list),
}

print(f"[RESULT] cell={args.cell} K={args.K} tps={tps:.2f} n={len(outputs)} total={total} elapsed={elapsed:.1f}s", flush=True)
if spec_metrics:
    print(f"[RESULT] AR={spec_metrics['draft_acceptance_rate']:.3f} "
          f"MAL={spec_metrics['mean_acceptance_length']:.2f} "
          f"per_pos={['{:.3f}'.format(r) for r in spec_metrics['per_position_acceptance_rate']]}",
          flush=True)

# Metrics + tokens JSON.
config_snapshot = {
    "cell": args.cell, "mode": args.mode, "K": args.K,
    "max_new": args.max_new, "n_requested": args.n,
    "slm": args.slm if args.mode in (
        "specsteer", "classical_sps_aug", "scd", "b1_drafter_aug",
        "b1_drafter_main") else None,
    "main_context": args.main_context if args.mode in (
        "specsteer", "b1_main", "scd") else None,
    "asym_method": args.asym_method if args.mode in ("specsteer", "scd") else None,
    "delta_src": os.environ.get("ASYMSPEC_DELTA_SRC")
        if args.mode in ("specsteer", "scd") else None,
    "last_k": args.last_k if args.main_context == "summary_last_k" else None,
    "llm_path": LLM_PATH, "slm_path": SLM_PATH,
    "specsteer_beta": args.beta, "specsteer_gamma": args.gamma,
    "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
    "live_specsteer_model_sha": SHA_MODEL,
    "live_specsteer_sampler_sha": SHA_SAMPLER,
    "enforce_eager": args.enforce_eager,
    "max_cudagraph_size": args.max_cudagraph_size,
}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump({
        "mode": args.mode, "K": args.K, "cell": args.cell,
        "out_by_idx": {str(k): v for k, v in outputs.items()},
        "tps": tps, "elapsed": elapsed, "total": total,
        "n_kept": len(outputs),
        "spec_metrics": spec_metrics,
        "distributions": distributions,
        "per_prompt_elapsed_s": {str(k): v for k, v in per_prompt_elapsed.items()},
        "prompt_token_lens": {str(k): v for k, v in prompt_lens.items()},
        "config": config_snapshot,
    }, f)
print(f"[saved] metrics → {args.out}", flush=True)

# Decoded responses JSONL.
os.makedirs(os.path.dirname(args.responses) or ".", exist_ok=True)
with open(args.responses, "w") as f:
    for idx, toks in outputs.items():
        text = tok.decode(toks, skip_special_tokens=True)
        f.write(json.dumps({"idx": idx, "qid": qid_by_idx[idx], "response": text}) + "\n")
print(f"[saved] responses → {args.responses}", flush=True)
