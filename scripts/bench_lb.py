"""LongBench multi-document QA benchmark for AsymSpec and its baselines.

Configuration:
  1. Dataset: LongBench Multi-doc QA (HotpotQA + 2WikiMQA + MuSiQue,
     200 each = 600 samples). Loaded from data/longbench/raw/*.jsonl
     and tagged with `dataset` field.
  2. Summary: per-_id cache from data/longbench/summaries.jsonl
     (Qwen3-32B offline, cap=1500, query-agnostic).
  3. Prompt: custom CoT template ("Read passages... Question:... Think
     step by step, then answer in 'The answer is X.' format"). Same
     template for all modes — only context content differs.
  4. main_context choices: "question_only" (closed-book floor) or
     "summary" (compressed floor).
  5. max_model_len: 24576 (LongBench passages can hit ~18K tokens).
  6. Output JSONL has `dataset`, `gold_answers`, `_id` per sample for
     downstream F1/EM/judge eval.

Modes:
  b1_aug             — LLM on full passages.
  b1_main            — LLM on question_only OR summary.
  classical_sps_aug  — LLM + draft_model SpS, both see full passages.
  specsteer          — LLM on summary, drafter on full passages, Gate A.
"""
import argparse, hashlib, json, os, sys, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from paths import LLM_PATH, slm_model, vllm_specsteer_targets, REPO_ROOT, LB_RAW, LB_SUMMARIES

SUMMARY_PATH = str(LB_SUMMARIES)
TASKS = ["hotpotqa", "2wikimqa", "musique"]

PROMPT_TEMPLATE = """You are answering a multi-hop question based on the following passages.

{passages}

Question: {question}

Think step by step, then provide your final answer in the format "The answer is XXX."."""

QUESTION_ONLY_TEMPLATE = """Answer the following multi-hop question using your own knowledge.

Question: {question}

Think step by step, then provide your final answer in the format "The answer is XXX."."""

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True,
                choices=["b1_aug", "b1_main", "specsteer", "classical_sps_aug",
                         "classical_sps_main",
                         "b1_drafter_aug", "b1_drafter_main", "scd", "b2_append"],
                help="b1_drafter_main=drafter(SLM) alone on the COMPRESSED "
                     "main context (B4, amateur floor); "
                     "b1_drafter_aug=drafter(SLM) alone on full passages (B1); "
                     "classical_sps_main=classical SD with verifier+drafter both on "
                     "the COMPRESSED context (fair-cost compressed-SD baseline); "
                     "scd=Speculative Contrastive Decoding baseline, expert(32B)"
                     "/amateur(SLM) on the SAME main context (B3); "
                     "b2_append=non-speculative transfer, 32B on main with the "
                     "drafter's answer appended (B2, needs --draft_answers)")
ap.add_argument("--K", type=int, default=2,
                help="num_speculative_tokens (drafts/step); ignored for b1_*")
ap.add_argument("--asym_method", default="jsd",
                choices=["gamma_rule", "cma", "jsd", "jsd_pos", "cma_vnorm", "cma_hbase"],
                help="AsymSpec acceptance/fusion variant (specsteer/scd mode)")
ap.add_argument("--delta_src", default="ours",
                choices=["ours", "raw_aug", "scd"],
                help="δ-source ablation (specsteer mode): ours=log sm(aug)-log "
                     "sm(base); raw_aug=log sm(aug); scd=log sm(target)-log "
                     "sm(base). Forced to 'scd' when --mode scd.")
ap.add_argument("--draft_answers", default="",
                help="b2_append mode: path to a b1_drafter_aug responses.jsonl "
                     "whose {_id: response} is appended to the 32B main prompt.")
ap.add_argument("--truncate_tokens", type=int, default=1500,
                help="main_context=truncate: hard token cap on passages "
                     "(matches the offline summary cap=1500).")
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--temperature", type=float, default=0.0,
                help="Sampling temperature; 0.0 = greedy.")
ap.add_argument("--top_p", type=float, default=1.0,
                help="Top-p nucleus sampling; 1.0 = disabled.")
ap.add_argument("--seed", type=int, default=None,
                help="Sampling seed (only meaningful when temperature>0).")
ap.add_argument("--slm", default="4B", choices=["0.6B", "1.7B", "4B"])
ap.add_argument("--main_context", default="summary",
                choices=["question_only", "summary", "truncate", "llmlingua"],
                help="Used by b1_main + specsteer + scd + b2_append. "
                     "summary=Qwen3-32B offline cap=1500; "
                     "question_only=closed-book floor; "
                     "truncate=passages hard-cut to --truncate_tokens (ABL-4); "
                     "llmlingua=LLMLingua-2 cache.")
ap.add_argument("--cell", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--responses", required=True)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--enforce_eager", action="store_true")
ap.add_argument("--max_cudagraph_size", type=int, default=128,
                help="Cap cudagraph_capture_sizes; 128 is the validated vLLM 0.28 "
                     "inductor bugs without affecting BS=1.")
ap.add_argument("--max_model_len", type=int, default=24576)
ap.add_argument(
    "--specsteer-main-max-model-len", "--specsteer_main_max_model_len",
    dest="specsteer_main_max_model_len", type=int, default=None,
    help=("Compressed-context verifier/base capacity. The global "
          "--max_model_len remains the full-context drafter capacity."),
)
ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
ap.add_argument("--cpu_offload_gb", type=float, default=0.0,
                help="GiB of model weights to offload to CPU per GPU.")
ap.add_argument("--tp", type=int, default=1,
                help="Tensor parallel size for verifier and drafter.")
# YaRN rope-scaling to extend Qwen3's native 40960 window. Applied symmetrically
# to verifier (via hf_overrides) AND drafter (via SpeculativeConfig monkey-patch
# — the drafter ModelConfig hardcodes hf_overrides=hf_config_override
# callable and ignores the speculative_config dict's hf fields, so the only
# non-destructive path is to chain that staticmethod at runtime). Both models
# must share rope_scaling for K-token speculation alignment to remain valid.
# Default = off (None) → unchanged native 40960. Qwen3 recommended base = 32768.
ap.add_argument("--yarn_factor", type=float, default=None,
                help="If set, enable YaRN rope-scaling with this factor on "
                     "BOTH verifier and drafter (Qwen3 family). "
                     "factor=4.0 → ~131K effective from 32768 base.")
ap.add_argument("--yarn_original_max", type=int, default=32768,
                help="YaRN original_max_position_embeddings (Qwen3 default 32768).")
ap.add_argument("--n", type=int, default=0,
                help="If >0, limit to first N samples (smoke testing). "
                     "Default 0 = full 600.")
ap.add_argument("--tasks", default="",
                help="Comma-separated subset of {hotpotqa,2wikimqa,musique}. "
                     "Empty = all 3. Used for per-dataset cells (e.g. "
                     "HotpotQA-only validation).")
args = ap.parse_args()
if args.tasks:
    _wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in _wanted:
        if t not in TASKS:
            raise SystemExit(f"Unknown task '{t}', expected from {TASKS}")
    TASKS = _wanted
    print(f"[setup] task filter: {TASKS}", flush=True)

SLM_PATH = slm_model(args.slm)

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

# Race fix: gpu_acquire returns when memory drops below 5GB, but the CUDA
# driver's free-memory view lags. Poll until the assigned physical GPU has
# >= 130GB free, or timeout after 300s.
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cvd and args.tp == 1:
    import subprocess
    _phys = _cvd.split(",")[0]
    _deadline = time.time() + 300
    _last = -1.0
    while time.time() < _deadline:
        try:
            _out = subprocess.check_output(
                ["nvidia-smi", f"--id={_phys}",
                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip()
            _free_gb = int(_out) / 1024.0
            if _free_gb >= 130.0:
                if _last >= 0:
                    print(f"[wait_gpu_free] GPU {_phys} ready free={_free_gb:.1f}GB",
                          flush=True)
                break
            _last = _free_gb
        except Exception as _e:
            print(f"[wait_gpu_free] poll err: {_e}", flush=True)
        time.sleep(2)
    else:
        print(f"[wait_gpu_free] TIMEOUT GPU {_phys} free={_last:.1f}GB",
              flush=True)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# B1 (drafter-alone) runs the SLM as the only model → use its tokenizer.
# Qwen3 0.6B/1.7B/4B/32B share one vocab, but be exact for encode/decode.
TOK_PATH = SLM_PATH if args.mode in ("b1_drafter_aug", "b1_drafter_main") else LLM_PATH
tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)


def _load_summaries() -> dict:
    """Load per-_id summaries from cache. Returns {_id: summary_text}."""
    out = {}
    with open(SUMMARY_PATH) as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d["summary"]
    return out


_MAIN_CTX_MODES = ("b1_main", "specsteer", "scd", "b2_append", "b1_drafter_main",
                   "classical_sps_main")
SUMMARY_LOOKUP = None
if args.mode in _MAIN_CTX_MODES and args.main_context == "summary":
    SUMMARY_LOOKUP = _load_summaries()
    print(f"[setup] loaded summaries n={len(SUMMARY_LOOKUP)}", flush=True)

LLMLINGUA_LOOKUP = None
if args.mode in _MAIN_CTX_MODES and args.main_context == "llmlingua":
    _cpath = f"{REPO_ROOT}/experiments/cache/lb_llmlingua.json"
    if not os.path.exists(_cpath):
        raise SystemExit(f"lb llmlingua needs {_cpath} (run gen_lb_llmlingua.py)")
    LLMLINGUA_LOOKUP = json.load(open(_cpath))["compressions"]
    print(f"[setup] loaded lb llmlingua compressions n={len(LLMLINGUA_LOOKUP)}",
          flush=True)


def _load_draft_answers() -> dict:
    """b2_append: {_id: drafter_response_text} from a b1_drafter_aug run."""
    out = {}
    with open(args.draft_answers) as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d["response"]
    return out


DRAFT_ANSWERS = None
if args.mode == "b2_append":
    if not args.draft_answers or not os.path.exists(args.draft_answers):
        raise SystemExit(
            "b2_append needs --draft_answers <b1_drafter_aug responses.jsonl>")
    DRAFT_ANSWERS = _load_draft_answers()
    print(f"[setup] loaded draft answers n={len(DRAFT_ANSWERS)}", flush=True)


def _format_user_prompt(passages: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(passages=passages, question=question)


def _format_q_only(question: str) -> str:
    return QUESTION_ONLY_TEMPLATE.format(question=question)


def _to_chat_ids(user_text: str) -> list[int]:
    msgs = [{"role": "user", "content": user_text}]
    p = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tok.encode(p)


def build_prompts(mode):
    """Returns list of (idx, _id, dataset, gold_answers, prompt_ids, aug_ids_or_None)."""
    samples = []
    for task in TASKS:
        with open(os.path.join(LB_RAW, f"{task}.jsonl")) as f:
            for line in f:
                ex = json.loads(line)
                ex["_task"] = task
                samples.append(ex)
    if args.n > 0:
        # Stratified subset: take first N//3 per task
        per_task = max(1, args.n // len(TASKS))
        samples = [s for t in TASKS for s in samples if s["_task"] == t][:0]
        for task in TASKS:
            ts = [s for s in (json.loads(line) for line in
                              open(os.path.join(LB_RAW, f"{task}.jsonl")))][:per_task]
            for s in ts:
                s["_task"] = task
            samples.extend(ts)
        print(f"[setup] subsetted to {len(samples)} samples ({per_task}/task)",
              flush=True)

    out = []
    MAX_L = args.max_model_len - args.max_new - 16
    skipped = 0
    for i, s in enumerate(samples):
        question = s["input"]
        passages = s["context"]
        sid = s["_id"]
        gold = s["answers"]
        task = s["_task"]

        # Build aug (full passages) and main (question_only/summary/truncate).
        aug_text = _format_user_prompt(passages, question)
        if args.main_context == "summary":
            summ = SUMMARY_LOOKUP.get(sid) if SUMMARY_LOOKUP else None
            if summ is None and mode in _MAIN_CTX_MODES:
                skipped += 1
                continue
            main_text = _format_user_prompt(summ or "", question)
        elif args.main_context == "llmlingua":
            comp = LLMLINGUA_LOOKUP.get(sid) if LLMLINGUA_LOOKUP else None
            if comp is None and mode in _MAIN_CTX_MODES:
                skipped += 1
                continue
            main_text = _format_user_prompt(comp or "", question)
        elif args.main_context == "truncate":
            _ptoks = tok.encode(passages)[:args.truncate_tokens]
            main_text = _format_user_prompt(
                tok.decode(_ptoks, skip_special_tokens=True), question)
        else:  # question_only
            main_text = _format_q_only(question)

        # Encode based on mode (SAME prompt template, only content differs).
        if mode in ("b1_aug", "classical_sps_aug", "b1_drafter_aug"):
            # b1_drafter_aug: the SLM is the only model, run it on full passages.
            ids = _to_chat_ids(aug_text)
            aug = None
        elif mode in ("b1_main", "b1_drafter_main", "classical_sps_main"):
            # b1_drafter_main: SLM alone on the COMPRESSED context (B4).
            # classical_sps_main: 32B verifier + SLM drafter, BOTH on the
            #   compressed context — fair-cost compressed-SD
            #   baseline. No aug substitution; drafter reads same prompt.
            ids = _to_chat_ids(main_text)
            aug = None
        elif mode == "b2_append":
            draft = DRAFT_ANSWERS.get(sid)
            if draft is None:
                skipped += 1
                continue
            ids = _to_chat_ids(
                main_text
                + "\n\nA draft answer from an assistant that could read the "
                  "full source passages:\n" + draft.strip()
                + "\n\nUsing the draft only as a hint, give your own final "
                  'answer in the format "The answer is XXX.".')
            aug = None
        elif mode == "scd":
            # SCD baseline: expert(32B) + amateur(SLM) contrastive on the
            # SAME main context (no asymmetric-context advantage). aug==main
            # so the drafter/amateur reads exactly the verifier's context;
            # the SCD-style δ = log sm(target) - log sm(base) is selected via
            # ASYMSPEC_DELTA_SRC=scd (forced below).
            ids = _to_chat_ids(main_text)
            aug = list(ids)
        else:  # specsteer
            ids = _to_chat_ids(main_text)
            aug = _to_chat_ids(aug_text)

        if len(ids) > MAX_L or (aug is not None and len(aug) > MAX_L):
            skipped += 1
            continue
        out.append((i, sid, task, gold, ids, aug))
    print(f"[setup] mode={mode} kept {len(out)}/{len(samples)} skipped={skipped}",
          flush=True)
    return out


prompts = build_prompts(args.mode)

# B1 (drafter-alone) loads the SLM as the only model; everything else uses 32B.
BASE_MODEL = SLM_PATH if args.mode in ("b1_drafter_aug", "b1_drafter_main") else LLM_PATH
kwargs = dict(
    model=BASE_MODEL, dtype="bfloat16", trust_remote_code=True,
    max_model_len=args.max_model_len,
    tensor_parallel_size=args.tp,
    cpu_offload_gb=args.cpu_offload_gb,
    # Path B evaluates full and compressed views independently, so the legacy
    # combined-forward requirement no longer applies. Keep a 16K scheduling
    # batch and let chunked prefill cover the 24K sequence. Profiling a single
    # 24K batch consumes enough activation memory to leave the three KV views
    # just short of capacity on 24 GiB cards.
    max_num_batched_tokens=min(args.max_model_len, 16384),
    gpu_memory_utilization=args.gpu_memory_utilization,
    enforce_eager=args.enforce_eager, disable_log_stats=False,
)
# vLLM's automatic profile assigns every otherwise-free byte to KV cache.  On
# the validated 4x24 GiB setup that leaves too little transient workspace for
# the longest augmented-context prefill.  The three AsymSpec cache views need
# about 4.69 GiB/rank at 24K, so reserve 5 GiB and leave the remaining memory
# as activation headroom for this exact hardware acceptance configuration.
if (args.mode in ("specsteer", "scd") and args.tp == 4
        and args.cpu_offload_gb > 0 and args.max_model_len == 24576):
    kwargs["kv_cache_memory_bytes"] = 5 * 1024**3
compilation_cfg = {
    "custom_ops": ["none", "+rms_norm"],
    "pass_config": {
        "fuse_norm_quant": False, "fuse_act_quant": False,
        "fuse_attn_quant": False, "enable_sp": False,
        "fuse_gemm_comms": False, "fuse_allreduce_rms": False,
    },
}
if args.max_cudagraph_size < 512:
    _DEFAULT_CGS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
                    112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                    208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336,
                    352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
    _capped = [s for s in _DEFAULT_CGS if s <= args.max_cudagraph_size]
    compilation_cfg["cudagraph_capture_sizes"] = _capped
    print(f"[setup] cudagraph_capture_sizes capped to <={args.max_cudagraph_size} "
          f"({len(_capped)} sizes; max={_capped[-1]})", flush=True)
print(f"[setup] custom_ops=['none', '+rms_norm']; pass_config disabled "
      f"for stable graph capture", flush=True)
kwargs["compilation_config"] = compilation_cfg

if args.mode in ("specsteer", "scd"):
    kwargs["speculative_config"] = {
        "method": "specsteer", "model": SLM_PATH,
        "num_speculative_tokens": args.K,
        "draft_tensor_parallel_size": args.tp,
        "specsteer_beta": args.beta, "specsteer_gamma": args.gamma,
    }
    if args.specsteer_main_max_model_len is not None:
        kwargs["speculative_config"]["specsteer_main_max_model_len"] = (
            args.specsteer_main_max_model_len)
    os.environ["ASYMSPEC_METHOD"] = args.asym_method
    # scd mode forces the SCD-style two-model δ; specsteer honors --delta_src.
    os.environ["ASYMSPEC_DELTA_SRC"] = (
        "scd" if args.mode == "scd" else args.delta_src)
    os.environ.pop("ASYMSPEC_BETA_OVERRIDE", None)
    print(f"[setup] ASYMSPEC_METHOD={os.environ['ASYMSPEC_METHOD']} "
          f"ASYMSPEC_DELTA_SRC={os.environ['ASYMSPEC_DELTA_SRC']}", flush=True)
elif args.mode in ("classical_sps_aug", "classical_sps_main"):
    # Both share the same vLLM spec config: classical (Leviathan) SD with
    # a draft_model. Only the prompt content differs (full vs summary),
    # handled in build_prompts above.
    kwargs["speculative_config"] = {
        "method": "draft_model", "model": SLM_PATH,
        "num_speculative_tokens": args.K,
        "draft_tensor_parallel_size": args.tp,
    }

# YaRN rope-scaling injection (optional, OFF by default). Extends Qwen3's
# native 40960 window symmetrically on BOTH verifier and drafter. K-token
# speculation requires both models to share an identical effective position
# encoding — otherwise the drafter's position-i logits and the verifier's
# reject-path comparison drift apart and acceptance rates collapse.
#
# Verifier: simple dict hf_overrides.
# Drafter:  SpeculativeConfig hardcodes
#           hf_overrides=SpeculativeConfig.hf_config_override (a callable);
#           the speculative_config={} dict gives no other lever. We chain
#           that staticmethod at runtime to inject rope_scaling.
if args.yarn_factor is not None:
    if args.max_model_len <= 40960:
        print(f"[setup] WARN: yarn_factor={args.yarn_factor} but "
              f"max_model_len={args.max_model_len} <= 40960 (native window). "
              f"YaRN extension wasted unless --max_model_len is raised.",
              flush=True)
    _rope_scaling = {
        "rope_type": "yarn",
        "factor": args.yarn_factor,
        "original_max_position_embeddings": args.yarn_original_max,
    }
    # vLLM's max_model_len validator reads config.max_position_embeddings
    # directly (NOT factor-scaled), so we must bump it in hf_overrides as
    # well or vLLM raises ValidationError when max_model_len > 40960.
    _effective_max = int(args.yarn_factor * args.yarn_original_max)
    kwargs["hf_overrides"] = {
        "rope_scaling": _rope_scaling,
        "max_position_embeddings": _effective_max,
    }
    if args.mode in ("specsteer", "scd", "classical_sps_aug"):
        from vllm.config.speculative import SpeculativeConfig as _SpecCfg
        _orig_override = _SpecCfg.hf_config_override
        def _patched_hf_override(hf_config, _orig=_orig_override,
                                 _rs=_rope_scaling, _max=_effective_max):
            cfg = _orig(hf_config)
            cfg.rope_scaling = _rs
            cfg.max_position_embeddings = _max
            return cfg
        _SpecCfg.hf_config_override = staticmethod(_patched_hf_override)
        print(f"[setup] YaRN drafter: SpeculativeConfig.hf_config_override "
              f"chained (rope_scaling={_rope_scaling})", flush=True)
    print(f"[setup] YaRN ON: factor={args.yarn_factor}, "
          f"original_max={args.yarn_original_max}, "
          f"effective max ~= {int(args.yarn_factor * args.yarn_original_max)} "
          f"(max_model_len={args.max_model_len})", flush=True)

print(f"[setup] loading LLM (K={args.K}) model={BASE_MODEL}...", flush=True)
llm = LLM(**kwargs)
if args.mode in ("specsteer", "scd"):
    print("[setup] SpecSteer Path B enabled", flush=True)

# Warmup
_ = llm.generate(
    [{"prompt_token_ids": prompts[0][4]}],
    SamplingParams(temperature=0, max_tokens=8), use_tqdm=False,
)

outputs = {}            # idx → token_ids
sid_by_idx = {}         # idx → _id
task_by_idx = {}        # idx → dataset name
gold_by_idx = {}        # idx → gold answers list
prompt_lens = {}        # idx → prompt token count
per_prompt_elapsed = {} # idx → wall seconds
t0 = time.perf_counter()


def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


for k_chunk, prompt_chunk in enumerate(_chunk(prompts, args.bs)):
    reqs = [{"prompt_token_ids": ids} for _, _, _, _, ids, _ in prompt_chunk]
    sps = []
    for _, _, _, _, _, aug in prompt_chunk:
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
    for idx, sid, task, gold, ids, _ in prompt_chunk:
        sid_by_idx[idx] = sid
        task_by_idx[idx] = task
        gold_by_idx[idx] = gold
        prompt_lens[idx] = len(ids)
    pt0 = time.perf_counter()
    try:
        out_list = llm.generate(reqs, sps, use_tqdm=False)
        for (idx, _, _, _, _, _), out in zip(prompt_chunk, out_list):
            outputs[idx] = list(out.outputs[0].token_ids)
            per_prompt_elapsed[idx] = (time.perf_counter() - pt0) / len(prompt_chunk)
    except Exception as e:
        import traceback
        print(f"[fail] chunk {k_chunk}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        if "dead" in str(e).lower() or "engine" in type(e).__name__.lower():
            print(f"[ABORT] engine dead at chunk {k_chunk} — stopping", flush=True)
            break
    if (k_chunk + 1) * args.bs % 50 < args.bs:
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

elapsed = time.perf_counter() - t0
total = sum(len(v) for v in outputs.values())
tps = total / elapsed if elapsed > 0 else 0

spec_metrics = None
if _SPEC_TOTALS["drafts"] > 0:
    n_drafts = _SPEC_TOTALS["drafts"]
    n_dtoks = _SPEC_TOTALS["draft_tokens"]
    n_acc = _SPEC_TOTALS["accepted_tokens"]
    per_pos = _SPEC_TOTALS["per_pos_accepted"] or []
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

import statistics as _st
def _dist(xs):
    if not xs: return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return {"n": n, "mean": _st.mean(xs_sorted),
            "median": xs_sorted[n // 2],
            "p10": xs_sorted[max(0, int(0.1 * n) - 1)],
            "p90": xs_sorted[min(n - 1, int(0.9 * n))],
            "min": xs_sorted[0], "max": xs_sorted[-1]}
out_lens = [len(v) for v in outputs.values()]
prompt_len_list = [prompt_lens[i] for i in outputs.keys()]
elapsed_list = [per_prompt_elapsed[i] for i in outputs.keys() if i in per_prompt_elapsed]
distributions = {
    "prompt_tokens": _dist(prompt_len_list),
    "output_tokens": _dist(out_lens),
    "per_prompt_elapsed_s": _dist(elapsed_list),
}

# Per-dataset metrics breakdown (average output length and count).
per_dataset = {t: {"n": 0, "out_tokens": 0} for t in TASKS}
for idx in outputs:
    t = task_by_idx[idx]
    per_dataset.setdefault(t, {"n": 0, "out_tokens": 0})
    per_dataset[t]["n"] += 1
    per_dataset[t]["out_tokens"] += len(outputs[idx])

print(f"[RESULT] cell={args.cell} K={args.K} tps={tps:.2f} n={len(outputs)} "
      f"total={total} elapsed={elapsed:.1f}s", flush=True)
print(f"[RESULT] per_dataset: " + ", ".join(
    f"{t}={per_dataset[t]['n']}({per_dataset[t]['out_tokens']}tok)"
    for t in TASKS), flush=True)
if spec_metrics:
    print(f"[RESULT] AR={spec_metrics['draft_acceptance_rate']:.3f} "
          f"MAL={spec_metrics['mean_acceptance_length']:.2f} "
          f"per_pos={['{:.3f}'.format(r) for r in spec_metrics['per_position_acceptance_rate']]}",
          flush=True)

config_snapshot = {
    "cell": args.cell, "mode": args.mode, "K": args.K,
    "max_new": args.max_new,
    "slm": args.slm if args.mode in (
        "specsteer", "classical_sps_aug", "classical_sps_main", "scd",
        "b1_drafter_aug", "b1_drafter_main") else None,
    "main_context": args.main_context if args.mode in _MAIN_CTX_MODES else None,
    "asym_method": args.asym_method if args.mode in ("specsteer", "scd") else None,
    "delta_src": os.environ.get("ASYMSPEC_DELTA_SRC")
        if args.mode in ("specsteer", "scd") else None,
    "truncate_tokens": args.truncate_tokens
        if args.main_context == "truncate" else None,
    "draft_answers": args.draft_answers if args.mode == "b2_append" else None,
    "llm_path": LLM_PATH, "slm_path": SLM_PATH,
    "specsteer_beta": args.beta, "specsteer_gamma": args.gamma,
    "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
    "live_specsteer_model_sha": SHA_MODEL,
    "live_specsteer_sampler_sha": SHA_SAMPLER,
    "enforce_eager": args.enforce_eager,
    "max_cudagraph_size": args.max_cudagraph_size,
    "max_model_len": args.max_model_len,
    "tensor_parallel_size": args.tp,
    "cpu_offload_gb": args.cpu_offload_gb,
    "specsteer_main_max_model_len": args.specsteer_main_max_model_len,
    "gpu_memory_utilization": args.gpu_memory_utilization,
    "kv_cache_memory_bytes": kwargs.get("kv_cache_memory_bytes"),
    "yarn_factor": args.yarn_factor,
    "yarn_original_max": args.yarn_original_max if args.yarn_factor is not None else None,
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
        "per_dataset": per_dataset,
        "per_prompt_elapsed_s": {str(k): v for k, v in per_prompt_elapsed.items()},
        "prompt_token_lens": {str(k): v for k, v in prompt_lens.items()},
        "config": config_snapshot,
    }, f)
print(f"[saved] metrics → {args.out}", flush=True)

os.makedirs(os.path.dirname(args.responses) or ".", exist_ok=True)
with open(args.responses, "w") as f:
    for idx, toks in outputs.items():
        text = tok.decode(toks, skip_special_tokens=True)
        f.write(json.dumps({
            "idx": idx, "_id": sid_by_idx[idx],
            "dataset": task_by_idx[idx],
            "gold_answers": gold_by_idx[idx],
            "response": text,
        }, ensure_ascii=False) + "\n")
print(f"[saved] responses → {args.responses}", flush=True)
