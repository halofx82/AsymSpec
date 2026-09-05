"""API-Bank L1+L2 per-call bench with BM25 retrieval.

Mirrors `bench_lb.py` / `bench_multichallenge.py` (sync LLM, single-pass per
sample).
Each "sample" comes from API-Bank's `Sample.from_chat_history`: every API
call position in a dialog generates two samples (one for the API call
prediction, one for the AI text response).

Modes:
  b1_aug              main = full 53 APIs, no SS
  b1_main_q           main = no APIs (closed-book floor, just the question)
  b1_main_topK        main = top-K retrieved APIs (no SS)
  classical_X_topK    main + drafter = top-K retrieved APIs (draft-model SpS)
  ss_X_topK           main = top-K retrieved, drafter = full 53 (SpecSteer)

Retrieval: BM25 over (api_name + description + parameter keys+descs) using
 the FIRST user message as query. Top-K fixed for the whole dialog.
K_retrieve default = 5 (covers most dialogs' API needs).

Why top-K, not truncate: API specs are structured JSON, can't be truncated
 mid-token without breaking parser. Retrieval is the standard compression
 method for tool catalogs (cited: ToolBench's ToolSearcher mechanism, Qin
 et al. ICLR'24).

Prompt format (Qwen3 chat template):
  system: "You have access to the following tools:\\n{specs}\\n
            When you need to call a tool, output: [ToolName(p1=v1, ...)]"
  user:    {first user msg}
  assistant: {gold AI text} OR predict next [ToolCall(...)]
  ...

Output JSON: per-sample (sample_id, dialog_file, ground_truth, prediction,
            n_in_main, n_out, ...). Eval is OFFLINE via
 evaluator_by_json.py from the API-Bank repo.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LLM_PATH, slm_model, REPO_ROOT, APIBANK_DIR as _APIB_DIR

APIBANK_DIR = str(_APIB_DIR)
APIS_DIR = f"{APIBANK_DIR}/apis"

sys.path.insert(0, APIBANK_DIR)
sys.path.insert(0, f"{APIBANK_DIR}/..")  # for tool_manager imports


def load_all_apis() -> list[dict]:
    """Load all 53 APIs via importlib. Returns list of {name, description,
    input_parameters, output_parameters}."""
    import importlib
    import importlib.util
    apis = []
    skip = {"__init__.py", "api.py"}
    saved_cwd = os.getcwd()
    os.chdir(APIBANK_DIR)
    try:
        # Make sure 'apis' is importable as a package
        from apis.api import API
        skipped_imports = []
        for fname in sorted(os.listdir("apis")):
            if not fname.endswith(".py") or fname in skip:
                continue
            mod_name = fname[:-3]
            try:
                mod = importlib.import_module(f"apis.{mod_name}")
            except (ImportError, ModuleNotFoundError) as e:
                # Some APIs need optional deps (googletrans etc.) we don't
                # need to actually CALL the API, just expose its spec — but
                # the spec is only available via the imported class. Skip.
                skipped_imports.append((mod_name, str(e)))
                continue
            for attr in dir(mod):
                cls = getattr(mod, attr)
                if isinstance(cls, type) and issubclass(cls, API) and cls is not API:
                    apis.append({
                        "name": cls.__name__,
                        "description": getattr(cls, "description", ""),
                        "input_parameters": dict(getattr(cls, "input_parameters", {})),
                        "output_parameters": dict(getattr(cls, "output_parameters", {})),
                    })
        if skipped_imports:
            print(f"[load_all_apis] skipped {len(skipped_imports)} APIs due to missing deps: "
                  f"{[s[0] for s in skipped_imports]}", flush=True)
    finally:
        os.chdir(saved_cwd)
    return apis


def api_to_doc(api: dict) -> str:
    """Flatten one API spec into a tokenizable retrieval document."""
    parts = [api["name"], api["description"]]
    for pname, pinfo in api["input_parameters"].items():
        parts.append(pname)
        if isinstance(pinfo, dict):
            parts.append(pinfo.get("description", ""))
    return " ".join(p for p in parts if p)


def api_to_spec_json(api: dict) -> str:
    """Render one API spec as JSON (what the model sees in the prompt)."""
    spec = {
        "name": api["name"],
        "description": api["description"],
        "input_parameters": api["input_parameters"],
        "output_parameters": api["output_parameters"],
    }
    return json.dumps(spec, ensure_ascii=False)


def api_to_name_sig(api: dict) -> str:
    """Method A: name + signature only (no description).
    e.g., '[Tool] GetUserToken(username, password)'  ~11 tok/api"""
    params = ", ".join(api["input_parameters"].keys())
    return f"[Tool] {api['name']}({params})"


def api_to_name_sig_desc(api: dict) -> str:
    """Method B: name + signature + 1-line description (~25 tok/api)."""
    params = ", ".join(api["input_parameters"].keys())
    desc = api["description"].split("\n")[0].strip()
    if len(desc) > 100:
        desc = desc[:97] + "..."
    return f"[Tool] {api['name']}({params}) — {desc}"


def build_bm25(apis: list[dict]):
    from rank_bm25 import BM25Okapi
    corpus = [api_to_doc(a).lower().split() for a in apis]
    bm25 = BM25Okapi(corpus)
    return bm25


def retrieve_topk(bm25, apis: list[dict], query: str, k: int) -> list[dict]:
    tokenized_q = query.lower().split()
    scores = bm25.get_scores(tokenized_q)
    import numpy as np
    top_idx = np.argsort(scores)[::-1][:k]
    return [apis[i] for i in top_idx]


def load_dialogs() -> list[dict]:
    """Load all L1 + L2 dialogs. Returns [{file, dialog: [{role, ...}, ...]}]."""
    dialogs = []
    for sub in ["level-1-given-desc", "level-2-toolsearcher"]:
        d = os.path.join(APIBANK_DIR, "lv1-lv2-samples", sub)
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(d, fname)
            turns = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        turns.append(json.loads(line))
            if turns:
                dialogs.append({"file": fname, "level": sub, "dialog": turns})
    return dialogs


def dialog_to_samples(dialog: list[dict]) -> list[tuple[list[dict], dict]]:
    """Mirror API-Bank's Sample.from_chat_history.

    For every API position i:
      - sample (history=dialog[:i], gt=dialog[i])     # predict API call
      - sample (history=dialog[:i+1], gt=dialog[i+1]) # predict AI response
    """
    samples = []
    for i, item in enumerate(dialog):
        if item["role"] == "API":
            # sample for predicting the API call itself
            samples.append((list(dialog[:i]), dict(item)))
            # sample for predicting the AI response after the API result
            if i + 1 < len(dialog):
                samples.append((list(dialog[:i + 1]), dict(dialog[i + 1])))
    return samples


def render_history_messages(history: list[dict]) -> list[dict]:
    """Convert API-Bank dialog turns to OpenAI-style chat messages.

    Maps:
      User       → user
      AI         → assistant
      API (call) → assistant text "[ApiName(...)]"
      API (result) → tool message "{api_name} result: {output}"
    For simplicity we collapse API call+result into one assistant turn that
    SAYS the call, and a follow-up tool message with the result.
    """
    msgs = []
    for t in history:
        role = t["role"]
        if role == "User":
            msgs.append({"role": "user", "content": t["text"]})
        elif role == "AI":
            msgs.append({"role": "assistant", "content": t["text"]})
        elif role == "API":
            # the assistant said the call
            param_str = ", ".join(f"{k}={json.dumps(v)}"
                                  for k, v in t["param_dict"].items())
            call_text = f"[{t['api_name']}({param_str})]"
            msgs.append({"role": "assistant", "content": call_text})
            # then the tool returned
            output = t["result"].get("output", "") if isinstance(t.get("result"), dict) else ""
            msgs.append({
                "role": "tool",
                "content": f"{t['api_name']} returned: {json.dumps(output, ensure_ascii=False)}",
            })
    return msgs


SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can call tools to solve user "
    "requests. You have access to the following tools:\n"
    "{specs}\n\n"
    "When you need to call a tool, output the call in the format: "
    "[ToolName(param1=value1, param2=value2)]\n"
    "Output ONLY the tool call (in brackets) when calling a tool. "
    "When responding with text, write the text directly."
)


def build_prompt_messages(specs_text: str, history: list[dict]) -> list[dict]:
    msgs = [{"role": "system",
             "content": SYSTEM_TEMPLATE.format(specs=specs_text)}]
    msgs.extend(render_history_messages(history))
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["b1_aug", "b1_main_q", "b1_main_topK",
                             "classical_sps_aug", "classical_sps_main",
                             "specsteer",
                             "b1_drafter_aug", "b1_drafter_main", "scd"],
                    help="b1_drafter_aug=drafter(SLM) alone on the FULL API "
                         "spec (B1); "
                         "classical_sps_main=classical SD with verifier+drafter "
                         "both on the COMPRESSED API spec (fair-cost "
                         "compressed-SD baseline); "
                         "scd=Speculative Contrastive Decoding baseline, "
                         "expert(32B)/amateur(SLM) on the SAME "
                         "compressed main spec (B3)")
    ap.add_argument("--asym_method", default="jsd",
                    choices=["gamma_rule", "cma", "jsd", "jsd_pos",
                             "cma_vnorm", "cma_hbase"])
    ap.add_argument("--delta_src", default="ours",
                    choices=["ours", "raw_aug", "scd"],
                    help="δ-source ablation (specsteer mode); forced to 'scd' "
                         "when --mode scd")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=2,
                    help="num_speculative_tokens; ignored for b1_*")
    ap.add_argument("--K_retrieve", type=int, default=5,
                    help="Top-K APIs to surface in main (when not full).")
    ap.add_argument("--max_new", type=int, default=256,
                    help="CANONICAL paper config: 256 (matches baseline ss_1.7B_K2).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature; 0.0 = greedy.")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p nucleus sampling; 1.0 = disabled.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed (only meaningful when temperature>0).")
    ap.add_argument("--slm", default="1.7B", choices=["0.6B", "1.7B", "4B"])
    ap.add_argument("--n", type=int, default=0,
                    help="If >0, limit to first N dialogs.")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--max_cudagraph_size", type=int, default=128)
    # API-Bank max prompt is ~7K (52-API spec + short history); 16384 is
    # plenty. Kept as default for headroom.
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    # Hierarchical compression args (mirror of BFCL bench API; no-op here
    # since API-Bank's per-call eval has no accumulating history per dialog).
    ap.add_argument("--main_context", default="truncate",
                    choices=["truncate", "summary_last_k"])
    ap.add_argument("--last_k", type=int, default=2)
    ap.add_argument("--early_summary_cap", type=int, default=200)
    ap.add_argument("--recent_trunc_n", type=int, default=1024)
    # Main-context compression choices.
    #   topK         → top-K retrieved JSON specs (ablation)
    #   name_sig     → all 52 APIs as "[Tool] Name(p1, p2)"  (~584 tok) ← Method A, PAPER CANONICAL
    #   name_sig_desc→ all 52 APIs as name+sig+1-line desc    (~1322 tok)
    # name_sig is the paper configuration (API-Bank Method A, roughly 10x
    # compression).
    ap.add_argument("--main_compression", default="name_sig",
                    choices=["topK", "name_sig", "name_sig_desc"])
    args = ap.parse_args()

    SLM_PATH = slm_model(args.slm)

    # Spec metrics monkey-patch (mirrors the other text benchmarks).
    import vllm.config.speculative as _sc
    _sc.SpeculativeConfig.verify_equal_vocab_size_if_draft_model = (
        lambda self: None
    )
    import vllm.v1.spec_decode.metrics as _sdm
    SPEC = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0,
            "per_pos": None, "K_seen": None}
    _orig = _sdm.SpecDecodingStats.observe_draft

    def _capture(self, *args, **kwargs):
        # vLLM signature changed: observe_draft now takes
        # (num_draft_tokens, num_accepted_tokens) — accept either positional
        # or keyword (supported runner versions may pass values by keyword).
        ndt = kwargs.get("num_draft_tokens", args[0] if args else 0)
        nat = kwargs.get("num_accepted_tokens", args[1] if len(args) > 1 else 0)
        SPEC["drafts"] += 1
        SPEC["draft_tokens"] += ndt
        SPEC["accepted_tokens"] += nat
        if SPEC["per_pos"] is None:
            SPEC["per_pos"] = [0] * self.num_spec_tokens
            SPEC["K_seen"] = self.num_spec_tokens
        for i in range(nat):
            if i < len(SPEC["per_pos"]):
                SPEC["per_pos"][i] += 1
        return _orig(self, *args, **kwargs)
    _sdm.SpecDecodingStats.observe_draft = _capture

    # Race fix: gpu_acquire returns when memory drops below 5GB, but the
    # CUDA driver's free-memory view lags by 5-30s. Poll until the assigned
    # physical GPU has >= 130GB free, or timeout (then proceed anyway).
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _cvd:
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
    TOK_PATH = SLM_PATH if args.mode in ("b1_drafter_aug", "b1_drafter_main") else LLM_PATH
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    print("[setup] loading APIs...", flush=True)
    all_apis = load_all_apis()
    print(f"[setup] loaded {len(all_apis)} APIs", flush=True)
    bm25 = build_bm25(all_apis)
    full_specs_text = "\n".join(api_to_spec_json(a) for a in all_apis)
    # Method A/B compressed-all-API spec (built once; one global string).
    if args.main_compression == "name_sig":
        compressed_all_text = "\n".join(api_to_name_sig(a) for a in all_apis)
    elif args.main_compression == "name_sig_desc":
        compressed_all_text = "\n".join(api_to_name_sig_desc(a) for a in all_apis)
    else:
        compressed_all_text = None  # unused when topK
    if compressed_all_text is not None:
        _ntok = len(tok.encode(compressed_all_text))
        print(f"[setup] main_compression={args.main_compression} "
              f"chars={len(compressed_all_text)} tokens={_ntok} "
              f"apis={len(all_apis)}", flush=True)

    print("[setup] loading dialogs...", flush=True)
    dialogs = load_dialogs()
    if args.n > 0:
        dialogs = dialogs[:args.n]
    print(f"[setup] {len(dialogs)} dialogs (L1={sum(1 for d in dialogs if 'level-1' in d['level'])} "
          f"L2={sum(1 for d in dialogs if 'level-2' in d['level'])})", flush=True)

    # Compilation config — mirror bench_lb settings exactly.
    compilation_cfg = {
        "custom_ops": ["none", "+rms_norm"],
        "pass_config": {
            "fuse_norm_quant": False, "fuse_act_quant": False,
            "fuse_attn_quant": False, "enable_sp": False,
            "fuse_gemm_comms": False, "fuse_allreduce_rms": False,
        },
    }
    if args.max_cudagraph_size < 512:
        _DEF = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
                112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200,
                208, 216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336,
                352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]
        compilation_cfg["cudagraph_capture_sizes"] = [
            s for s in _DEF if s <= args.max_cudagraph_size
        ]

    BASE_MODEL = SLM_PATH if args.mode in ("b1_drafter_aug", "b1_drafter_main") else LLM_PATH
    eng_kwargs = dict(
        model=BASE_MODEL, dtype="bfloat16", trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager, disable_log_stats=False,
        compilation_config=compilation_cfg,
    )
    if args.mode in ("specsteer", "scd"):
        eng_kwargs["speculative_config"] = {
            "method": "specsteer", "model": SLM_PATH,
            "num_speculative_tokens": args.K,
            "specsteer_beta": args.beta, "specsteer_gamma": args.gamma,
        }
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _asym_method_util import setup_asym_method
        setup_asym_method(args.asym_method, beta=args.beta)
        # scd forces SCD-style two-model δ; specsteer honors --delta_src.
        os.environ["ASYMSPEC_DELTA_SRC"] = (
            "scd" if args.mode == "scd" else args.delta_src)
        print(f"[setup] ASYMSPEC_METHOD={os.environ.get('ASYMSPEC_METHOD')} "
              f"ASYMSPEC_DELTA_SRC={os.environ['ASYMSPEC_DELTA_SRC']}",
              flush=True)
    elif args.mode in ("classical_sps_aug", "classical_sps_main"):
        # Same vLLM config; prompt content (full vs compressed) differs
        # in dialog_main_specs setup below.
        eng_kwargs["speculative_config"] = {
            "method": "draft_model", "model": SLM_PATH,
            "num_speculative_tokens": args.K,
        }
    print(f"[setup] loading LLM (K={args.K})...", flush=True)
    llm = LLM(**eng_kwargs)
    if args.mode in ("specsteer", "scd"):
        # Incremental base-model evaluation is enabled by default.
        print("[setup] SpecSteer Path B enabled", flush=True)

    # Build all samples
    samples = []  # list of (sample_id, dialog_file, history, gt, level, dialog_idx)
    for d_idx, d in enumerate(dialogs):
        for s_idx, (history, gt) in enumerate(dialog_to_samples(d["dialog"])):
            samples.append({
                "id": f"{d['file']}::s{s_idx}",
                "file": d["file"], "level": d["level"],
                "dialog_idx": d_idx,
                "history": history, "ground_truth": gt,
            })
    print(f"[setup] {len(samples)} per-call samples", flush=True)

    # Pre-compute retrieved spec text per dialog (once per dialog)
    print(f"[setup] computing top-{args.K_retrieve} retrieved specs per dialog...", flush=True)
    dialog_main_specs: dict[int, str] = {}
    dialog_aug_specs: dict[int, str] = {}
    for d_idx, d in enumerate(dialogs):
        # First user message is the query for retrieval
        first_user = next((t["text"] for t in d["dialog"] if t["role"] == "User"), "")
        topk = retrieve_topk(bm25, all_apis, first_user, args.K_retrieve)
        topk_text = "\n".join(api_to_spec_json(a) for a in topk)
        if args.mode in ("b1_aug", "b1_drafter_aug"):
            # b1_drafter_aug: SLM alone reads the FULL spec (= aug view).
            dialog_main_specs[d_idx] = full_specs_text
        elif args.mode == "b1_main_q":
            dialog_main_specs[d_idx] = ""  # closed-book
        elif args.mode in ("b1_main_topK", "classical_sps_aug", "classical_sps_main",
                            "specsteer", "scd", "b1_drafter_main"):
            # b1_drafter_main (B4): SLM alone on the COMPRESSED main spec.
            # Method A/B: replace topK retrieval with compressed-all-API view.
            # All 52 API names visible to verifier — only descriptions differ.
            if args.main_compression == "name_sig":
                dialog_main_specs[d_idx] = compressed_all_text
            elif args.main_compression == "name_sig_desc":
                dialog_main_specs[d_idx] = compressed_all_text
            else:
                dialog_main_specs[d_idx] = topk_text
        # aug always full (drafter sees all 53 in ss mode)
        dialog_aug_specs[d_idx] = full_specs_text

    # Warmup
    warmup_msgs = build_prompt_messages(dialog_main_specs[0],
                                         samples[0]["history"])
    warmup_text = tok.apply_chat_template(
        warmup_msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    warmup_ids = tok.encode(warmup_text)
    _ = llm.generate([{"prompt_token_ids": warmup_ids}],
                     SamplingParams(temperature=0, max_tokens=8),
                     use_tqdm=False)

    sp_base = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new,
    )
    if args.seed is not None:
        sp_base["seed"] = args.seed

    # Main loop — chunk samples by --bs (single-stream BS=1 by default).
    def _chunk(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    out_records = []
    t0 = time.perf_counter()
    n_done = 0
    for batch in _chunk(samples, args.bs):
        reqs = []
        sps = []
        for s in batch:
            d_idx = s["dialog_idx"]
            main_specs = dialog_main_specs[d_idx]
            msgs = build_prompt_messages(main_specs, s["history"])
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            main_ids = tok.encode(text)
            reqs.append({"prompt_token_ids": main_ids})

            sp_kw = dict(sp_base)
            if args.mode in ("specsteer", "scd"):
                if args.mode == "scd":
                    # SCD: amateur reads the SAME compressed main context.
                    aug_ids = list(main_ids)
                else:
                    aug_specs = dialog_aug_specs[d_idx]
                    aug_msgs = build_prompt_messages(aug_specs, s["history"])
                    aug_text = tok.apply_chat_template(
                        aug_msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                    aug_ids = tok.encode(aug_text)
                sp_kw["extra_args"] = {"specsteer_aug_prompt_ids": aug_ids}
            sps.append(SamplingParams(**sp_kw))

        outs = llm.generate(reqs, sps, use_tqdm=False)
        for s, o in zip(batch, outs):
            pred_text = o.outputs[0].text.strip()
            n_in = len(o.prompt_token_ids) if o.prompt_token_ids else None
            n_out = len(o.outputs[0].token_ids)
            out_records.append({
                "id": s["id"], "file": s["file"], "level": s["level"],
                "ground_truth": s["ground_truth"],
                "prediction": pred_text,
                "n_in": n_in, "n_out": n_out,
            })
            n_done += 1
        if n_done % 50 == 0 or n_done == len(samples):
            elapsed = time.perf_counter() - t0
            tot_out = sum(r["n_out"] for r in out_records)
            tps = tot_out / elapsed if elapsed else 0
            ar_str = ""
            if SPEC["draft_tokens"]:
                ar = SPEC["accepted_tokens"] / SPEC["draft_tokens"]
                mal = 1 + SPEC["accepted_tokens"] / SPEC["drafts"]
                ar_str = f" AR={ar:.3f} MAL={mal:.2f}"
            print(f"[{n_done}/{len(samples)}] tps={tps:.2f} elapsed={elapsed:.1f}s{ar_str}",
                  flush=True)

    elapsed = time.perf_counter() - t0
    total_out = sum(r["n_out"] for r in out_records)
    spec_metrics = None
    if SPEC["drafts"] > 0:
        n_d = SPEC["drafts"]; n_dt = SPEC["draft_tokens"]
        n_acc = SPEC["accepted_tokens"]
        spec_metrics = {
            "K": args.K, "num_drafts": n_d, "num_draft_tokens": n_dt,
            "num_accepted_tokens": n_acc,
            "draft_acceptance_rate": n_acc / n_dt if n_dt else None,
            "mean_acceptance_length": 1 + (n_acc / n_d),
            "per_position_acceptance_rate":
                [c / n_d for c in (SPEC["per_pos"] or [])],
        }

    print(f"\n[RESULT] cell={args.cell} K={args.K} mode={args.mode} "
          f"tps={total_out / elapsed:.2f} n={len(out_records)} "
          f"total={total_out} elapsed={elapsed:.1f}s", flush=True)
    if spec_metrics:
        print(f"[RESULT] AR={spec_metrics['draft_acceptance_rate']:.3f} "
              f"MAL={spec_metrics['mean_acceptance_length']:.2f}", flush=True)

    # Save metrics + responses
    out_obj = {
        "mode": args.mode, "K": args.K, "K_retrieve": args.K_retrieve,
        "cell": args.cell, "n_dialogs": len(dialogs),
        "n_samples": len(out_records),
        "total_out_tokens": total_out, "elapsed": elapsed,
        "tps": total_out / elapsed if elapsed else 0,
        "spec_metrics": spec_metrics,
        "config": vars(args),
        "delta_src_effective": os.environ.get("ASYMSPEC_DELTA_SRC")
            if args.mode in ("specsteer", "scd") else None,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_obj, f, default=str)
    print(f"[saved] metrics → {args.out}", flush=True)

    with open(args.responses, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
    print(f"[saved] responses → {args.responses}", flush=True)


if __name__ == "__main__":
    main()
