"""GAIA bench for VL drafter + text verifier (mirrors bench_mathvista.py cfg_ss).

Subset: GAIA validation 2023, file_name endswith {.png,.jpg,.jpeg}
        → 10 samples total (8 png + 2 jpg across L1/L2/L3).
        (Audio samples explicitly excluded per user request.)

Cells:
  --cfg b   B1_aug_vl    : Qwen3-VL-2B alone, sees image + question
  --cfg c   B1_main_q    : Qwen3-32B alone, sees question ONLY (no image)
  --cfg ss  SpecSteer    : verifier=Qwen3-32B (q only), drafter=Qwen3-VL (img+q)

Answer extraction: GAIA scoring is exact case-insensitive match after light
normalization (mirrors run_gaia_l1_web.gaia_score).
"""
from __future__ import annotations
import argparse, io, json, os, re, sys, time
import pandas as pd
from PIL import Image

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from paths import get_hf_token, gaia_validation_dir  # noqa: E402

if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = get_hf_token() or ""

VL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
TEXT_PATH = "Qwen/Qwen3-32B"
OUT_DIR = f"{_REPO}/outputs/gaia_vl"
MAX_OUT_TOKENS = 1024
COT_SUFFIX = "\nThink step by step, then provide the final answer after 'Answer:'."


# ---------- sample loading ----------
def load_samples():
    """Load the 10 image-bearing non-audio GAIA validation samples."""
    base = str(gaia_validation_dir())
    rows = []
    for L in [1, 2, 3]:
        df = pd.read_parquet(f"{base}/metadata.level{L}.parquet")
        df["level"] = L
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["ext"] = df["file_name"].fillna("").str.extract(
        r"\.([a-zA-Z0-9]+)$")[0].str.lower()
    img = df[df["ext"].isin(["png", "jpg", "jpeg"])]
    out = []
    for _, r in img.iterrows():
        fp = os.path.join(base, r["file_name"])
        if not os.path.exists(fp):
            print(f"  [skip] missing file: {fp}")
            continue
        try:
            image = Image.open(fp).convert("RGB")
        except Exception as e:
            print(f"  [skip] cannot open {fp}: {e}")
            continue
        out.append({
            "task_id": r["task_id"],
            "level": int(r["level"]),
            "file_name": r["file_name"],
            "ext": r["ext"],
            "question": r["Question"],
            "answer": r["Final answer"],
            "image": image,
        })
    print(f"[load] GAIA image samples n={len(out)}", flush=True)
    return out


# ---------- answer scoring (mirror of run_gaia_l1_web.gaia_score) ----------
def normalize_gaia(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:!?'\"")
    return s


def gaia_score(pred: str, gold: str) -> bool:
    """Case-insensitive substring or exact match after normalization."""
    np_pred = normalize_gaia(pred)
    np_gold = normalize_gaia(gold)
    if not np_gold:
        return False
    return np_gold == np_pred or np_gold in np_pred


def extract_answer(text: str) -> str:
    """Extract the model's stated answer after 'Answer:'.  Falls back to last line."""
    if not text:
        return ""
    m = re.search(r"Answer\s*:?\s*\*{0,2}([^\n*]+?)\*{0,2}\s*$",
                  text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip().rstrip(".")
    m = re.search(r"final\s*answer\s*:?\s*\*{0,2}([^\n*]+?)\*{0,2}",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


# ---------- tokenizer helpers ----------
_VERIFIER_TOK = None
def _verifier_tok():
    global _VERIFIER_TOK
    if _VERIFIER_TOK is None:
        from transformers import AutoTokenizer
        _VERIFIER_TOK = AutoTokenizer.from_pretrained(TEXT_PATH,
                                                      trust_remote_code=True)
    return _VERIFIER_TOK


CAPTION_PATH = f"{OUT_DIR}/captions.json"


def _load_captions() -> dict:
    """Load per-task_id captions if available; empty dict otherwise."""
    if os.path.exists(CAPTION_PATH):
        return json.load(open(CAPTION_PATH))
    return {}


def build_main_user_text(s: dict, captions: dict | None = None) -> str:
    """Verifier sees caption (if available) + question.
    No caption → blind verifier (degenerate; AR collapses to 0)."""
    cap = (captions or {}).get(s["task_id"], "")
    parts = []
    if cap.strip():
        parts.append(f"Image description: {cap}")
    parts.append(s["question"] + COT_SUFFIX)
    return "\n\n".join(parts)


def build_main_prompt(s: dict, captions: dict | None = None) -> str:
    msgs = [{"role": "user", "content": build_main_user_text(s, captions)}]
    return _verifier_tok().apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)


# ---------- save ----------
def save_results(cfg, samples, outs, out_path, spec_metrics=None):
    n_ok = sum(o["ok"] for o in outs)
    rec = {
        "cfg": cfg,
        "n_total": len(outs),
        "n_correct": n_ok,
        "accuracy_pct": 100.0 * n_ok / len(outs) if outs else 0.0,
        "spec_metrics": spec_metrics,
        "results": [
            {"task_id": s["task_id"], "level": s["level"],
             "question": s["question"], "gold": s["answer"],
             "pred": o["pred"], "ans": o["ans"], "ok": o["ok"]}
            for s, o in zip(samples[:len(outs)], outs)
        ],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    print(f"\n[saved] {cfg} → {out_path}  acc={rec['accuracy_pct']:.1f}%")


# ---------- CFG CAP: Build captions for all images using VL-2B alone ----------
def cfg_cap(samples, out_path):
    """Generate one caption per image with VL-2B and save to CAPTION_PATH.
    No accuracy reported — this is a setup phase, not an eval cell."""
    print("\n=== CFG CAP: Qwen3-VL-2B caption generation ===", flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    llm = LLM(model=VL_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=32768, gpu_memory_utilization=0.85,
              enforce_eager=True)
    proc = AutoProcessor.from_pretrained(VL_PATH, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=512)
    captions: dict[str, str] = _load_captions()
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        if s["task_id"] in captions:
            print(f"  [{i+1}/{len(samples)}] cached")
            continue
        cap_prompt = ("Describe this image in as much detail as possible, "
                      "including any text, numbers, diagrams, code, charts, "
                      "tables, or notation. Be exhaustive — every detail "
                      "should be in the caption.")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": cap_prompt}]}]
        ptxt = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
        try:
            r = llm.generate([{
                "prompt": ptxt,
                "multi_modal_data": {"image": s["image"]},
            }], sp, use_tqdm=False)
            captions[s["task_id"]] = r[0].outputs[0].text.strip()
            print(f"  [{i+1}/{len(samples)}] L{s['level']} cap_len={len(captions[s['task_id']])}")
        except Exception as e:
            print(f"  [{i+1}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            captions[s["task_id"]] = ""
    os.makedirs(os.path.dirname(CAPTION_PATH), exist_ok=True)
    with open(CAPTION_PATH, "w") as f:
        json.dump(captions, f, indent=2)
    el = time.perf_counter() - t0
    print(f"\n[CFG CAP] saved {len(captions)} captions → {CAPTION_PATH}  elapsed={el:.1f}s")


# ---------- CFG B: VL drafter alone ----------
def cfg_b_aug_vl(samples, out_path):
    print("\n=== CFG B: Qwen3-VL-2B alone (image+question) ===", flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    llm = LLM(model=VL_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=32768, gpu_memory_utilization=0.85,
              enforce_eager=True)
    proc = AutoProcessor.from_pretrained(VL_PATH, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUT_TOKENS)
    outs = []
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": s["question"] + COT_SUFFIX}]}]
        ptxt = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=False)
        try:
            r = llm.generate([{
                "prompt": ptxt,
                "multi_modal_data": {"image": s["image"]},
            }], sp, use_tqdm=False)
            ans = r[0].outputs[0].text.strip()
            pred = extract_answer(ans)
            ok = gaia_score(pred, s["answer"])
            outs.append({"ans": ans, "pred": pred, "ok": ok})
            print(f"  [{i+1}/{len(samples)}] L{s['level']} ok={ok}  pred={pred!r}  gold={s['answer']!r}")
        except Exception as e:
            print(f"  [{i+1}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            outs.append({"ans": f"FAIL:{type(e).__name__}", "pred": "", "ok": False})
    el = time.perf_counter() - t0
    print(f"\n[CFG B] elapsed={el:.1f}s")
    save_results("b1_aug_vl", samples, outs, out_path)


# ---------- CFG C: text verifier alone, blind to image ----------
def cfg_c_main_q(samples, out_path):
    print("\n=== CFG C: Qwen3-32B alone (caption+question; blind verifier if no caption) ===", flush=True)
    from vllm import LLM, SamplingParams
    captions = _load_captions()
    print(f"[setup] loaded {len(captions)} captions")
    llm = LLM(model=TEXT_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=32768, gpu_memory_utilization=0.85,
              enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=MAX_OUT_TOKENS)
    outs = []
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            r = llm.generate([{"prompt": build_main_prompt(s, captions)}], sp, use_tqdm=False)
            ans = r[0].outputs[0].text.strip()
            pred = extract_answer(ans)
            ok = gaia_score(pred, s["answer"])
            outs.append({"ans": ans, "pred": pred, "ok": ok})
            print(f"  [{i+1}/{len(samples)}] L{s['level']} ok={ok}  pred={pred!r}  gold={s['answer']!r}")
        except Exception as e:
            print(f"  [{i+1}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            outs.append({"ans": f"FAIL:{type(e).__name__}", "pred": "", "ok": False})
    el = time.perf_counter() - t0
    print(f"\n[CFG C] elapsed={el:.1f}s")
    save_results("b1_main_q", samples, outs, out_path)


# ---------- CFG SS: VL drafter (image) + text verifier (blind) ----------
def cfg_ss(samples, out_path, K=2, beta=1.0, gamma=0.5,
           asym_method="jsd"):
    print(f"\n=== CFG SS: VL drafter + text verifier  K={K} β={beta} γ={gamma} method={asym_method} ===", flush=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    import vllm.v1.spec_decode.metrics as _sdm

    os.environ["ASYMSPEC_METHOD"] = asym_method
    os.environ["ASYMSPEC_DELTA_SRC"] = "ours"

    SPEC = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0,
            "per_pos": None}
    _orig = _sdm.SpecDecodingStats.observe_draft

    def _capture(self, num_draft_tokens, num_accepted_tokens):
        SPEC["drafts"] += 1
        SPEC["draft_tokens"] += num_draft_tokens
        SPEC["accepted_tokens"] += num_accepted_tokens
        if SPEC["per_pos"] is None:
            SPEC["per_pos"] = [0] * self.num_spec_tokens
        for j in range(num_accepted_tokens):
            if j < len(SPEC["per_pos"]):
                SPEC["per_pos"][j] += 1
        return _orig(self, num_draft_tokens, num_accepted_tokens)
    _sdm.SpecDecodingStats.observe_draft = _capture

    proc = AutoProcessor.from_pretrained(VL_PATH, trust_remote_code=True)
    captions = _load_captions()
    print(f"[setup] loaded {len(captions)} captions")

    aug_prepped = []
    for s in samples:
        # Drafter prompt: same caption+Q as verifier + image (info superset).
        aug_user_text = build_main_user_text(s, captions)
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": aug_user_text}]}]
        aug_text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        po = proc(text=[aug_text], images=[s["image"]], return_tensors="pt",
                  padding=False)
        aug_prepped.append({
            "aug_ids": po["input_ids"][0].tolist(),
            "pixel_values": po["pixel_values"],
            "image_grid_thw": po["image_grid_thw"],
        })

    llm = LLM(model=TEXT_PATH, dtype="bfloat16", trust_remote_code=True,
              max_model_len=32768, gpu_memory_utilization=0.85,
              enforce_eager=True,
              speculative_config={
                  "method": "specsteer", "model": VL_PATH,
                  "num_speculative_tokens": K,
                  "specsteer_beta": beta, "specsteer_gamma": gamma,
              })
    print("[CFG SS] LLM init OK")

    sp_base = dict(temperature=0, max_tokens=MAX_OUT_TOKENS)
    AUG_MAX = 32768 - MAX_OUT_TOKENS - 64
    outs = []
    n_skip = 0
    t0 = time.perf_counter()
    for i, (s, prep) in enumerate(zip(samples, aug_prepped)):
        if len(prep["aug_ids"]) > AUG_MAX:
            print(f"  [{i+1}] SKIP aug_len={len(prep['aug_ids'])} > {AUG_MAX}")
            outs.append({"ans": "SKIP", "pred": "", "ok": False})
            n_skip += 1
            continue
        sp_kw = dict(sp_base, extra_args={
            "specsteer_aug_prompt_ids": prep["aug_ids"],
            "specsteer_aug_pixel_values": prep["pixel_values"],
            "specsteer_aug_image_grid_thw": prep["image_grid_thw"],
        })
        try:
            r = llm.generate([{"prompt": build_main_prompt(s, captions)}],
                             SamplingParams(**sp_kw), use_tqdm=False)
            ans = r[0].outputs[0].text.strip()
            pred = extract_answer(ans)
            ok = gaia_score(pred, s["answer"])
            outs.append({"ans": ans, "pred": pred, "ok": ok})
            print(f"  [{i+1}/{len(samples)}] L{s['level']} ok={ok}  pred={pred!r}  gold={s['answer']!r}")
        except Exception as e:
            print(f"  [{i+1}] FAIL: {type(e).__name__}: {str(e)[:200]}")
            outs.append({"ans": f"FAIL:{type(e).__name__}", "pred": "", "ok": False})
            break

    el = time.perf_counter() - t0
    metrics = {
        "K": K, "beta": beta, "gamma": gamma, "asym_method": asym_method,
        "elapsed": el,
        "AR": SPEC["accepted_tokens"] / SPEC["draft_tokens"] if SPEC["draft_tokens"] else 0,
        "MAL": 1 + SPEC["accepted_tokens"] / SPEC["drafts"] if SPEC["drafts"] else 1,
        "drafts": SPEC["drafts"],
        "draft_tokens": SPEC["draft_tokens"],
        "accepted_tokens": SPEC["accepted_tokens"],
        "per_position_acceptance_rate":
            [c / SPEC["drafts"] for c in (SPEC["per_pos"] or [])],
        "n_skipped": n_skip,
    }
    print(f"\n[SS metrics] {json.dumps(metrics, indent=2)}")
    save_results(f"ss_K{K}", samples, outs, out_path, spec_metrics=metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", choices=["b", "c", "ss", "cap"], required=True)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--asym_method", default="jsd")
    ap.add_argument("--tag", default="run1")
    args = ap.parse_args()

    samples = load_samples()
    out_path = os.path.join(
        OUT_DIR,
        f"{args.tag}_n{len(samples)}_{args.cfg}"
        + (f"_K{args.K}" if args.cfg == "ss" else "")
        + ".json")

    if args.cfg == "cap":
        cfg_cap(samples, out_path)
    elif args.cfg == "b":
        cfg_b_aug_vl(samples, out_path)
    elif args.cfg == "c":
        cfg_c_main_q(samples, out_path)
    else:
        cfg_ss(samples, out_path, K=args.K, beta=args.beta, gamma=args.gamma,
               asym_method=args.asym_method)


if __name__ == "__main__":
    main()
