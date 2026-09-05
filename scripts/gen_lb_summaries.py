"""Generate Qwen3-32B summaries for LongBench Multi-doc QA passages.

Reads:  data/longbench/raw/{hotpotqa,2wikimqa,musique}.jsonl
Writes: data/longbench/summaries.jsonl

One summary per sample (no k-variants — LongBench doesn't have multi-turn).
Uses Qwen3-32B as summarizer (self-distillation; same model as benchmark
target for paper-internal consistency). Prompt is faithful-summarizer
instruction asking for ≤300-word dense summary preserving entities/facts.

Output JSONL schema (one line per sample):
  {"_id": "<id>", "dataset": "<task>", "n_input_tokens": <int>,
   "n_summary_tokens": <int>, "summary": "<text>"}

Run on 1 GPU, vLLM batched, ~10 min total for 600 samples.
"""
from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from paths import LLM_PATH, LB_RAW, LB_SUMMARIES

DATA_DIR = str(LB_RAW)
OUT_DIR = str(LB_SUMMARIES.parent)
TASKS = ["hotpotqa", "2wikimqa", "musique"]

SUMMARY_INSTRUCTION = (
    "You are a faithful passage summarizer. The user will provide multiple "
    "Wikipedia-style passages. Produce a COMPREHENSIVE, DENSE summary "
    "(\u2264900 words) that preserves:\n"
    "  1. EVERY named entity (people, places, organizations, works, dates) "
    "and its key attributes (birthplace, death date, role, founding year, "
    "etc.)\n"
    "  2. ALL factual claims, numbers, events, and relationships between "
    "entities (e.g., 'X is the spouse of Y born in Z', 'X directed Y in "
    "year Z')\n"
    "  3. Specific details that may serve as answers to factual questions "
    "(nicknames, alternative names, predecessors, successors, locations)\n"
    "Use plain text. No headers. No bullet points. Write as cohesive "
    "paragraphs. Do not invent information not in the passages. Be exhaustive "
    "rather than selective \u2014 prefer including more facts even if the "
    "summary is at the upper word limit."
)


def build_prompt(passages: str, tok) -> list[int]:
    """Apply Qwen3 chat template (enable_thinking=False for clean output)."""
    msgs = [
        {"role": "system", "content": SUMMARY_INSTRUCTION},
        {"role": "user", "content": passages},
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    return tok.encode(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_summary_tokens", type=int, default=500)
    ap.add_argument("--max_model_len", type=int, default=24576)
    ap.add_argument("--tp", type=int, default=1,
                    help="Tensor parallel size.")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "summaries.jsonl"))
    ap.add_argument("--dry_run_n", type=int, default=0,
                    help="if >0, only process first N samples (smoke test)")
    args = ap.parse_args()

    # Load all samples
    samples: list[dict] = []
    for task in TASKS:
        path = os.path.join(DATA_DIR, f"{task}.jsonl")
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                ex["_task"] = task
                samples.append(ex)
    if args.dry_run_n > 0:
        samples = samples[:args.dry_run_n]
    print(f"[setup] {len(samples)} samples loaded across {len(TASKS)} tasks",
          flush=True)

    # Skip already-cached entries
    done_ids: set[tuple[str, str]] = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                d = json.loads(line)
                done_ids.add((d["dataset"], d["_id"]))
        print(f"[setup] cache exists with {len(done_ids)} entries; skipping",
              flush=True)
    samples = [s for s in samples
               if (s["_task"], s["_id"]) not in done_ids]
    if not samples:
        print("[done] all samples already cached", flush=True)
        return
    print(f"[setup] {len(samples)} samples to generate", flush=True)

    # Build prompts
    tok = AutoTokenizer.from_pretrained(LLM_PATH)
    prompts: list[dict] = []
    skipped = 0
    for s in samples:
        ids = build_prompt(s["context"], tok)
        # Reserve space for summary tokens
        if len(ids) > args.max_model_len - args.max_summary_tokens - 16:
            skipped += 1
            continue
        prompts.append({
            "_id": s["_id"], "_task": s["_task"],
            "n_in": len(ids), "prompt_token_ids": ids,
        })
    if skipped:
        print(f"[setup] {skipped} samples skipped (input too long)", flush=True)

    # vLLM with same compilation config as benchmark (avoids fuse_norm_quant
    # crash; see scripts/bench_multichallenge.py rationale).
    compilation_cfg = {
        "custom_ops": ["none", "+rms_norm"],
        "pass_config": {
            "fuse_norm_quant": False, "fuse_act_quant": False,
            "fuse_attn_quant": False, "enable_sp": False,
            "fuse_gemm_comms": False, "fuse_allreduce_rms": False,
        },
    }
    print(f"[setup] loading {LLM_PATH}...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model=LLM_PATH, dtype="bfloat16", trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len, gpu_memory_utilization=0.85,
        enforce_eager=False, disable_log_stats=False,
        compilation_config=compilation_cfg,
    )
    print(f"[setup] LLM loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0, max_tokens=args.max_summary_tokens)
    print(f"[run] generating {len(prompts)} summaries...", flush=True)
    t0 = time.perf_counter()
    outs = llm.generate(
        [{"prompt_token_ids": p["prompt_token_ids"]} for p in prompts],
        sp, use_tqdm=True,
    )
    elapsed = time.perf_counter() - t0
    total_out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[run] done in {elapsed:.1f}s "
          f"({total_out_toks} out tokens, {total_out_toks/elapsed:.1f} tps)",
          flush=True)

    # Write JSONL (append mode for resume safety)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        for p, o in zip(prompts, outs):
            text = o.outputs[0].text.strip()
            n_out = len(o.outputs[0].token_ids)
            f.write(json.dumps({
                "_id": p["_id"], "dataset": p["_task"],
                "n_input_tokens": p["n_in"], "n_summary_tokens": n_out,
                "summary": text,
            }, ensure_ascii=False) + "\n")
    print(f"[done] appended {len(prompts)} summaries to {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
