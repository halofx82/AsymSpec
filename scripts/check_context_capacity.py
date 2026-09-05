#!/usr/bin/env python3
"""Run one real near-limit AsymSpec prefill against both context views."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams

from paths import LLM_MODEL, slm_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--specsteer-main-max-model-len", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--token-id", type=int, default=42)
    ap.add_argument("--full-prompt-len", type=int)
    ap.add_argument("--main-prompt-len", type=int, default=128)
    ap.add_argument("--out")
    args = ap.parse_args()

    full_len = args.full_prompt_len
    if full_len is None:
        full_len = args.max_model_len - args.max_new - 16
    if full_len <= 0:
        ap.error("full prompt length must be positive")

    llm = LLM(
        model=LLM_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        cpu_offload_gb=0,
        max_num_batched_tokens=min(args.max_model_len, 16384),
        gpu_memory_utilization=0.97,
        enforce_eager=True,
        speculative_config={
            "method": "specsteer",
            "model": slm_model("4B"),
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": args.tp,
            "specsteer_beta": 1.0,
            "specsteer_gamma": 0.5,
            "specsteer_main_max_model_len": args.specsteer_main_max_model_len,
        },
    )
    main_ids = [args.token_id] * args.main_prompt_len
    full_ids = [args.token_id] * full_len
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_new,
        extra_args={"specsteer_aug_prompt_ids": full_ids},
    )
    start = time.perf_counter()
    result = llm.generate(
        [{"prompt_token_ids": main_ids}], sampling, use_tqdm=False)[0]
    elapsed = time.perf_counter() - start
    report = {
        "max_model_len": args.max_model_len,
        "specsteer_main_max_model_len": args.specsteer_main_max_model_len,
        "full_prompt_tokens": full_len,
        "main_prompt_tokens": len(main_ids),
        "requested_output_tokens": args.max_new,
        "output_tokens": len(result.outputs[0].token_ids),
        "elapsed_s": elapsed,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
