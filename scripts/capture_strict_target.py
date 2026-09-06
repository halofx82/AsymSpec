#!/usr/bin/env python3
"""Capture token IDs from one standalone or strict-target local run.

Run this script once per mode. It deliberately never loads the standalone and
AsymSpec engines in one process, so the output files can be compared by
``scripts/check_strict_target.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams


VERIFIER = "Qwen/Qwen3.8-27B"
DRAFTER = "Qwen/Qwen3.5-4B"


def load_messages(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("--messages must contain a non-empty JSON messages array")
    if not all(isinstance(message, dict) for message in value):
        raise ValueError("--messages must be an array of message objects")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("target", "strict-target"), required=True)
    parser.add_argument("--messages", type=Path, required=True,
                        help="JSON array passed unchanged to the Qwen chat template")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=2,
                        help="SpecSteer K for --mode strict-target")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--main-max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.num_speculative_tokens <= 0:
        parser.error("--num-speculative-tokens must be positive")

    # The strict run deliberately uses the same rendered prompt on both paths:
    # this isolates target verification from context-compression differences.
    if args.mode == "strict-target":
        os.environ["ASYMSPEC_METHOD"] = "strict_target"

    common = dict(
        model=VERIFIER,
        # The Python LLM API has no ``language_model_only`` keyword in vLLM
        # 0.28. ``generate`` selects the text-generation runner; no image
        # inputs are supplied by this text-only capture helper.
        runner="generate",
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    if args.mode == "strict-target":
        llm = LLM(
            **common,
            speculative_config={
                "method": "specsteer",
                "model": DRAFTER,
                "num_speculative_tokens": args.num_speculative_tokens,
                "draft_tensor_parallel_size": args.tp,
                "specsteer_beta": 1.0,
                "specsteer_gamma": 0.5,
                "specsteer_main_max_model_len": args.main_max_model_len,
            },
        )
    else:
        llm = LLM(**common)

    tokenizer = llm.get_tokenizer()
    messages = load_messages(args.messages)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if (args.mode == "strict-target"
            and len(prompt_ids) + args.max_tokens + args.num_speculative_tokens
            > args.main_max_model_len):
        parser.error("prompt + completion + K exceeds --main-max-model-len")
    sampling_kwargs = {"temperature": 0, "max_tokens": args.max_tokens}
    if args.mode == "strict-target":
        sampling_kwargs["extra_args"] = {"specsteer_aug_prompt_ids": prompt_ids}
    result = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(**sampling_kwargs),
        use_tqdm=False,
    )[0]
    output = result.outputs[0]
    capture = {
        "mode": args.mode,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": list(output.token_ids),
        "generated_text": output.text,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(capture, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "prompt_tokens": len(prompt_ids),
                      "generated_tokens": len(output.token_ids)}, indent=2))


if __name__ == "__main__":
    main()
