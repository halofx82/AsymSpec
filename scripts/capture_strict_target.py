#!/usr/bin/env python3
"""Capture token IDs from standalone or SpecSteer local runs.

Run this script once per mode. It deliberately never loads the standalone and
AsymSpec engines in one process, so the output files can be compared by
``scripts/check_strict_target.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer
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


def load_suite(path: Path) -> list[dict]:
    """Load a small, ordered chat-quality suite without changing messages."""
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("--suite must contain a non-empty JSON array")
    seen_ids: set[str] = set()
    for case in value:
        if not isinstance(case, dict):
            raise ValueError("every --suite entry must be an object")
        case_id = case.get("id")
        messages = case.get("messages")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("every --suite entry needs a unique non-empty id")
        if not isinstance(messages, list) or not messages or not all(
                isinstance(message, dict) for message in messages):
            raise ValueError(f"suite case {case_id!r} needs non-empty messages")
        if "max_tokens" in case and (not isinstance(case["max_tokens"], int)
                                     or case["max_tokens"] <= 0):
            raise ValueError(f"suite case {case_id!r} has invalid max_tokens")
        seen_ids.add(case_id)
    return value


def render_prompt_ids(tokenizer, messages: list[dict], chat_template: str) -> list[int]:
    """Render every mode with the verifier template and no visible reasoning.

    Qwen3.5 and Qwen3.8 share the checked vocabulary but may ship distinct
    template defaults.  Keeping this template explicit makes A/B/C/D a model
    comparison rather than a chat-template or thinking-mode comparison.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=chat_template, enable_thinking=False)
    return list(tokenizer.encode(prompt, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("target", "strict-target", "jsd", "draft"),
        required=True,
        help="target=27B, strict-target=target-authoritative SpecSteer, "
             "jsd=three-path SpecSteer, draft=standalone 4B",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--messages", type=Path,
                        help="JSON array passed unchanged to the Qwen chat template")
    source.add_argument("--suite", type=Path,
                        help="JSON array of {id, messages, max_tokens?}; runs sequentially "
                             "through one loaded model")
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
    is_specsteer = args.mode in {"strict-target", "jsd"}
    if args.mode == "strict-target":
        os.environ["ASYMSPEC_METHOD"] = "strict_target"
    elif args.mode == "jsd":
        os.environ["ASYMSPEC_METHOD"] = "jsd"

    common = dict(
        model=DRAFTER if args.mode == "draft" else VERIFIER,
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
    if is_specsteer:
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
    reference_tokenizer = AutoTokenizer.from_pretrained(VERIFIER)
    if not isinstance(reference_tokenizer.chat_template, str):
        raise RuntimeError("Qwen3.8 reference tokenizer has no chat template")
    chat_template = reference_tokenizer.chat_template
    cases = (load_suite(args.suite) if args.suite else [{
        "id": "single", "messages": load_messages(args.messages),
        "max_tokens": args.max_tokens,
    }])
    results = []
    for case in cases:
        max_tokens = case.get("max_tokens", args.max_tokens)
        prompt_ids = render_prompt_ids(tokenizer, case["messages"], chat_template)
        if (is_specsteer and len(prompt_ids) + max_tokens
                + args.num_speculative_tokens > args.main_max_model_len):
            parser.error(f"case {case['id']!r}: prompt + completion + K exceeds "
                         "--main-max-model-len")
        sampling_kwargs = {"temperature": 0, "max_tokens": max_tokens,
                           "seed": 0}
        if is_specsteer:
            sampling_kwargs["extra_args"] = {
                "specsteer_aug_prompt_ids": prompt_ids}
        started = time.perf_counter()
        result = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            SamplingParams(**sampling_kwargs),
            use_tqdm=False,
        )[0]
        elapsed = time.perf_counter() - started
        output = result.outputs[0]
        output_ids = list(output.token_ids)
        results.append({
            "id": case["id"],
            "prompt_token_ids": prompt_ids,
            "prompt_sha256": hashlib.sha256(
                json.dumps(prompt_ids, separators=(",", ":")).encode()).hexdigest(),
            "generated_token_ids": output_ids,
            "generated_text": output.text,
            "generated_tokens": len(output_ids),
            "elapsed_generation_seconds": elapsed,
            "output_tokens_per_second": len(output_ids) / elapsed if elapsed else None,
        })
    capture = {
        "mode": args.mode,
        "model": common["model"],
        "temperature": 0,
        "seed": 0,
        "enable_thinking": False,
        "chat_template_source": VERIFIER,
        "num_speculative_tokens": args.num_speculative_tokens if is_specsteer else None,
        "cases": results,
    }
    if not args.suite:
        # Preserve the legacy single-message capture shape for comparison tools.
        capture.update(results[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(capture, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "cases": [
        {"id": result["id"], "prompt_tokens": len(result["prompt_token_ids"]),
         "generated_tokens": result["generated_tokens"],
         "output_tokens_per_second": result["output_tokens_per_second"]}
        for result in results]}, indent=2))


if __name__ == "__main__":
    main()
