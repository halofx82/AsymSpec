#!/usr/bin/env python3
"""Interactive terminal chat using AsymSpec's full and compressed views.

This is intentionally a local REPL, not an OpenAI-compatible server.  The
patched AsymSpec engine needs both prompt token streams on every request;
stock vLLM's HTTP schema has no field for the full-context drafter stream.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm import LLM, SamplingParams

from paths import LLM_MODEL, slm_model


def read_optional_file(path: str | None, option: str) -> str:
    if path is None:
        return ""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise SystemExit(f"{option} is not a readable file: {file_path}")
    return file_path.read_text(encoding="utf-8")


def add_context(messages: list[dict[str, str]], text: str, label: str) -> None:
    if text:
        messages.append({"role": "system", "content": f"{label}\n{text}"})


def prompt_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Render Qwen's chat template once, retaining an explicit token list."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return list(tokenizer.encode(text, add_special_tokens=False))


def compressed_messages(
    tokenizer,
    system: str,
    main_context: str,
    history: list[dict[str, str]],
    token_budget: int,
) -> list[dict[str, str]]:
    """Keep the newest whole turns that fit; never silently truncate a turn."""
    fixed: list[dict[str, str]] = []
    add_context(fixed, system, "System instructions:")
    add_context(fixed, main_context, "Compressed reference context:")
    if len(prompt_ids(tokenizer, fixed)) > token_budget:
        raise ValueError("system prompt and --main-context-file exceed compressed capacity")

    kept_reversed: list[dict[str, str]] = []
    for message in reversed(history):
        candidate = fixed + list(reversed(kept_reversed + [message]))
        if len(prompt_ids(tokenizer, candidate)) > token_budget:
            if not kept_reversed:
                raise ValueError("latest user turn exceeds compressed capacity")
            break
        kept_reversed.append(message)
    return fixed + list(reversed(kept_reversed))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slm", default="4B", help="Qwen3 drafter size (default: 4B)")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel world size")
    ap.add_argument("--K", type=int, default=2, help="speculative lookahead")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--asym-method", default="jsd", choices=["jsd", "none"])
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--specsteer-main-max-model-len", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    ap.add_argument("--cpu-offload-gb", type=float, default=0.0)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--system", default="", help="system instructions for both views")
    ap.add_argument("--full-context-file", help="reference text included only in the full drafter view")
    ap.add_argument("--main-context-file", help="compressed/summary reference text for verifier/base")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.specsteer_main_max_model_len > args.max_model_len:
        raise SystemExit("--specsteer-main-max-model-len must not exceed --max-model-len")
    reserve = args.max_new + args.K
    full_budget = args.max_model_len - reserve
    main_budget = args.specsteer_main_max_model_len - reserve
    if main_budget <= 0 or full_budget <= 0:
        raise SystemExit("context limit must leave room for --max-new and --K")

    full_context = read_optional_file(args.full_context_file, "--full-context-file")
    main_context = read_optional_file(args.main_context_file, "--main-context-file")
    os.environ["ASYMSPEC_METHOD"] = args.asym_method
    os.environ["ASYMSPEC_DELTA_SRC"] = "ours"

    llm = LLM(
        model=LLM_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        max_num_batched_tokens=min(args.max_model_len, 16384),
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        speculative_config={
            "method": "specsteer",
            "model": slm_model(args.slm),
            "num_speculative_tokens": args.K,
            "draft_tensor_parallel_size": args.tp,
            "specsteer_beta": args.beta,
            "specsteer_gamma": args.gamma,
            "specsteer_main_max_model_len": args.specsteer_main_max_model_len,
        },
    )
    tokenizer = llm.get_tokenizer()
    history: list[dict[str, str]] = []
    print("AsymSpec chat ready. Commands: /reset, /quit. Ctrl-D also exits.")

    while True:
        try:
            user_text = input("\nyou> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            break
        if user_text == "/reset":
            history.clear()
            print("conversation cleared")
            continue

        candidate_history = history + [{"role": "user", "content": user_text}]
        full_messages: list[dict[str, str]] = []
        add_context(full_messages, args.system, "System instructions:")
        add_context(full_messages, full_context, "Full reference context:")
        full_messages.extend(candidate_history)
        try:
            full_ids = prompt_ids(tokenizer, full_messages)
            if len(full_ids) > full_budget:
                raise ValueError("full context capacity exceeded; use /reset or a shorter turn")
            main_messages = compressed_messages(
                tokenizer, args.system, main_context, candidate_history, main_budget)
            main_ids = prompt_ids(tokenizer, main_messages)
        except ValueError as exc:
            print(f"[not sent] {exc}")
            continue

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new,
            extra_args={"specsteer_aug_prompt_ids": full_ids},
        )
        result = llm.generate(
            [{"prompt_token_ids": main_ids}], sampling, use_tqdm=False)[0]
        answer = result.outputs[0].text
        print(f"assistant> {answer}")
        history = candidate_history + [{"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
