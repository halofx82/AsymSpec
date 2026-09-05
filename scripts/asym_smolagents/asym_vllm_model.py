"""
AsymSpecVLLMModel — smolagents Model that runs AsymSpec via our patched vLLM fork.

On every generate() call:
  1. aug_messages = full message history (verbatim, what the agent sees)
  2. main_messages = compressed version (top-and-tail truncation OR LLMLingua-2)
  3. Tokenize both, pass aug_prompt_ids via SamplingParams.extra_args["specsteer_aug_prompt_ids"]
  4. vLLM AsymSpec sampler does δ-fusion + CDA acceptance gating

Compression strategies (--main_compression):
  - "truncate"  : top-and-tail for messages > THRESHOLD chars (first N + last M)
  - "llmlingua" : LLMLingua-2 per-message compression at target_ratio
                  (skip short messages; preserve role/structure)
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from typing import Any

from smolagents import VLLMModel
from smolagents.models import ChatMessage, MessageRole, TokenUsage

# Avoid LLMLingua's verbose deprecation warnings polluting agent logs.
warnings.filterwarnings("ignore", category=FutureWarning, module="llmlingua")
warnings.filterwarnings("ignore", category=UserWarning, module="llmlingua")

LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Message content longer than this triggers compression. Short messages
# (questions, tool calls, single-line thoughts) pass through unchanged.
COMPRESS_THRESHOLD_CHARS = 2000


def _compress_truncate(text: str, head: int = 1000, tail: int = 500) -> str:
    """Top-and-tail: keep first `head` + last `tail` chars, mark elision."""
    if len(text) <= head + tail + 50:
        return text
    return f"{text[:head]}\n[...{len(text) - head - tail} chars truncated...]\n{text[-tail:]}"


class _LazyLingua:
    """Singleton holder for LLMLingua-2 compressor (loads on first use)."""
    _compressor = None

    @classmethod
    def get(cls):
        if cls._compressor is None:
            from llmlingua import PromptCompressor
            print(f"[AsymSpec] loading LLMLingua-2: {LLMLINGUA2_MODEL}", flush=True)
            cls._compressor = PromptCompressor(
                model_name=LLMLINGUA2_MODEL,
                use_llmlingua2=True,
                device_map="cpu",  # keep GPU for our vLLM
            )
        return cls._compressor


# Function-signature pattern (matches smolagents tool defs + APIB-style sigs).
# Preserved verbatim while surrounding prose is compressed.
_SIG_LINE_RE = re.compile(
    r"^(?:"
    r"\s*def\s+\w+\([^)]*\)"        # def fn_name(args)
    r"|\s*\w+\s*\([^)]*\)\s*->.*"   # fn_name(args) -> ret
    r"|\s*\w+\s*\([^)]*\)\s*:?\s*$" # fn_name(args)[:]
    r"|\s*[Aa]rguments?\s*:"        # Arguments:
    r"|\s*[Rr]eturns?\s*:"          # Returns:
    r"|\s*-\s*\w+\s*\([^)]*\)\s*:"  # - arg_name (type): desc
    r"|\s*\w+\s*:\s*(?:str|int|bool|float|list|dict|tuple|None|Any|object)"
    r")", re.MULTILINE,
)


def _compress_with_sig_preserve(text: str, target_ratio: float = 0.3) -> str:
    """Compress text while preserving lines matching function-signature patterns.

    Mirrors APIB Method A (name_sig) compression: drop API descriptions but
    keep `function_name(param: type, ...)` signature lines verbatim. The
    verifier on main can then always identify available functions and their
    parameter schemas, while saving compute on descriptive prose.

    Splits input by lines into (sig, prose) groups. Sig groups: verbatim.
    Prose groups (consecutive non-sig lines): batched and compressed via
    LLMLingua at the given target_ratio.
    """
    if len(text) < COMPRESS_THRESHOLD_CHARS:
        return text

    lines = text.split("\n")
    chunks = []  # list of (kind, "joined-text")
    current_kind = None
    current_lines = []
    for line in lines:
        is_sig = bool(_SIG_LINE_RE.match(line))
        kind = "sig" if is_sig else "prose"
        if kind != current_kind and current_lines:
            chunks.append((current_kind, "\n".join(current_lines)))
            current_lines = []
        current_kind = kind
        current_lines.append(line)
    if current_lines:
        chunks.append((current_kind, "\n".join(current_lines)))

    out_parts = []
    for kind, chunk_text in chunks:
        if kind == "sig":
            out_parts.append(chunk_text)  # verbatim
        else:
            if len(chunk_text) >= COMPRESS_THRESHOLD_CHARS:
                out_parts.append(_compress_llmlingua(chunk_text, target_ratio=target_ratio))
            else:
                out_parts.append(chunk_text)
    return "\n".join(out_parts)


# In-process cache: maps (content_hash, target_ratio) → compressed_text.
# Purpose: incremental per-turn compression. In agent ReAct, generate() is
# called repeatedly with growing message history; older messages get
# re-compressed at each call. With this cache, each unique (content, ratio)
# is compressed exactly once and reused, ensuring stable compressed history
# across turns AND saving compute.
_LLMLINGUA_CACHE: dict[tuple[str, float], str] = {}
_LLMLINGUA_CACHE_HITS = 0
_LLMLINGUA_CACHE_MISSES = 0


def _compress_llmlingua(text: str, target_ratio: float = 0.3) -> str:
    """LLMLingua-2 compression to ~target_ratio of original tokens. CPU-only.

    Cached by SHA256(text) + target_ratio to ensure incremental per-turn
    behaviour: each unique (content, ratio) is compressed once and reused.
    """
    global _LLMLINGUA_CACHE_HITS, _LLMLINGUA_CACHE_MISSES
    if len(text) < COMPRESS_THRESHOLD_CHARS:
        return text
    import hashlib
    key = (hashlib.sha256(text.encode("utf-8")).hexdigest(), target_ratio)
    if key in _LLMLINGUA_CACHE:
        _LLMLINGUA_CACHE_HITS += 1
        return _LLMLINGUA_CACHE[key]
    _LLMLINGUA_CACHE_MISSES += 1
    compressor = _LazyLingua.get()
    target_tokens = max(50, int(len(text) / 4 * target_ratio))
    try:
        out = compressor.compress_prompt(
            text, target_token=target_tokens,
            force_tokens=["\n", ".", "==="],
        )
        result = out["compressed_prompt"]
    except Exception as e:
        print(f"[AsymSpec] LLMLingua failed ({e}), falling back to truncate", flush=True)
        result = _compress_truncate(text)
    _LLMLINGUA_CACHE[key] = result
    return result


def get_llmlingua_cache_stats() -> dict:
    return {
        "size": len(_LLMLINGUA_CACHE),
        "hits": _LLMLINGUA_CACHE_HITS,
        "misses": _LLMLINGUA_CACHE_MISSES,
    }


def _compress_messages(
    messages: list[dict], strategy: str, skip_system: bool = True,
    llmlingua_rate: float = 0.3, keep_last_k: int = 0,
    preserve_fn_sigs: bool = False,
) -> list[dict]:
    """Apply per-message compression. Preserve roles + short messages.

    skip_system=True: NEVER compress role='system' messages.
    skip_system=False (recommended for agent ReAct with LLMLingua): compress
    system too — smolagents' verbose few-shot scaffolding distracts the
    verifier; LLMLingua-2 strips redundant tokens cleanly.

    keep_last_k: preserve the LAST k messages verbatim (matches MC paper's
    `summary_last_k=1` strategy adapted to agent ReAct). Use this to keep
    the current-turn tool observation (and optionally the preceding
    assistant action) uncompressed while compressing earlier turns. This
    creates real Floor-Ceiling headroom for AsymSpec to recover on
    multi-turn agent loops where the latest observation is most informative.

    llmlingua_rate: target retention ratio for LLMLingua-2 (default 0.3).
    Only applies when strategy='llmlingua'.
    """
    if strategy == "none":
        return messages
    if strategy == "llmlingua":
        if preserve_fn_sigs:
            fn = lambda c: _compress_with_sig_preserve(c, target_ratio=llmlingua_rate)
        else:
            fn = lambda c: _compress_llmlingua(c, target_ratio=llmlingua_rate)
    else:
        fn = _compress_truncate
    # Indices of the last k non-system messages to preserve verbatim
    keep_idxs = set()
    if keep_last_k > 0:
        non_sys = [i for i, m in enumerate(messages) if m.get("role") != "system"]
        keep_idxs = set(non_sys[-keep_last_k:])
    out = []
    for i, m in enumerate(messages):
        content = m.get("content", "")
        role = m.get("role", "")
        if skip_system and role == "system":
            out.append(m)
            continue
        if i in keep_idxs:
            out.append(m)
            continue
        # smolagents sometimes uses list-of-parts content (vision); skip those.
        if isinstance(content, str) and len(content) > COMPRESS_THRESHOLD_CHARS:
            new_content = fn(content)
        else:
            new_content = content
        out.append({**m, "content": new_content})
    return out


class AsymSpecVLLMModel(VLLMModel):
    """smolagents.Model wrapper that runs AsymSpec on each generate() call.

    Inherits VLLMModel's offline LLM init; overrides generate() to inject
    aug context via SamplingParams.extra_args.
    """

    def __init__(
        self,
        model_id: str,
        drafter_id: str,
        K: int = 4,
        gamma: float = 0.5,
        beta: float = 1.0,
        asym_method: str = "jsd",  # CDA: gamma_eff = gamma * exp(-JSD)
        cma_lambda: float = 0.1,
        main_compression: str = "truncate",  # "truncate" | "llmlingua" | "none"
        skip_system_compress: bool = True,  # keep system prompt intact when compressing
        llmlingua_rate: float = 0.3,  # LLMLingua-2 target retention ratio (lower = more aggressive)
        keep_last_k: int = 0,  # preserve last k non-system messages verbatim (MC-style)
        code_zone_suppress: bool = False,  # NEW: zone-aware δ-fusion gating
        cz_variant: str = "in_code",  # in_code | outside_code | in_final_arg
        preserve_fn_sigs: bool = False,  # APIB Method A-style fn signature preservation
        model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        self._skip_system_compress = skip_system_compress
        self._llmlingua_rate = llmlingua_rate
        self._keep_last_k = keep_last_k
        self._code_zone_suppress = code_zone_suppress
        self._cz_variant = cz_variant
        self._preserve_fn_sigs = preserve_fn_sigs
        assert cz_variant in ("in_code", "outside_code", "in_final_arg"), cz_variant
        # Inject AsymSpec speculative_config into vLLM init.
        model_kwargs = dict(model_kwargs or {})
        model_kwargs["speculative_config"] = {
            "method": "specsteer",
            "model": drafter_id,
            "num_speculative_tokens": K,
            "specsteer_beta": beta,
            "specsteer_gamma": gamma,
        }
        # Bench-style env vars consumed by our specsteer kernel.
        os.environ["ASYMSPEC_METHOD"] = asym_method
        os.environ["CMA_LAMBDA"] = str(cma_lambda)
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # Critical: Qwen3 chat template defaults enable_thinking=True. With
        # AsymSpec δ-fusion, drafter's <think>...</think> tokens propagate
        # to verifier output, which smolagents' parser misinterprets as the
        # final answer (causing ~30% format failure). Match bench_lb.py /
        # benchmark convention by disabling thinking explicitly.
        actk = dict(kwargs.pop("apply_chat_template_kwargs", None) or {})
        actk.setdefault("enable_thinking", False)
        kwargs["apply_chat_template_kwargs"] = actk

        super().__init__(model_id=model_id, model_kwargs=model_kwargs, **kwargs)

        # Path B is enabled by default in every TP worker's proposer.
        print("[AsymSpec] PathB enabled", flush=True)

        self.K = K
        self.gamma = gamma
        self.beta = beta
        self.asym_method = asym_method
        self.main_compression = main_compression
        self._call_count = 0
        self._diag_log = []  # per-call: (aug_tokens, main_tokens, ratio)

    def _render(self, messages: list[dict], tools=None) -> str:
        """Apply chat template the same way base VLLMModel does."""
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True,
            tokenize=False, **self.apply_chat_template_kwargs,
        )

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list = None,
        **kwargs,
    ) -> ChatMessage:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        # Mirror base VLLMModel.generate() preparation.
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            flatten_messages_as_text=(not self._is_vlm),
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        structured_outputs = (
            StructuredOutputsParams(json=response_format["json_schema"]["schema"])
            if response_format else None
        )
        msgs = completion_kwargs.pop("messages")
        prepared_stop = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        # === AsymSpec aug/main split ===
        # msgs already has list-of-dicts form after _prepare_completion_kwargs.
        # Defensive: convert any ChatMessage objects.
        msgs_dict = [
            m.dict() if hasattr(m, "dict") else dict(m)
            for m in msgs
        ]
        aug_messages = msgs_dict
        main_messages = _compress_messages(
            msgs_dict, self.main_compression,
            skip_system=self._skip_system_compress,
            llmlingua_rate=self._llmlingua_rate,
            keep_last_k=self._keep_last_k,
            preserve_fn_sigs=self._preserve_fn_sigs,
        )

        aug_prompt = self._render(aug_messages, tools=tools)
        main_prompt = self._render(main_messages, tools=tools)

        aug_ids = self.tokenizer.encode(aug_prompt)
        main_ids = self.tokenizer.encode(main_prompt)
        main_ids_len = len(main_ids)
        ratio = len(aug_ids) / max(main_ids_len, 1)
        self._call_count += 1
        if self._call_count <= 3 or self._call_count % 10 == 0:
            mode = f"cz:{self._cz_variant}" if self._code_zone_suppress else "vanilla"
            print(f"[AsymSpec call#{self._call_count}] aug_tok={len(aug_ids)} "
                  f"main_tok={main_ids_len} ratio={ratio:.2f}x "
                  f"compression={self.main_compression} mode={mode}", flush=True)
        self._diag_log.append({
            "call": self._call_count,
            "aug_tokens": len(aug_ids), "main_tokens": main_ids_len,
            "ratio": ratio,
        })

        # === Code-zone-aware generation ===
        # Mode-switching loop: alternate AsymSpec (Thought zone) and
        # verifier-equivalent (Code zone). Detected by ``` markers.
        # Inside code zone, set aug_ids = main_ids → δ ≡ 0 → effective β=0.
        if self._code_zone_suppress:
            output_text = self._code_zone_aware_generate(
                main_prompt=main_prompt, aug_ids=aug_ids, main_ids=main_ids,
                user_stop=prepared_stop, max_tokens=kwargs.get("max_tokens", 2048),
                structured_outputs=structured_outputs,
                completion_kwargs=completion_kwargs,
            )
        else:
            sampling_params = SamplingParams(
                n=kwargs.get("n", 1),
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 2048),
                stop=prepared_stop,
                structured_outputs=structured_outputs,
                extra_args={"specsteer_aug_prompt_ids": aug_ids},
            )
            out = self.model.generate(
                main_prompt, sampling_params=sampling_params, **completion_kwargs,
            )
            output_text = out[0].outputs[0].text

        from smolagents.models import remove_content_after_stop_sequences
        if stop_sequences is not None and not getattr(self, "supports_stop_parameter", False):
            output_text = remove_content_after_stop_sequences(output_text, stop_sequences)

        # Estimate output token count (used for TokenUsage)
        output_token_count = len(self.tokenizer.encode(output_text))

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={"out": output_text, "completion_kwargs": completion_kwargs,
                 "aug_tokens": len(aug_ids), "main_tokens": main_ids_len},
            token_usage=TokenUsage(
                input_tokens=main_ids_len,  # what verifier sees
                output_tokens=output_token_count,
            ),
        )

    def _code_zone_aware_generate(
        self, main_prompt: str, aug_ids: list, main_ids: list,
        user_stop: list, max_tokens: int, structured_outputs,
        completion_kwargs: dict,
    ) -> str:
        """Zone-aware mode-switching loop with 3 variants:

        - in_code (default): β=0 INSIDE ```py code blocks (suppress drafter
          where drafter's pattern bias corrupts function-name tokens).
        - outside_code: β=0 OUTSIDE code blocks (let drafter's factual recall
          help inside code; suppress drafter's entity bias in Thought).
        - in_final_arg: β=0 only between `final_answer(` and `)` (narrowest;
          targets only the final-answer argument emission).
        """
        from vllm import SamplingParams

        # Zone-open / -close markers per variant. open=enter suppress zone (or
        # enter δ zone for outside_code); close=exit it.
        if self._cz_variant in ("in_code", "outside_code"):
            ZONE_OPEN = ["```py", "```python"]
            ZONE_CLOSE = ["\n```\n", "```\n", "\n```<", "```<end_code>"]
        else:  # in_final_arg
            ZONE_OPEN = ["final_answer("]
            ZONE_CLOSE = [")\n", ")<end_code>", ")```"]

        # For in_code: zone 0 = thought (δ on), zone 1 = code (δ off).
        # For outside_code: zone 0 = thought (δ off), zone 1 = code (δ on).
        # For in_final_arg: zone 0 = pre-arg (δ on), zone 1 = inside-arg (δ off).
        def delta_on(in_zone1: bool) -> bool:
            if self._cz_variant == "outside_code":
                return in_zone1  # δ on only inside code
            return not in_zone1  # δ off inside zone1 (code / final-arg)

        out_so_far = ""
        in_zone1 = False
        prompt = main_prompt
        remaining = max_tokens
        iteration = 0
        MAX_ITERS = 16  # raised from 8 to handle multi-round Thought↔Code chats

        while remaining > 0 and iteration < MAX_ITERS:
            iteration += 1
            effective_aug_ids = aug_ids if delta_on(in_zone1) else main_ids
            phase_stops = (ZONE_CLOSE if in_zone1 else ZONE_OPEN) + (user_stop or [])

            sp = SamplingParams(
                n=1, temperature=0.0, max_tokens=remaining,
                stop=phase_stops,
                structured_outputs=structured_outputs,
                extra_args={"specsteer_aug_prompt_ids": effective_aug_ids},
                include_stop_str_in_output=True,
            )
            chunk_out = self.model.generate(
                prompt, sampling_params=sp, **completion_kwargs,
            )
            chunk_text = chunk_out[0].outputs[0].text
            chunk_tokens = len(chunk_out[0].outputs[0].token_ids)
            remaining -= chunk_tokens
            out_so_far += chunk_text

            stop_reason = chunk_out[0].outputs[0].finish_reason
            if stop_reason != "stop":
                break
            hit_user_stop = any(
                chunk_text.rstrip().endswith(s) for s in (user_stop or [])
            )
            if hit_user_stop:
                break
            # Zone-transition marker hit: flip zone, continue
            in_zone1 = not in_zone1
            prompt = main_prompt + out_so_far

        return out_so_far

    def get_diag_summary(self) -> dict:
        """Return aggregate stats on aug/main ratio across calls."""
        if not self._diag_log:
            return {}
        ratios = [d["ratio"] for d in self._diag_log]
        return {
            "n_calls": len(self._diag_log),
            "ratio_mean": sum(ratios) / len(ratios),
            "ratio_max": max(ratios),
            "ratio_min": min(ratios),
            "aug_tokens_mean": sum(d["aug_tokens"] for d in self._diag_log) / len(self._diag_log),
            "main_tokens_mean": sum(d["main_tokens"] for d in self._diag_log) / len(self._diag_log),
        }
