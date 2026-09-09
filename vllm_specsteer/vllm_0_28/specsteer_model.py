# SPDX-License-Identifier: Apache-2.0
# AsymSpec proposer for vLLM 0.28.0. The legacy ``SpecSteer`` class and field
# names are retained to minimize changes to vLLM's integration surface.
#
# The proposer extends DraftModelProposer with a second view of the same small
# model. The full-context view proposes tokens and produces ``aug_logits``;
# the compressed-context view maintains a separate KV cache and produces
# ``base_logits``. The verifier remains on the compressed context. Text and
# single-image multimodal requests are supported.
#
# ╭──────────────────────── per-step data flow ────────────────────────╮
# │   aug context (full)  ──▶ SLM_aug  ─┐                               │
# │                                     ├─▶ K draft tokens + aug_logits │
# │   main context (compressed) ──▶ LLM ├─▶ target_logits @ K+1 pos     │
# │   main context (compressed) ──▶ SLM_base ─▶ base_logits @ K pos     │
# │                                     │                               │
# │                                     ▼                               │
# │               specsteer_greedy_sample                               │
# │         (γ-rule accept, fused argmax on reject)                     │
# ╰─────────────────────────────────────────────────────────────────────╯
#
from __future__ import annotations

import os
from copy import copy
import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.hybrid_specsteer import (
    is_hybrid_model_config,
    is_strict_target_mode,
    require_distinct_runtime_layer_sets,
    require_strict_target_runtime_layer_set,
    split_hybrid_layer_names,
    text_mrope_positions,
    uses_compressed_base,
)

logger = init_logger(__name__)


def _module_and_leaf(root: nn.Module, qualified_name: str):
    """Return the owning module and final registration name."""
    parent_name, _, leaf = qualified_name.rpartition(".")
    return (root.get_submodule(parent_name) if parent_name else root), leaf


def _share_model_state(drafter: nn.Module, base: nn.Module) -> tuple[int, int]:
    """Rebind ``base`` state to ``drafter`` without sharing module objects.

    The base tree is constructed on the meta device, so this is also the only
    materialization step for its parameters and buffers.  Keeping the trees
    distinct preserves attention-layer names and independent KV identities.
    """
    # Keep duplicate registration names: tied embeddings/lm_head must retain
    # the drafter's exact alias graph, not merely equivalent storage.
    draft_params = dict(drafter.named_parameters(remove_duplicate=False))
    base_params = dict(base.named_parameters(remove_duplicate=False))
    if draft_params.keys() != base_params.keys():
        mismatch = sorted(draft_params.keys() ^ base_params.keys())[:16]
        raise RuntimeError(
            "AsymSpec base/drafter parameter layouts differ: " + repr(mismatch))

    parameter_bytes = 0
    seen_parameters: set[int] = set()
    for name, draft_param in draft_params.items():
        parent, leaf = _module_and_leaf(base, name)
        if id(draft_param) not in seen_parameters:
            parameter_bytes += draft_param.numel() * draft_param.element_size()
            seen_parameters.add(id(draft_param))
        parent._parameters[leaf] = draft_param

    draft_buffers = dict(drafter.named_buffers(remove_duplicate=False))
    base_buffers = dict(base.named_buffers(remove_duplicate=False))
    if draft_buffers.keys() != base_buffers.keys():
        mismatch = sorted(draft_buffers.keys() ^ base_buffers.keys())[:16]
        raise RuntimeError(
            "AsymSpec base/drafter buffer layouts differ: " + repr(mismatch))

    buffer_bytes = 0
    seen_buffers: set[int] = set()
    for name, draft_buffer in draft_buffers.items():
        parent, leaf = _module_and_leaf(base, name)
        if id(draft_buffer) not in seen_buffers:
            buffer_bytes += draft_buffer.numel() * draft_buffer.element_size()
            seen_buffers.add(id(draft_buffer))
        parent._buffers[leaf] = draft_buffer

    # These are intentionally hard failures: a partially materialized base
    # model can survive startup and fail nondeterministically on first forward.
    rebound_params = dict(base.named_parameters(remove_duplicate=False))
    rebound_buffers = dict(base.named_buffers(remove_duplicate=False))
    for name, draft_param in draft_params.items():
        base_param = rebound_params[name]
        if base_param is not draft_param:
            raise RuntimeError(f"AsymSpec parameter was not shared: {name}")
    for name, draft_buffer in draft_buffers.items():
        base_buffer = rebound_buffers[name]
        if base_buffer is not draft_buffer:
            raise RuntimeError(f"AsymSpec buffer was not shared: {name}")
    for name, tensor in (*rebound_params.items(), *rebound_buffers.items()):
        if tensor.is_meta:
            raise RuntimeError(f"AsymSpec left a meta tensor behind: {name}")
        if isinstance(tensor, torch.nn.parameter.UninitializedParameter):
            raise RuntimeError(f"AsymSpec left an uninitialized tensor: {name}")
    return parameter_bytes, buffer_bytes


def _install_drafter_gid_split_patch():
    # Scheduler integration is deployed in core/*.py. Only worker-local
    # caches live here; never delete another run's compilation cache.
    import vllm.v1.core.kv_cache_utils as kvu
    for name in ("_specsteer_aug_offsets", "_specsteer_aug_prefilled",
                 "_specsteer_aug_mm_data"):
        if not hasattr(kvu, name):
            setattr(kvu, name, {})


class SpecSteerProposer(DraftModelProposer):
    """Draft-model proposer augmented with contrast fusion.

    SLM_base is a vLLM-native model (loaded via `get_model()`) running on the
    main context to produce base_logits. The fused score
        δ = log_softmax(aug_logits) - log_softmax(base_logits)
    drives the contrast fusion at reject positions, so SpecSteer is genuinely
    non-trivial vs LLM-only γ-rule.

    Per-step base forward goes through `_base_forward_fast` (line ~905,
    vLLM's `build_for_drafting` metadata + paged KV in gid=2-equivalent
    block_table; BS≥1 capable). Slow fallback `_base_parallel_verify`
    (line ~655) handles prefill, shape changes, and over-sized batches.

    Batched execution uses one code path for BS≥1:
    - `_compute_aug_first_bonus`: batched ragged drafter forward over N reqs
      via concat input_ids + per-req block_table rows + per-req query_start_loc.
    - Gate A bonus correction call site: collect-then-batched (single drafter
      forward for all pending reqs across batch).
    - PREFILL aug-context swap (line ~1730) and decode aug-offset shift
      (line ~1860): rewritten to handle per-req aug_ids / per-req offsets via
      ragged tensor construction. PREFILL gates on "homogeneous batch" (all
      N reqs have aug with positive offset); heterogeneous mixed-mode batches
      still skip the swap and use the symmetric fallback path.
    - Aug_offset detection (line ~1685): loops all N reqs, builds per_req_aug
      list with (req_idx, aug_ids, aug_offset, L_main, L_aug) per req.

    At BS=1, the batched tensor operations reduce to the corresponding
    single-request operations.

    Opt-in via speculative_config.method == "specsteer".
    """

    def model_returns_tuple(self):
        return False

    def _raise_if_mrope(self):
        # AsymSpec supplies its own image embeddings and M-RoPE positions.
        pass

    def _warn_if_multimodal(self):
        pass

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        # Install KV-cache-group split patch BEFORE super (which may allocate
        # KV cache during load_model later). Idempotent.
        _install_drafter_gid_split_patch()
        # Must be available while the parent invokes _get_model().  Strict
        # target has no compressed 4B view at all, rather than a dormant one.
        self._strict_target_mode = is_strict_target_mode()
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        self.runner = runner  # parent doesn't store it
        # The parent only allocates self.positions for non-M-RoPE
        # drafters (eagle.py:148-160). For M-RoPE drafter (Qwen3-VL), the
        # inherited Triton kernel `copy_and_expand_eagle_inputs_kernel` at
        # eagle.py:729 still hardcodes `out_positions_ptr=self.positions`.
        # Allocate a dummy 1D buffer so the kernel doesn't AttributeError.
        # The set_inputs_first_pass override mirrors the kernel output
        # to mrope_positions for actual forward consumption.
        if getattr(self, "uses_mrope", False) and not hasattr(self, 'positions'):
            self.positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device,
            )
            logger.info("AsymSpec: allocated dummy self.positions (M-RoPE drafter)")
        self.beta: float = getattr(
            self.speculative_config, "specsteer_beta", 1.0,
        )
        self.gamma: float = getattr(
            self.speculative_config, "specsteer_gamma", 0.5,
        )
        # Per-step ring buffer: list of [batch_size, vocab_size] tensors,
        # one per drafted token position. Reset at the start of each propose().
        self._draft_logits_per_pos: list[torch.Tensor] = []
        # Parallel ring buffer for base's per-position logits, populated from
        # dual_forward's stashed base hidden at each drafter forward call.
        self._base_logits_per_pos: list[torch.Tensor] = []
        # Last base hidden states from dual_forward (populated inside drafter's
        # forward hook). Consumed by _greedy_sample.

        # SLM_base: a second vLLM-native SLM with its own KV cache pool.
        # Same weights as drafter but INDEPENDENT cache blocks so its state
        # can track main ctx while drafter tracks aug ctx. Loaded lazily in
        # _get_model after the drafter is set up (parent calls _get_model
        # itself; we hook there to also initialize the base instance).
        self.base_model: nn.Module | None = None
        self._base_attn_layer_names: set[str] | None = None
        self._hybrid_specsteer = is_hybrid_model_config(
            self._create_draft_vllm_config().model_config)
        self._base_full_attn_layer_names: set[str] = set()
        self._base_gdn_layer_names: set[str] = set()
        self._draft_full_attn_layer_names: set[str] = set()
        self._draft_gdn_layer_names: set[str] = set()
        self.base_attn_groups: list = []   # filled by initialize_attn_backend
        self.base_kv_cache_gid: int = -1
        self.base_kv_cache_gids: set[int] = set()
        self.drafter_kv_cache_gids: set[int] = set()
        # PathB: base_logits computed via parallel_verify after the drafter
        # propose, NOT inline via dual_forward. This is the canonical
        # SpecSteer path — dual_forward base was an early experiment and is
        # always skipped here.
        self._pathb_skip_dual_base: bool = True
        # DIAG: set True to route parallel_verify to drafter model (self.model)
        # instead of base_model — isolates attn_metadata vs base_model bugs.
        self._pv_use_drafter_for_diag: bool = False  # use base_model
        # DIAG: forward ONLY committed tokens (no drafts). hidden[-1] should
        # match drafter's aug_logits[0] — isolates attn setup bugs.
        self._pv_committed_only: bool = False  # diag off — full prefill path

        # Base-model state populated by _base_mirror_forward each step.
        # _last_mirror_tail_hidden: [1, hidden] — last hidden state of
        # base's mirror forward. Its logit (via compute_logits) predicts
        # the next token = drafts[0] under base main ctx.
        self._last_mirror_tail_hidden: torch.Tensor | None = None
        self._last_base_hidden: torch.Tensor | None = None
        self._last_base_logits: torch.Tensor | None = None
        # dual_forward input-swap state: stashed target_token_ids (aug/main)
        # used to reconstruct base's input when Gate A is active.
        self._stashed_token_indices_to_sample: torch.Tensor | None = None

        # T_profile: per-segment CUDA event timing (enabled via env var)
        import os as _os
        # ``ASYMSPEC_PROFILE`` is the public switch.  Keep the older name
        # working for local experiments made before this profiler was wired
        # into strict-target mode.
        self._profile_enabled = (
            _os.environ.get("ASYMSPEC_PROFILE",
                            _os.environ.get("SPECSTEER_PROFILE", "0"))
            == "1"
        )
        self._prof_events: dict[str, list] = {}
        self._prof_step = 0
        self._prof_report_every = int(_os.environ.get(
            "SPECSTEER_PROFILE_EVERY", "20"))
        # Cache-hit telemetry: counts (hits, misses, sum_skipped_tokens)
        # tracked per-task by the merged-prefill path.
        self._prof_cache_hits = 0
        self._prof_cache_misses = 0
        self._prof_cache_skipped_tokens = 0
        self._prof_steps: list[dict] = []
        self._prof_path = _os.environ.get("ASYMSPEC_PROFILE_FILE")

        # T1: pre-allocated buffers for fast base PV path. Strict target never
        # uses them; retain tiny placeholders so legacy helper attributes stay
        # well-defined without allocating the former large base staging area.
        # models use the legacy K+2 warmup rewrite. Hybrid GDN metadata has a
        # native speculative layout of exactly K+1 tokens, so it must start at
        # the first uncomputed position instead.
        K = getattr(self, "num_speculative_tokens", 2) or 2
        max_bs = 1 if self._strict_target_mode else 256
        tpr = K + (1 if self._hybrid_specsteer else 2)
        total = max_bs * tpr
        self._base_pv_max_bs = max_bs
        self._base_pv_tokens_per_req = tpr
        self._base_input_ids_buf = torch.zeros(
            total, dtype=torch.int32, device=self.device,
        )
        self._base_positions_buf = torch.zeros(
            total, dtype=torch.int64, device=self.device,
        )
        self._base_slot_mapping_buf = torch.zeros(
            total, dtype=torch.int64, device=self.device,
        )
        self._base_seq_lens_buf = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device,
        )
        # query_start_loc is static: [0, tpr, 2*tpr, ..., B*tpr]
        self._base_qsl_gpu = (
            torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.device)
            * tpr
        )
        self._base_qsl_cpu = self._base_qsl_gpu.cpu()
        # Per-req CPU staging — filled from input_batch.
        self._base_staging_last_tok_cpu = torch.zeros(max_bs, dtype=torch.int32)
        self._base_staging_active_m1_cpu = torch.zeros(max_bs, dtype=torch.int32)
        # Track which req has had base prefill done
        self._base_prefilled: set[str] = set()
        # Incremental drafter aug-prefill: per-req high-water mark of how
        # many aug positions are already in the drafter's gid=1 KV cache.
        # First call for a req: full prefill, set to L_aug.
        # Later calls (streaming-session chunks with same rid): forward only
        # [L_prev..L_now-1] using cached KV at [0..L_prev-1].
        # Shared with the engine-side patches so _patched_free wipes the
        # entry when the request finishes — avoids stale L_prev for reused
        # request IDs.
        import vllm.v1.core.kv_cache_utils as _kvu
        self._aug_prefilled: dict[str, tuple[int, int]] = getattr(
            _kvu, "_specsteer_aug_prefilled", None,
        )
        if self._aug_prefilled is None:
            self._aug_prefilled = {}
            _kvu._specsteer_aug_prefilled = self._aug_prefilled
        # Streaming-session chunk-boundary detector. AsyncLLM streaming reuses
        # the same internal_req_id across BFCL turns; each new chunk extends
        # both main + aug contexts in lockstep. We use aug_len growth as the
        # chunk-boundary signal — when aug_ids grows beyond the cached high-
        # water mark for a rid, we invalidate per-rid caches that remember
        # "already done for this rid" without tracking length:
        #   - _aug_bonus_computed: aug-bonus must be recomputed for new aug
        #   - _base_prefilled: base mirror's KV missing chunk-N's main tokens,
        #     forces full re-prefill via slow path next call
        # _aug_prefilled is NOT invalidated — it's a high-water mark and grows
        # monotonically; the merged path forwards only new aug positions.
        self._chunk_seen_aug_len: dict[str, int] = {}
        # T1 flag: True = use fast path for incremental; False = always slow
        self._use_fast_base_fwd = True

        logger.info(
            "SpecSteerProposer initialized: mode=%s β=%.2f γ=%.2f (profile=%s)",
            "strict_target" if self._strict_target_mode else "contrast",
            self.beta, self.gamma, self._profile_enabled,
        )

    def _prof_start(self, name: str):
        if not self._profile_enabled:
            return None
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        pair = [ev, None]
        self._prof_events.setdefault(name, []).append(pair)
        return pair

    def _prof_end(self, pair):
        if pair is None or not self._profile_enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        pair[1] = ev

    def _prof_report_if_due(self):
        if not self._profile_enabled:
            return
        self._prof_step += 1
        if self._prof_step % self._prof_report_every != 0:
            return
        torch.cuda.synchronize()
        parts = []
        for name, pairs in self._prof_events.items():
            total_ms = 0.0
            n = 0
            for s, e in pairs:
                if s is not None and e is not None:
                    total_ms += s.elapsed_time(e)
                    n += 1
            if n > 0:
                parts.append(f"{name}={total_ms/n:.2f}ms (n={n})")
        total_cache_calls = (self._prof_cache_hits + self._prof_cache_misses)
        cache_hit_rate = (
            100 * self._prof_cache_hits / total_cache_calls
            if total_cache_calls > 0 else 0.0
        )
        avg_skip = (
            self._prof_cache_skipped_tokens / max(self._prof_cache_hits, 1)
        )
        logger.info(
            "PROFILE step=%d avg_per_step: %s | cache_hits=%d misses=%d "
            "hit_rate=%.1f%% avg_skipped=%.0f tok",
            self._prof_step, " | ".join(parts),
            self._prof_cache_hits, self._prof_cache_misses,
            cache_hit_rate, avg_skip,
        )
        if self._prof_path and self._prof_steps:
            # Every TP worker sees the same request.  Keep a single artifact
            # without adding a distributed synchronization to the hot path.
            is_rank_zero = (not torch.distributed.is_initialized()
                            or torch.distributed.get_rank() == 0)
            if is_rank_zero:
                import json
                from pathlib import Path
                path = Path(self._prof_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    for record in self._prof_steps:
                        events = record.pop("_events")
                        (start, replay_end, proposal_start, proposal_end, end,
                         target_start, target_end, sampler_start, sampler_end) = events
                        record["replay_ms"] = start.elapsed_time(replay_end)
                        record["proposal_ms"] = proposal_start.elapsed_time(proposal_end)
                        record["drafter_postprocess_ms"] = proposal_end.elapsed_time(end)
                        record["verifier_ms"] = target_start.elapsed_time(target_end)
                        record["sampler_ms"] = sampler_start.elapsed_time(sampler_end)
                        f.write(json.dumps(record, sort_keys=True) + "\n")
        # Reset windows
        self._prof_events = {}
        self._prof_steps = []
        self._prof_cache_hits = 0
        self._prof_cache_misses = 0
        self._prof_cache_skipped_tokens = 0

    def _maybe_invalidate_chunk_caches(self, req_ids, aug_ids_by_idx):
        """Detect a streaming-session chunk boundary by aug_len growth.

        Called from dual_forward Gate A on every step. For each rid we have
        aug_ids for, compare current aug_len vs `_chunk_seen_aug_len[rid]`.
        Growth → new chunk arrived → invalidate per-rid caches that don't
        track length:
          - `_aug_bonus_computed`: aug-bonus was for old (shorter) aug;
             new chunk's longer aug needs fresh bonus.
          - `_base_prefilled`: base mirror's KV is missing the chunk's new
             main tokens; force full re-prefill via slow path next call.

        `_aug_prefilled` is left alone — it's a high-water mark, the merged
        path forwards only the new aug positions (no waste).

        Single-chunk requests (the legacy non-streaming flow) get the same
        behavior because aug_len doesn't grow → no invalidation. Byte-
        identical to the non-streaming BS=1 path.
        """
        if not req_ids:
            return
        if not hasattr(self, "_aug_bonus_computed"):
            self._aug_bonus_computed = set()
        if not hasattr(self, "_base_prefilled"):
            self._base_prefilled = set()
        for i, rid in enumerate(req_ids):
            aug_ids = aug_ids_by_idx.get(i)
            if not aug_ids:
                continue
            cur_len = len(aug_ids)
            prev_len = self._chunk_seen_aug_len.get(rid, -1)
            if cur_len > prev_len and prev_len > 0:
                # Growth from a non-zero baseline = streaming chunk arrived.
                # (prev_len == -1 = first time we see this rid; do NOT
                # invalidate — initial bonus + base prefill is needed exactly
                # once on first call, same as legacy flow.)
                self._aug_bonus_computed.discard(rid)
                self._base_prefilled.discard(rid)
                logger.info(
                    "SpecSteer chunk-boundary: rid=%s aug_len %d → %d, "
                    "invalidated _aug_bonus_computed + _base_prefilled",
                    str(rid)[:20], prev_len, cur_len,
                )
            if cur_len != prev_len:
                self._chunk_seen_aug_len[rid] = cur_len

    @override
    def _get_model(self) -> nn.Module:
        """Load one SLM checkpoint and construct two distinct module trees.

        The compressed/base tree is initialized structurally on ``meta`` and
        then rebound to the drafter's Parameters and buffers.  Its attention
        objects and prefix stay distinct, while checkpoint I/O and physical
        parameter storage occur exactly once.

        The drafter is returned to satisfy the parent contract. The base
        instance is stashed in self.base_model and has its attn layers
        available via self._base_attn_layer_names.
        """
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.model_loader import get_model
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.model_executor.models import supports_multimodal  # noqa: F401
        from vllm.utils.torch_utils import set_default_torch_dtype

        # Snapshot registered attn layer names BEFORE loading anything so
        # we can diff to figure out which belong to each newly loaded model.
        pre_attn = set(self._all_attn_layer_names())

        # 1. Drafter — same path as parent DraftModelProposer._get_model.
        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("draft_model"):
            drafter = get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )
        after_drafter_attn = set(self._all_attn_layer_names())

        after_base_attn = after_drafter_attn
        if uses_compressed_base(self._strict_target_mode):
            # 2. Base — construct the same architecture without touching the
            # checkpoint or allocating a second set of model weights.
            with set_model_tag("specsteer_base"):
                with set_default_torch_dtype(draft_vllm_config.model_config.dtype):
                    with torch.device("meta"):
                        self.base_model = initialize_model(
                            vllm_config=draft_vllm_config,
                            model_config=draft_vllm_config.model_config,
                            prefix="specsteer_base",
                        )
            shared_bytes, shared_buffer_bytes = _share_model_state(
                drafter, self.base_model)
            self.base_model.eval()
            logger.info(
                "AsymSpec: base view structurally initialized without checkpoint "
                "I/O; sharing %.2f GiB parameters and %.2f MiB buffers",
                shared_bytes / (1024**3),
                shared_buffer_bytes / (1024**2),
            )
            after_base_attn = set(self._all_attn_layer_names())

        # Attn-layer bookkeeping: new layers introduced by each load.
        drafter_new_layers = after_drafter_attn - pre_attn
        base_new_layers = after_base_attn - after_drafter_attn
        self._base_attn_layer_names = base_new_layers
        self._hybrid_specsteer = is_hybrid_model_config(
            draft_vllm_config.model_config)
        if self._hybrid_specsteer:
            draft_names = {
                name for name in drafter_new_layers if "draft_model." in name
            }
            base_names = {
                name for name in base_new_layers if "specsteer_base." in name
            }
            if self._strict_target_mode:
                require_strict_target_runtime_layer_set(
                    draft_names, after_base_attn)
            else:
                require_distinct_runtime_layer_sets(draft_names, base_names)
            (self._draft_full_attn_layer_names,
             self._draft_gdn_layer_names) = split_hybrid_layer_names(draft_names)
            (self._base_full_attn_layer_names,
             self._base_gdn_layer_names) = split_hybrid_layer_names(base_names)
            logger.info(
                "AsymSpec hybrid runtime: drafter=%d full-attn + %d GDN; "
                "base=%d full-attn + %d GDN",
                len(self._draft_full_attn_layer_names),
                len(self._draft_gdn_layer_names),
                len(self._base_full_attn_layer_names),
                len(self._base_gdn_layer_names),
            )
        if self._strict_target_mode:
            if any(name.startswith("specsteer_base.")
                   for name in after_base_attn):
                raise RuntimeError("Strict-target must not construct specsteer_base")
            logger.info(
                "SpecSteer strict-target: full drafter=%d layers; "
                "compressed base disabled", len(drafter_new_layers))
        else:
            logger.info(
                "SpecSteer: loaded drafter (%d attn layers) + base (%d attn layers)",
                len(drafter_new_layers), len(base_new_layers),
            )
        return drafter

    def invalidate_requests(self, request_ids):
        import vllm.v1.core.kv_cache_utils as kvu
        for name in ("_specsteer_aug_offsets", "_specsteer_aug_prefilled",
                     "_specsteer_aug_mm_data"):
            for rid in request_ids:
                getattr(kvu, name, {}).pop(rid, None)
        for name in ("_base_prefilled", "_aug_bonus_computed",
                     "_chunk_seen_aug_len", "_h_base_off"):
            state = getattr(self, name, None)
            for rid in request_ids:
                if isinstance(state, set):
                    state.discard(rid)
                elif isinstance(state, dict):
                    state.pop(rid, None)

    def _sync_request_caches(self):
        """Reconstruct worker-local metadata from transported request state."""
        if self.runner is None:
            return
        import vllm.v1.core.kv_cache_utils as kvu
        active = set(self.runner.requests)
        for name in ("_specsteer_aug_offsets", "_specsteer_aug_prefilled",
                     "_specsteer_aug_mm_data"):
            registry = getattr(kvu, name)
            for rid in set(registry) - active:
                registry.pop(rid, None)
        for rid, req in self.runner.requests.items():
            extra = getattr(req.sampling_params, "extra_args", None) or {}
            aug = extra.get("specsteer_aug_prompt_ids")
            if aug:
                kvu._specsteer_aug_offsets[rid] = len(aug) - req.num_prompt_tokens
            if extra.get("specsteer_aug_pixel_values") is not None:
                kvu._specsteer_aug_mm_data.setdefault(rid, {
                    "pixel_values": extra["specsteer_aug_pixel_values"],
                    "image_grid_thw": extra["specsteer_aug_image_grid_thw"],
                })
        # Freed/preempted requests must not retain logical cache state.
        for name in ("_aug_bonus_computed", "_base_prefilled"):
            state = getattr(self, name, None)
            if isinstance(state, set):
                state.intersection_update(active)

    def _all_attn_layer_names(self) -> set[str]:
        """Return the set of attn-layer names currently registered in the
        runtime. Used by _get_model to diff before/after each model load."""
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
        return set(
            get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

    def _hybrid_text_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Use Qwen3.5's native text M-RoPE representation.

        vLLM's Qwen3.5 language-only model defines M-RoPE positions as three
        equal text axes.  The legacy pure-Qwen3 path uses a 1-D arange; only
        normalize it for a detected hybrid model.
        """
        if self._hybrid_specsteer and positions.dim() == 1:
            return text_mrope_positions(positions)
        return positions

    def _build_base_layer_metadata(
        self,
        common_metadata,
        num_reqs: int,
        *,
        use_speculative_gdn_state: bool = True,
    ) -> dict:
        """Build base metadata with each layer's own logical cache group.

        Pure-attention SpecSteer historically reused one base block table for
        every base layer.  That is invalid for Qwen3.5: GDN consumes a Mamba
        state-index table while full attention consumes token KV slots.  The
        native builders own both interpretations, so this method only selects
        the correct group-local table and forwards vLLM's accepted-token state.
        """
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

        metadata: dict[str, object] = {}
        for attn_group in self.base_attn_groups:
            group_metadata = copy(common_metadata)
            gid = attn_group.kv_cache_group_id
            if gid != self.base_kv_cache_gid:
                group_metadata.block_table_tensor = (
                    self.runner.input_batch.block_table[gid].block_table.gpu[:num_reqs]
                )
                # Mamba ignores slot_mapping; full-attention groups use the
                # group's own mapping when they are not the legacy base gid.
                group_slot_mapping = getattr(
                    self.runner.input_batch.block_table[gid], "slot_mapping", None
                )
                group_metadata.slot_mapping = (
                    group_slot_mapping.gpu[:common_metadata.num_actual_tokens]
                    if group_slot_mapping is not None else common_metadata.slot_mapping
                )
            spec = attn_group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[attn_group.layer_names[0]]
            build_kwargs = {"common_prefix_len": 0,
                            "common_attn_metadata": group_metadata,
                            "fast_build": True}
            if isinstance(spec, MambaSpec) and use_speculative_gdn_state:
                # This is the same prior-step acceptance state supplied to
                # native target GDN builders. It selects the committed
                # recurrent state before this speculative step; rejected
                # suffixes remain in speculative state slots.
                build_kwargs["num_accepted_tokens"] = (
                    self.runner.num_accepted_tokens.gpu[:num_reqs])
                build_kwargs["num_decode_draft_tokens_cpu"] = (
                    self.runner.num_decode_draft_tokens.cpu[:num_reqs])
            built = attn_group.get_metadata_builder().build(**build_kwargs)
            for layer_name in attn_group.layer_names:
                metadata[layer_name] = built
        return metadata

    def _build_drafter_layer_metadata(
        self,
        common_metadata,
        num_reqs: int,
        *,
        draft_index: int | None = None,
        gdn_state_seq_lens: torch.Tensor | None = None,
    ) -> dict:
        """Build drafter metadata from each cache group's own block table.

        Qwen3.5 splits full-attention and GDN layers into distinct cache
        groups. Reusing the primary full-attention table for GDN addresses a
        different physical pool and causes recurrent state corruption after a
        few decode steps. Pure-attention models retain their existing layout.
        """
        metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            group_metadata = copy(common_metadata)
            gid = attn_group.kv_cache_group_id
            if gid != self.drafter_kv_cache_gid:
                group_metadata.block_table_tensor = (
                    self.runner.input_batch.block_table[gid].block_table.gpu[
                        :num_reqs
                    ]
                )
            spec = attn_group.kv_cache_spec
            from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[attn_group.layer_names[0]]
            # A hybrid proposal is rebuilt from committed history on every
            # outer step. Its sequential inner drafts intentionally mutate one
            # state slot: the scheduler's aligned state block may change at a
            # token-block boundary, but no native auxiliary-state copy lifecycle
            # owns these two model views. Pin only GDN metadata to the state
            # selected by the committed prefix; token-KV metadata keeps true
            # sequence lengths and positions.
            if isinstance(spec, MambaSpec) and gdn_state_seq_lens is not None:
                group_metadata.seq_lens = gdn_state_seq_lens
                group_metadata._seq_lens_cpu = gdn_state_seq_lens.cpu()
            builder = attn_group.get_metadata_builder()
            if isinstance(spec, MambaSpec):
                # Do not enter vLLM's speculative GDN protocol here: its
                # accepted-token state ownership belongs to the verifier only.
                built = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=group_metadata,
                    fast_build=True,
                )
            elif draft_index is None:
                built = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=group_metadata,
                    fast_build=True,
                )
            else:
                built = builder.build_for_drafting(
                    common_attn_metadata=group_metadata,
                    draft_index=draft_index,
                )
            for layer_name in attn_group.layer_names:
                metadata[layer_name] = built
        return metadata

    def _base_slot_mapping_dict(self, slot_mapping: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return slot mappings only for token-KV base layers.

        A GDN/Mamba group does not consume attention slots: its metadata
        builder turns its group-local block table into recurrent-state indices.
        Feeding it the full-attention slot mapping is not merely redundant;
        the mapping has a different address space and can reach an invalid
        CUDA state slot during the next speculative step.
        """
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

        result: dict[str, torch.Tensor] = {}
        for group in self.base_attn_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[group.layer_names[0]]
            if isinstance(spec, MambaSpec):
                continue
            for layer_name in group.layer_names:
                result[layer_name] = slot_mapping
        return result

    @override
    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        """The drafter and base model may be in different cache groups.
        Validate each of them is internally in a single gid (not both).
        """
        layer_to_gid = {}
        for gid, g in enumerate(kv_cache_config.kv_cache_groups):
            for ln in g.layer_names:
                layer_to_gid[ln] = gid
        drafter_set = {ln for ln in self._draft_attn_layer_names
                       if "draft_model." in ln}
        base_set = {ln for ln in self._draft_attn_layer_names
                    if "specsteer_base." in ln}
        if drafter_set and not self._hybrid_specsteer:
            gids = {layer_to_gid[ln] for ln in drafter_set}
            assert len(gids) == 1, f"drafter in multiple gids: {gids}"
        if base_set and not self._hybrid_specsteer:
            gids = {layer_to_gid[ln] for ln in base_set}
            assert len(gids) == 1, f"base in multiple gids: {gids}"

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        """Keep the drafter in gid=1 and the base model in gid=0 with the LLM.
        Build `draft_attn_groups` (drafter-only) and `base_attn_groups`
        (base-only) with their respective kv_cache_group_ids.
        """
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
        from vllm.v1.worker.utils import AttentionGroup

        self.validate_same_kv_cache_group(kv_cache_config)

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase,  # type: ignore[type-abstract]
        )

        # Locate gids by layer prefix. Pure attention has one group per view;
        # Qwen3.5 has a full-attention group and a separate GDN/Mamba group.
        drafter_set = {ln for ln in self._draft_attn_layer_names
                       if "draft_model." in ln}
        base_set = {ln for ln in self._draft_attn_layer_names
                    if "specsteer_base." in ln}

        def _find_gids(layer_set):
            found = []
            for gid, g in enumerate(kv_cache_config.kv_cache_groups):
                if layer_set & set(g.layer_names):
                    found.append((gid, g.kv_cache_spec))
            return found

        draft_groups = _find_gids(drafter_set)
        base_groups = _find_gids(base_set)
        drafter_gid, drafter_spec = draft_groups[0] if draft_groups else (-1, None)
        base_gid, base_spec = base_groups[0] if base_groups else (-1, None)
        self.kv_cache_gid = drafter_gid           # parent attribute (drafter)
        self.drafter_kv_cache_gid = drafter_gid   # explicit alias
        self.base_kv_cache_gid = base_gid
        self.drafter_kv_cache_gids = {gid for gid, _ in draft_groups}
        self.base_kv_cache_gids = {gid for gid, _ in base_groups}
        if self._hybrid_specsteer and len(draft_groups) < 2:
            raise RuntimeError(
                "Hybrid SpecSteer requires independent full-attention and "
                "GDN cache groups for the full drafter")
        if (self._hybrid_specsteer and not self._strict_target_mode
                and len(base_groups) < 2):
            raise RuntimeError(
                "Hybrid contrast SpecSteer requires independent full-attention "
                "and GDN cache groups for the compressed base")

        def _build_groups(layer_set, located_groups):
            if not layer_set or not located_groups:
                return []
            groups: dict[str, AttentionGroup] = {}
            for gid, spec in located_groups:
                group_layers = sorted(
                    layer_set & set(kv_cache_config.kv_cache_groups[gid].layer_names))
                for layer_name in group_layers:
                    ab = all_attn_layers[layer_name].get_attn_backend()
                    # The gid is part of the key: same backend/spec in separate
                    # logical views must never share mutable metadata buffers.
                    key = f"{gid}:{ab.full_cls_name()}"
                    if key not in groups:
                        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
                        layer_spec = spec
                        if isinstance(spec, UniformTypeKVCacheSpecs):
                            layer_spec = spec.kv_cache_specs[layer_name]
                        kbs = (kernel_block_sizes[gid]
                               if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                               else None)
                        ag = AttentionGroup(
                            backend=ab, layer_names=[layer_name],
                            kv_cache_spec=layer_spec, kv_cache_group_id=gid,
                        )
                        ag.create_metadata_builders(
                            self.vllm_config, self.device, kernel_block_size=kbs,
                        )
                        groups[key] = ag
                    else:
                        groups[key].layer_names.append(layer_name)
            return list(groups.values())

        self.draft_attn_groups = _build_groups(drafter_set, draft_groups)
        self.base_attn_groups = _build_groups(base_set, base_groups)

        if self._strict_target_mode:
            if base_set or base_groups or self.base_attn_groups:
                raise RuntimeError(
                    "Strict-target allocated compressed-base cache groups")

        if self.draft_attn_groups:
            self.block_size = (
                self.draft_attn_groups[0].get_metadata_builder()
                .kv_cache_spec.block_size
            )
        if kernel_block_sizes is not None and drafter_gid >= 0:
            self.block_size = kernel_block_sizes[drafter_gid]


        if self._strict_target_mode:
            logger.info(
                "SpecSteer strict-target cache: drafter gids=%s (%d layers); "
                "compressed-base cache disabled",
                sorted(self.drafter_kv_cache_gids), len(drafter_set))
            main_limit = getattr(self.speculative_config,
                                 "specsteer_main_max_model_len", None)
            full_limit = self.vllm_config.model_config.max_model_len
            logger.info(
                "SpecSteer strict-target: full drafter=%s; verifier=%s; "
                "compressed base=disabled; full context limit=%d; "
                "verifier context limit=%d",
                self.speculative_config.model,
                self.vllm_config.model_config.model,
                full_limit, main_limit or full_limit)
        else:
            logger.info(
                "AsymSpec: drafter_gid=%d (%d layers) + base_gid=%d "
                "(%d layers). Separate block tables will be used.",
                drafter_gid, len(drafter_set), base_gid, len(base_set),
            )

        # The obsolete dual-forward path is disabled. Keep the original
        # drafter forward for _compute_aug_first_bonus.
        self._orig_drafter_forward = self.model.forward

    @torch.no_grad()
    def _base_parallel_verify(
        self,
        drafts_flat: torch.Tensor,  # [K] int64 drafted tokens (BS=1)
        next_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run SLM_base in ONE parallel forward on [last_committed_main, drafts].

        Uses K+1 tokens
        parallel at positions [L-1..L+K-1], returns K+1 logits where the
        first K predict drafts[0..K-1] (consumed by sampler) and the K+1-th
        is a bonus prediction.

        Assumes main-ctx committed tokens are in input_batch.token_ids_cpu
        (what LLM + base should see); drafter's aug substitution via Gate A
        doesn't touch this. For the FIRST call on a new request, extends the
        forward to include the full prompt prefix so base's KV cache is
        properly populated.

        BS=1 only. Returns K+1 logits [K+1, V] or None on failure.
        """
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if self.runner is None or self.base_model is None or not self.base_attn_groups:
            return None
        input_batch = self.runner.input_batch
        B = input_batch.num_reqs
        if B < 1:
            return None

        # Handle drafts_flat: could be [K] (BS=1 flat) or [B*K] (BS>1 flat)
        # Assume draft_token_ids.shape[-1] == K per request; total = B*K.
        total_drafts = int(drafts_flat.numel())
        if total_drafts % B != 0:
            return None
        K = total_drafts // B
        drafts_per_req = drafts_flat.view(B, K).cpu().tolist()
        next_toks_per_req = None
        if next_token_ids is not None:
            next_toks_per_req = next_token_ids.view(-1).cpu().tolist()

        req_ids = getattr(input_batch, "req_ids", None)
        if not req_ids or len(req_ids) < B:
            return None
        if not hasattr(self, "_base_prefilled"):
            self._base_prefilled: set[str] = set()
        initial_base_prefill = self._hybrid_specsteer or any(
            req_ids[i] not in self._base_prefilled for i in range(B)
        )

        # Per-request: build input tokens, positions, decide prefill/incremental
        all_input_tokens = []    # flat list across all requests
        all_positions = []       # flat list
        per_req_num_tokens = []  # how many tokens each request contributes
        per_req_start_pos = []
        per_req_active = []
        per_req_seq_len = []     # cache_after_fwd per request

        for i in range(B):
            req_id_i = req_ids[i]
            num_committed_i = int(input_batch.num_tokens_no_spec[i])
            if num_committed_i < 1:
                return None
            spec_tokens_i = (input_batch.spec_token_ids[i]
                             if hasattr(input_batch, "spec_token_ids") else [])
            active_i = num_committed_i + len(spec_tokens_i)
            # The hybrid fallback deliberately rebuilds GDN state rather than
            # replaying native speculative checkpoints. Its seed must be the
            # *committed* compressed history only: input_batch's speculative
            # suffix can be rejected by the verifier and must never become a
            # recurrent-state prefix on the next round.
            prefix_len_i = num_committed_i if self._hybrid_specsteer else active_i
            token_ids_i = input_batch.token_ids_cpu[i, :prefix_len_i]
            next_tok_i = (next_toks_per_req[i] if next_toks_per_req else None)
            nt_list = [next_tok_i] if next_tok_i is not None else []

            if self._hybrid_specsteer:
                # GDN recurrent state cannot safely use the attention-only
                # rewrite/rollback protocol below. Rebuild the compressed
                # sequence through vLLM's ordinary GDN prefill path for each
                # verification round. This is deliberately conservative but
                # gives the base exact compressed-context logits and leaves no
                # rejected speculative suffix in its recurrent state.
                start_pos_i = 0
                input_tokens_i = list(token_ids_i) + nt_list + drafts_per_req[i]
            elif req_id_i not in self._base_prefilled:
                start_pos_i = 0
                input_tokens_i = (
                    list(token_ids_i) + nt_list + drafts_per_req[i]
                )
                self._base_prefilled.add(req_id_i)
            else:
                # Incremental: writes [active-1 .. active+K]. Prior PV must
                # have populated [0..active-2].
                start_pos_i = active_i - 1
                input_tokens_i = (
                    [int(token_ids_i[-1])] + nt_list + drafts_per_req[i]
                )

            num_tokens_i = len(input_tokens_i)
            cache_after_i = start_pos_i + num_tokens_i
            all_input_tokens.extend(input_tokens_i)
            all_positions.extend(range(start_pos_i, start_pos_i + num_tokens_i))
            per_req_num_tokens.append(num_tokens_i)
            per_req_start_pos.append(start_pos_i)
            per_req_active.append(active_i)
            per_req_seq_len.append(cache_after_i)

        input_ids = torch.tensor(
            all_input_tokens, dtype=torch.int32, device=self.device,
        )
        positions = torch.tensor(
            all_positions, dtype=torch.int64, device=self.device,
        )
        num_tokens = input_ids.shape[0]
        # Legacy single-req bookkeeping (first req, for backward-compat logs).
        num_committed = int(input_batch.num_tokens_no_spec[0])
        active_count = per_req_active[0]
        start_pos = per_req_start_pos[0]

        # Build per-request slot_mapping and concatenate. Block table sliced
        # to [:B] covers all B requests. Each request has its own rows of
        # block ids in block_table_tensor[i].
        blk_tbl_obj = input_batch.block_table[self.base_kv_cache_gid]
        block_size = blk_tbl_obj.block_size
        block_table_full = blk_tbl_obj.block_table.gpu
        block_table_tensor = block_table_full[:B]  # [B, max_blocks]

        # Per-request slot_mapping: for request i, its positions are the
        # slice of global positions at offsets [cumoff_i, cumoff_i+num_i).
        slot_mapping_parts = []
        cum_offsets = [0]
        for i in range(B):
            cum_offsets.append(cum_offsets[-1] + per_req_num_tokens[i])
        for i in range(B):
            lo, hi = cum_offsets[i], cum_offsets[i + 1]
            pos_i = positions[lo:hi]
            pos_int32_i = pos_i.to(torch.int32)
            block_ids_i = block_table_tensor[i, pos_int32_i // block_size]
            slot_i = block_ids_i.to(torch.int64) * block_size + (pos_i % block_size)
            slot_mapping_parts.append(slot_i)
        slot_mapping = torch.cat(slot_mapping_parts, dim=0)

        # CAD for batched forward.
        query_start_loc_cpu = torch.tensor(cum_offsets, dtype=torch.int32)
        query_start_loc = query_start_loc_cpu.to(self.device)
        seq_lens = torch.tensor(per_req_seq_len, dtype=torch.int32, device=self.device)
        num_computed_tokens_cpu = torch.tensor(per_req_start_pos, dtype=torch.int32)
        seq_lens_cpu = torch.tensor(per_req_seq_len, dtype=torch.int32)
        num_prompt_tokens_list = (
            input_batch.num_prompt_tokens_cpu_tensor[:B].tolist()
            if hasattr(input_batch, "num_prompt_tokens_cpu_tensor")
            else [0] * B
        )
        is_prefilling = torch.tensor(
            [per_req_start_pos[i] < num_prompt_tokens_list[i] for i in range(B)],
            dtype=torch.bool, device=self.device,
        )
        max_query_len = max(per_req_num_tokens)
        max_seq_len = max(per_req_seq_len)

        cad = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=B,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
            is_prefilling=is_prefilling,
        )

        # Per-layer attn metadata for base. build() (not build_for_drafting)
        # is the STANDARD prefill/decode path — build_for_drafting is tuned for
        # single-token autoregressive drafting shape and misbehaves on a
        # multi-token parallel forward.
        per_layer_attn_metadata = self._build_base_layer_metadata(
            cad,
            B,
            # The first base pass contains a complete compressed prompt plus
            # proposal tokens. It is a real prefill, not the runner's target
            # speculative batch, so its GDN state must be initialized through
            # the ordinary recurrent prefill path. Reusing the target's
            # accepted-token arrays here misclassifies the whole prompt as a
            # K-token decode and corrupts its state slots asynchronously.
            use_speculative_gdn_state=not (
                self._hybrid_specsteer and initial_base_prefill),
        )

        # GDN layers use their group-local recurrent-state indices from
        # metadata, not token-KV slots. Only expose this mapping to the
        # base full-attention layers.
        slot_mapping_dict = self._base_slot_mapping_dict(slot_mapping)

        if not getattr(self, "_pv_logged", False):
            import logging as _log
            logger.info(
                "SpecSteer parallel_verify: num_tokens=%d start_pos=%d "
                "num_committed=%d K=%d gid=%d block_size=%d block_table_shape=%s "
                "max(slot)=%d input_ids.shape=%s positions.shape=%s "
                "seq_lens=%s",
                num_tokens, start_pos, num_committed, K,
                self.base_kv_cache_gid, block_size, tuple(block_table_tensor.shape),
                int(slot_mapping.max().item()), tuple(input_ids.shape),
                tuple(positions.shape), seq_lens.tolist(),
            )
            self._pv_logged = True

        _diag_use_drafter = getattr(self, "_pv_use_drafter_for_diag", False)
        _model_to_use = self.model if _diag_use_drafter else self.base_model

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            slot_mapping=slot_mapping_dict,
        ):
            ret = _model_to_use(
                input_ids=input_ids,
                positions=self._hybrid_text_positions(positions),
                inputs_embeds=None,
            )
            hidden = ret[0] if isinstance(ret, tuple) else ret

        # DIAGNOSTIC: compare hidden stats to drafter's _draft_logits_per_pos.
        # Drafter's aug_logits[0] came from hidden[L] at drafter's forward.
        # My hidden at position L-1 (= num_committed - 1) should predict
        # the SAME draft token when using drafter model + main context.
        if not getattr(self, "_pv_hidden_diag", False):
            self._pv_hidden_diag = True
            # My prediction for drafts[0] comes from hidden[num_committed - 1]
            # IF start_pos=0 (prefill) and positions=[0..num_tokens-1].
            # Relative index in this forward: num_committed - 1 - start_pos.
            rel_idx = num_committed - 1 - start_pos
            my_hidden_at_L = hidden[rel_idx]
            my_logits_at_L = _model_to_use.compute_logits(my_hidden_at_L[None, :])
            my_top1 = int(my_logits_at_L.argmax().item())
            my_hmean = float(my_hidden_at_L.float().abs().mean().item())
            my_hnorm = float(my_hidden_at_L.float().norm().item())
            # Compare with drafter's own per_pos logits (what super() captured).
            dr = (self._draft_logits_per_pos[0] if self._draft_logits_per_pos else None)
            drafter_top1 = int(dr.argmax().item()) if dr is not None else -1
            logger.info(
                "SpecSteer hidden_diag: rel_idx=%d start_pos=%d num_committed=%d  "
                "my_top1=%d drafter_top1=%d  hidden[L].mean=%.3f .norm=%.3f",
                rel_idx, start_pos, num_committed, my_top1, drafter_top1,
                my_hmean, my_hnorm,
            )

        # Per-request: take the last K+1 hiddens per request segment (each
        # predicts drafts[0..K-1] + bonus). Stack to [B, K+1, V].
        per_req_logits = []
        for i in range(B):
            lo, hi = cum_offsets[i], cum_offsets[i + 1]
            tail_hidden = hidden[hi - (K + 1):hi]  # [K+1, H]
            log_i = self.base_model.compute_logits(tail_hidden)  # [K+1, V]
            per_req_logits.append(log_i)
        logits = torch.stack(per_req_logits, dim=0)  # [B, K+1, V]
        return logits

    @torch.no_grad()
    def _base_forward_fast(
        self,
        drafts_flat: torch.Tensor,  # [B*K] on GPU
        next_token_ids: torch.Tensor | None = None,  # [B] on GPU
    ) -> torch.Tensor | None:
        """T1: fast incremental base parallel verify.

        Pure-attention models retain the legacy warmup rewrite
        ``[last, next, drafts...]``. Hybrid Qwen3.5 models instead use the
        native GDN speculative layout ``[next, drafts...]`` at positions
        ``[active..active+K]``. GDN's native metadata owns K+1 recurrent-state
        indices; feeding it the legacy K+2 layout corrupts those state slots.

        Optimizations vs _base_parallel_verify:
          - No Python lists for input_tokens / positions — use GPU ops
          - No drafts.cpu().tolist() — drafts stay on GPU
          - Pre-allocated buffers (_base_input_ids_buf etc.)
          - Minimal CPU→GPU syncs (2× H2D: staging_last_tok + staging_active_m1)
          - Reuses base_attn_groups metadata builder (same as slow path)

        Falls back to slow path for requests that haven't been prefilled yet.
        BS=1 and BS>1 both supported.
        """
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if self.runner is None or self.base_model is None or not self.base_attn_groups:
            return None
        if self._hybrid_specsteer:
            # See _base_parallel_verify: use an exact native GDN prefill for
            # every round until the custom method has a native recurrent-state
            # commit/rollback integration. The attention-only fast rewrite is
            # invalid for GDN and can leave CUDA state slots corrupt.
            return self._base_parallel_verify(drafts_flat, next_token_ids)
        input_batch = self.runner.input_batch
        B = input_batch.num_reqs
        if B < 1:
            return None

        total_drafts = int(drafts_flat.numel())
        if total_drafts % B != 0:
            return None
        K = total_drafts // B

        req_ids = getattr(input_batch, "req_ids", None)
        if not req_ids or len(req_ids) < B:
            return None

        # FIRST call per req: fall back to slow path (which handles prefill)
        needs_prefill = any(
            req_ids[i] not in self._base_prefilled for i in range(B)
        )
        if needs_prefill:
            result = self._base_parallel_verify(drafts_flat, next_token_ids)
            for i in range(B):
                self._base_prefilled.add(req_ids[i])
            if not hasattr(self, "_t1_fast_counter"):
                self._t1_fast_counter = 0
                self._t1_slow_counter = 0
            self._t1_slow_counter += 1
            if (self._t1_fast_counter + self._t1_slow_counter) % 200 == 1:
                logger.info("T1 counter: fast=%d slow=%d",
                            self._t1_fast_counter, self._t1_slow_counter)
            return result

        # Check buffer capacity
        tpr = self._base_pv_tokens_per_req
        expected_tpr = K + (1 if self._hybrid_specsteer else 2)
        if expected_tpr != tpr or B > self._base_pv_max_bs:
            # Shape changed or batch too large — fallback safely
            return self._base_parallel_verify(drafts_flat, next_token_ids)

        if not hasattr(self, "_t1_fast_counter"):
            self._t1_fast_counter = 0
            self._t1_slow_counter = 0
        self._t1_fast_counter += 1

        total_tokens = B * tpr

        # --- Gather per-req small-int state from CPU side (unavoidable) ---
        # We need active_i = num_tokens_no_spec[i] + len(spec_token_ids[i]).
        # The legacy pure-attention path additionally rewrites active_i - 1.
        last_tok_cpu = self._base_staging_last_tok_cpu[:B]
        active_m1_cpu = self._base_staging_active_m1_cpu[:B]
        for i in range(B):
            num_committed_i = int(input_batch.num_tokens_no_spec[i])
            spec_tokens_i = (
                input_batch.spec_token_ids[i]
                if hasattr(input_batch, "spec_token_ids") else []
            )
            active_i = num_committed_i + len(spec_tokens_i)
            if active_i < 1:
                return self._base_parallel_verify(drafts_flat, next_token_ids)
            active_m1_cpu[i] = active_i - 1
            last_tok_cpu[i] = int(input_batch.token_ids_cpu[i, active_i - 1])

        # Send CPU → GPU once (small)
        last_tok_gpu = last_tok_cpu.to(self.device, non_blocking=True)
        active_m1_gpu = active_m1_cpu.to(self.device, non_blocking=True).to(torch.int64)

        # --- Fill input_ids buffer on GPU ---
        # Hybrid layout: [next_tok, drafts[0..K-1]].  The pure-attention
        # compatibility layout retains [last_tok, next_tok, drafts...].
        input_ids_view = self._base_input_ids_buf[:total_tokens].view(B, tpr)
        next_tokens = (next_token_ids.view(-1).to(torch.int32)
                       if next_token_ids is not None else last_tok_gpu)
        if self._hybrid_specsteer:
            input_ids_view[:, 0].copy_(next_tokens)
            input_ids_view[:, 1:].copy_(drafts_flat.view(B, K).to(torch.int32))
        else:
            input_ids_view[:, 0].copy_(last_tok_gpu)
            input_ids_view[:, 1].copy_(next_tokens)
            input_ids_view[:, 2:].copy_(drafts_flat.view(B, K).to(torch.int32))
        input_ids_flat = self._base_input_ids_buf[:total_tokens]

        # --- Fill positions buffer on GPU ---
        # Hybrid GDN starts at the first uncomputed token; pure attention
        # retains the active-1 warmup rewrite.
        positions_view = self._base_positions_buf[:total_tokens].view(B, tpr)
        offsets = torch.arange(0, tpr, dtype=torch.int64, device=self.device)
        position_start = active_m1_gpu + (1 if self._hybrid_specsteer else 0)
        positions_view.copy_(position_start[:, None] + offsets[None, :])
        positions_flat = self._base_positions_buf[:total_tokens]

        # --- Fill seq_lens buffer on GPU ---
        # Both layouts end at active + K + 1.
        seq_lens = self._base_seq_lens_buf[:B]
        seq_lens.copy_(position_start.to(torch.int32))
        seq_lens.add_(tpr)
        seq_lens_cpu = position_start.cpu().to(torch.int32) + tpr

        # --- query_start_loc: static pre-computed ---
        query_start_loc = self._base_qsl_gpu[:B + 1]
        query_start_loc_cpu = self._base_qsl_cpu[:B + 1]

        # --- num_computed_tokens_cpu ---
        num_computed_cpu = position_start.cpu().to(torch.int32)

        # --- block_table + slot_mapping (gid=0 base shares with LLM) ---
        blk_tbl_obj = input_batch.block_table[self.base_kv_cache_gid]
        block_size = blk_tbl_obj.block_size
        block_table_full = blk_tbl_obj.block_table.gpu
        block_table_tensor = block_table_full[:B]

        # slot_mapping: block_ids * block_size + (pos % block_size)
        positions_int32 = positions_view.to(torch.int32)
        block_indices = positions_int32 // block_size  # [B, tpr]
        block_ids = block_table_tensor.gather(1, block_indices)  # [B, tpr]
        slot_mapping_view = self._base_slot_mapping_buf[:total_tokens].view(B, tpr)
        slot_mapping_view.copy_(
            block_ids.to(torch.int64) * block_size
            + (positions_view % block_size)
        )
        slot_mapping_flat = self._base_slot_mapping_buf[:total_tokens]

        is_prefilling = torch.zeros(B, dtype=torch.bool, device=self.device)
        max_query_len = tpr
        max_seq_len = int(seq_lens_cpu.max().item())

        cad = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_cpu,
            num_reqs=B,
            num_actual_tokens=total_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping_flat,
            causal=True,
            is_prefilling=is_prefilling,
        )

        # Build per-layer attn_metadata for base
        per_layer_attn_metadata = self._build_base_layer_metadata(cad, B)

        slot_mapping_dict = self._base_slot_mapping_dict(slot_mapping_flat)

        with set_forward_context(
            per_layer_attn_metadata, self.vllm_config,
            num_tokens=total_tokens,
            slot_mapping=slot_mapping_dict,
        ):
            ret = self.base_model(
                input_ids=input_ids_flat,
                positions=self._hybrid_text_positions(positions_flat),
                inputs_embeds=None,
            )
            hidden = ret[0] if isinstance(ret, tuple) else ret

        # Slice last K+1 per req, compute logits
        # hidden shape: [B * tpr, H], tpr=K+2
        hidden_reshape = hidden.view(B, tpr, -1)
        tail_hidden = (hidden_reshape if self._hybrid_specsteer
                       else hidden_reshape[:, 1:, :])
        tail_flat = tail_hidden.reshape(B * (K + 1), -1)
        logits = self.base_model.compute_logits(tail_flat)
        return logits.view(B, K + 1, -1)

    @torch.no_grad()
    def _compute_aug_first_bonus(
        self,
        items: "list[tuple[int, list[int]]] | tuple[int, list[int]] | list[int]",
        req_index: int | None = None,
    ) -> "list[int | None] | int | None":
        """Compute aug SLM's prediction at position L for one or more requests.

        Purpose: fix the vLLM async-spec-decode architectural gap where
        `next_token_ids[i]` is LLM's main-ctx bonus, which ANCHORS drafter
        to main-context predictions even under the context swap. We replace
        next_token_ids[i] with
        aug SLM's own prediction at position L_i.

        Runs a SINGLE batched drafter forward over the concatenation of all
        N requests' aug prefixes [0..L_i-1]; extracts each request's last
        position logit argmax. Per-request drafter KV gets written at its
        own positions [0..L_i-1] via per-request block_table indexing.

        Two calling conventions (overloaded for BS=1 backward-compat):

        (A) Batched (BS≥1, preferred):
            items = [(req_idx_0, aug_ids_0), (req_idx_1, aug_ids_1), ...]
            req_index is ignored.
            Returns: [bonus_0, bonus_1, ...] (None entries on failure)

        (B) Single-request compatibility form:
            items = aug_ids_list, req_index = int  (or items=(req_idx, aug_ids))
            Returns: bonus_int (or None on failure)

        BS=1 byte-identity: at N=1 the batched code path produces identical
        tensor shapes to the single-request code (same input_ids shape (L,),
        same positions arange(L), same block_table[req_idx:req_idx+1] slice,
        same query_start_loc=[0,L], same per_layer_attn_metadata builder call,
        same drafter forward args). PyTorch selects the same kernels for the
        same shapes and numerical behavior.
        """
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        # Normalize the single-request calling convention to the batched form.
        single_call = False
        if req_index is not None:
            # Legacy form: items is the aug_ids list, req_index is int
            items = [(req_index, items)]
            single_call = True
        elif isinstance(items, tuple) and len(items) == 2 and isinstance(items[0], int):
            # Convenience: single (req_idx, aug_ids) tuple
            items = [items]
            single_call = True
        # Else: items is already list[(req_idx, aug_ids)]

        if self.runner is None or not self.draft_attn_groups or not items:
            return None if single_call else [None] * len(items)
        input_batch = self.runner.input_batch
        N = len(items)

        # Validate each request and filter out invalid ones (None placeholder).
        valid: list[tuple[int, int, list[int]]] = []  # (slot_in_items, req_idx, aug_ids)
        for slot, (req_idx, aug_ids) in enumerate(items):
            if req_idx >= input_batch.num_reqs or len(aug_ids) < 1:
                continue
            valid.append((slot, req_idx, aug_ids))
        if not valid:
            return None if single_call else [None] * N

        # Per-request lengths.
        Ls = [len(a) for _, _, a in valid]
        sum_L = sum(Ls)
        max_L = max(Ls)
        # Per-request offset within the concatenated batch.
        offsets = [0]
        for L in Ls:
            offsets.append(offsets[-1] + L)
        # offsets[i]..offsets[i+1] is request i's slice.

        # ---- Build batched ragged tensors (single code path, N≥1) ----
        # At N=1, all the lists/cats/index_selects below have length 1 and
        # produce shape-equivalent tensors to single-request allocations.
        # PyTorch may pick different memory layouts (contiguous copy vs view)
        # but downstream kernels read element values → token output identical.
        all_input_ids = torch.cat([
            torch.tensor(aug_ids, dtype=torch.int32, device=self.device)
            for _, _, aug_ids in valid
        ], dim=0)
        all_positions = torch.cat([
            torch.arange(0, L, dtype=torch.int64, device=self.device) for L in Ls
        ], dim=0)

        # Per-request block_table rows, gathered via advanced indexing.
        blk_tbl_obj = input_batch.block_table[self.drafter_kv_cache_gid]
        block_size = blk_tbl_obj.block_size
        req_indices = torch.tensor(
            [r for _, r, _ in valid], dtype=torch.int64, device=self.device,
        )
        block_table_tensor = blk_tbl_obj.block_table.gpu[req_indices]

        # slot_mapping: per-request gather block ids using each req's positions
        # and its own row of block_table_tensor.
        slot_pieces = []
        for i, (_, _, aug_ids) in enumerate(valid):
            L = Ls[i]
            row = block_table_tensor[i]
            local_pos = torch.arange(0, L, dtype=torch.int64, device=self.device)
            block_ids = row[local_pos.to(torch.int32) // block_size]
            slot = block_ids.to(torch.int64) * block_size + (local_pos % block_size)
            slot_pieces.append(slot)
        slot_mapping = torch.cat(slot_pieces, dim=0)

        # CAD fields. At N=1, query_start_loc=[0, L].
        query_start_loc_cpu = torch.tensor(offsets, dtype=torch.int32)
        query_start_loc = query_start_loc_cpu.to(self.device)
        seq_lens = torch.tensor(Ls, dtype=torch.int32, device=self.device)
        seq_lens_cpu = torch.tensor(Ls, dtype=torch.int32)
        num_computed_tokens_cpu = torch.tensor([0] * len(valid), dtype=torch.int32)
        is_prefilling = torch.tensor(
            [True] * len(valid), dtype=torch.bool, device=self.device,
        )
        cad = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens, _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=len(valid), num_actual_tokens=sum_L,
            max_query_len=max_L, max_seq_len=max_L,
            block_table_tensor=block_table_tensor, slot_mapping=slot_mapping,
            causal=True, is_prefilling=is_prefilling,
        )

        # fast_build=True skips AOT scheduler_metadata precompute (FA3 kernel
        # computes internally; safe across N). Hybrid groups must still use
        # their own physical block tables.
        per_layer_attn_metadata = self._build_drafter_layer_metadata(cad, N)
        drafter_layers = [n for n in self._draft_attn_layer_names
                          if "draft_model." in n]
        slot_mapping_dict = {n: slot_mapping for n in drafter_layers}

        # Optionally compute multimodal input embeddings.
        _vlmm_embeds = self._vlmm_compute_aug_inputs_embeds(valid, all_input_ids)
        # An M-RoPE drafter requires (3, N) positions. For multimodal
        # aug prompts (image+text), use proper image-aware mrope positions
        # (image patch tokens get spatial h/w coords on axes 1,2). Falls back
        # to broadcast for text-only.
        if getattr(self, "uses_mrope", False):
            mrope_pos = self._vlmm_get_aug_mrope_positions(valid, all_input_ids)
            if mrope_pos is not None:
                all_positions = mrope_pos
            elif all_positions.dim() == 1:
                all_positions = all_positions.unsqueeze(0).expand(3, -1).contiguous()
        with set_forward_context(
            per_layer_attn_metadata, self.vllm_config,
            num_tokens=sum_L, slot_mapping=slot_mapping_dict,
        ):
            if _vlmm_embeds is not None:
                ret = self._orig_drafter_forward(
                    input_ids=None, positions=all_positions,
                    inputs_embeds=_vlmm_embeds,
                )
            else:
                ret = self._orig_drafter_forward(
                    input_ids=all_input_ids, positions=all_positions, inputs_embeds=None,
                )
            dh = ret[0] if isinstance(ret, tuple) else ret
        # dh shape: (sum_L, H). Gather last position per request via index_select.
        # At N=1 this returns a (1, H) view of dh's last row — same data as
        # selecting `dh[-1:]`; downstream compute_logits receives the same row.
        last_indices = torch.tensor(
            [offsets[i + 1] - 1 for i in range(len(valid))],
            dtype=torch.int64, device=self.device,
        )
        last_hidden = dh.index_select(0, last_indices)
        logits = self.model.compute_logits(last_hidden)
        argmax_per_req = logits.argmax(dim=-1).tolist()  # list[int], length=len(valid)

        # Map back to original items order (some may be invalid → None).
        results: list[int | None] = [None] * N
        for j, (slot, _, _) in enumerate(valid):
            results[slot] = int(argmax_per_req[j])

        if single_call:
            return results[0]
        return results

    @torch.no_grad()
    def _profile_event(self, label: str = ""):
        """Create + record a CUDA event for fine-grained profiling.
        Returns the event. Caller stores in a list to compute deltas later.
        """
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    @torch.no_grad()
    def _merged_aug_prefill_and_kdecode(
        self,
        items: list[tuple[int, list[int]]],
        K: int,
        next_token_ids: torch.Tensor,
    ) -> "torch.Tensor | None":
        """Run full-context drafter prefill plus K incremental decodes.

        This merges the bonus and draft-prefill work, saving one full-context
        drafter prefill per speculation step.

        Prefill stage forwards the full prompt plus every token already
        committed by the verifier, including ``next_token_ids``. Its last
        hidden state predicts the first draft token.  The verifier, not the
        drafter, owns the newly committed token.

        Decode stage forwards prior draft tokens only for positions 2..K.

        Returns: draft_token_ids of shape (N, K) where N = len(valid items).
        Returns None on failure.

        BS≥1: per-req L_i and per-req block tables. Each iter forwards N
        tokens (one per req), each at its own position.
        """
        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if self.runner is None or not self.draft_attn_groups or not items:
            return None
        input_batch = self.runner.input_batch
        N = len(items)

        # Validate
        valid: list[tuple[int, int, list[int]]] = []
        for slot, (req_idx, aug_ids) in enumerate(items):
            if req_idx >= input_batch.num_reqs or len(aug_ids) < 1:
                continue
            valid.append((slot, req_idx, aug_ids))
        if not valid:
            return None
        if next_token_ids.numel() < input_batch.num_reqs:
            raise RuntimeError("Hybrid SpecSteer is missing verifier next_token_ids")

        # === PROFILING: per-phase CUDA events (buffered; no per-step sync) ===
        _PROF = self._profile_enabled
        prof_evts: list[tuple[str, "torch.cuda.Event"]] = []
        if _PROF:
            prof_evts.append(("start", self._profile_event()))

        # A recurrent GDN state cannot be left at the end of an uncommitted
        # speculative suffix.  The attention-only implementation caches the
        # full drafter prefill and advances it through K candidates, which is
        # safe for paged KV (the scheduler rolls the block table forward) but
        # not for Qwen3.5's in-place recurrent state.  Reconstruct the full
        # drafter sequence for each hybrid proposal round instead:
        #
        #   original full prompt + committed main-path completion
        #
        # The main input batch is authoritative for the committed completion;
        # its prompt prefix is compressed, so only its suffix is appended to
        # the original/full prompt carried in extra_args.  This deliberately
        # trades prefill work for exact state semantics until the native
        # aligned Mamba checkpoint path is used end-to-end by SpecSteer.
        if self._hybrid_specsteer:
            rebuilt_valid: list[tuple[int, int, list[int]]] = []
            prompt_counts = getattr(input_batch, "num_prompt_tokens_cpu_tensor", None)
            for slot, req_idx, aug_prompt_ids in valid:
                if prompt_counts is None:
                    raise RuntimeError(
                        "Hybrid SpecSteer requires prompt-token bookkeeping "
                        "to rebuild the drafter GDN state"
                    )
                main_prompt_len = int(prompt_counts[req_idx])
                committed_len = int(input_batch.num_tokens_no_spec[req_idx])
                if committed_len < main_prompt_len:
                    raise RuntimeError(
                        "Hybrid SpecSteer received a committed length before "
                        "the compressed prompt boundary"
                    )
                committed_suffix = input_batch.token_ids_cpu[
                    req_idx, main_prompt_len:committed_len
                ].tolist()
                # ``num_tokens_no_spec`` ends immediately before the target
                # sampler's just-selected next token.  Append that token once;
                # replacing it with a drafter prediction breaks the committed
                # history and was the source of decode-step divergence.
                verifier_next = int(next_token_ids.view(-1)[req_idx].item())
                rebuilt_valid.append((
                    slot, req_idx,
                    list(aug_prompt_ids) + committed_suffix + [verifier_next],
                ))
            valid = rebuilt_valid
            # A cached prefix ends after speculative candidates, not after the
            # accepted continuation.  It is invalid by construction for GDN.
            self._aug_prefilled.clear()

        # Per-request lengths and offsets
        Ls = [len(a) for _, _, a in valid]
        sum_L = sum(Ls)
        max_L = max(Ls)
        offsets = [0]
        for L in Ls:
            offsets.append(offsets[-1] + L)

        # Incrementally prefill the full context and obtain the bonus.
        # Per-req start position from _aug_prefilled cache. First call for a
        # given req_id does full prefill [0..L-1]; subsequent calls (within
        # the same streaming session) only forward [L_prev..L-1] using cached
        # KV at [0..L_prev-1]. New rid = new session = full prefill.
        # Cache value is (L_prev, hash_of_aug_ids[:L_prev]) — verifies the
        # cached prefix actually matches the current aug_ids[:L_prev] before
        # trusting it. On mismatch: drop cache (L_prev = 0). See cache-key-
        # redesign comment near _aug_prefilled definition for why.
        req_ids_batch = getattr(input_batch, "req_ids", None) or []
        per_req_start = []
        per_req_inc_len = []
        for slot, req_idx, aug_ids in valid:
            L_now = len(aug_ids)
            rid = req_ids_batch[req_idx] if req_idx < len(req_ids_batch) else None
            L_prev = 0
            cached = (
                None if self._hybrid_specsteer
                else self._aug_prefilled.get(rid) if rid else None
            )
            if cached is not None:
                cached_L, cached_hash = cached
                # Aug must be append-only relative to the cached prefix.
                if 0 < cached_L <= L_now:
                    cur_prefix_hash = hash(tuple(aug_ids[:cached_L]))
                    if cur_prefix_hash == cached_hash:
                        L_prev = cached_L  # cache valid → incremental
                    else:
                        # Prefix mutated → KV is stale. Drop cache + log.
                        # Going to full prefill restores correctness.
                        logger.warning(
                            "SpecSteer aug-cache prefix MISMATCH for rid=%s "
                            "(cached_L=%d L_now=%d): KV stale, full re-prefill",
                            str(rid)[:24], cached_L, L_now,
                        )
            # Telemetry: classify hit (skipped >0 tokens) vs miss (full prefill).
            if L_prev > 0:
                self._prof_cache_hits += 1
                self._prof_cache_skipped_tokens += L_prev
            else:
                self._prof_cache_misses += 1
            # SHAPE-0 GUARD: if cache hit gives L_prev == L_now (aug didn't
            # grow — happens when aug is at the bench's truncation cap), the
            # incremental forward would have shape 0, which trips vLLM's
            # torch.compile dynamic-shape range assertion `Shape: 0 out of
            # considered ranges: [(1, max_model_len)]`. Force re-forward of
            # the last 1 position so the bonus hidden state is available.
            # Wasteful (1 extra forward per affected step) but correct;
            # byte-identical for byte-id tests because they don't hit the cap.
            if L_prev >= L_now and L_now >= 1:
                L_prev = L_now - 1
            per_req_start.append(L_prev)
            per_req_inc_len.append(L_now - L_prev)

        sum_inc = sum(per_req_inc_len)
        # Per-req incremental input: aug_ids[L_prev..L_now-1]
        all_input_ids = torch.cat([
            torch.tensor(
                aug_ids[per_req_start[i]:],
                dtype=torch.int32, device=self.device,
            )
            for i, (_, _, aug_ids) in enumerate(valid)
        ], dim=0)
        # Per-req incremental positions: [L_prev..L_now-1]
        all_positions = torch.cat([
            torch.arange(
                per_req_start[i], per_req_start[i] + per_req_inc_len[i],
                dtype=torch.int64, device=self.device,
            )
            for i in range(len(valid))
        ], dim=0)

        blk_tbl_obj = input_batch.block_table[self.drafter_kv_cache_gid]
        block_size = blk_tbl_obj.block_size
        req_indices = torch.tensor(
            [r for _, r, _ in valid], dtype=torch.int64, device=self.device,
        )
        block_table_tensor = blk_tbl_obj.block_table.gpu[req_indices]

        # slot_mapping for INCREMENTAL prefill: only positions [L_prev..L-1]
        slot_pieces = []
        for i, (_, _, aug_ids) in enumerate(valid):
            L_prev_i = per_req_start[i]
            inc_len_i = per_req_inc_len[i]
            row = block_table_tensor[i]
            local_pos = torch.arange(
                L_prev_i, L_prev_i + inc_len_i,
                dtype=torch.int64, device=self.device,
            )
            block_ids = row[local_pos.to(torch.int32) // block_size]
            slot = block_ids.to(torch.int64) * block_size + (local_pos % block_size)
            slot_pieces.append(slot)
        slot_mapping_prefill = torch.cat(slot_pieces, dim=0) if slot_pieces else \
            torch.empty(0, dtype=torch.int64, device=self.device)

        # CAD for INCREMENTAL prefill (or full if L_prev=0)
        # offsets are over the incremental tokens only
        offsets_inc = [0]
        for inc_len in per_req_inc_len:
            offsets_inc.append(offsets_inc[-1] + inc_len)
        query_start_loc_cpu = torch.tensor(offsets_inc, dtype=torch.int32)
        query_start_loc = query_start_loc_cpu.to(self.device)
        # seq_lens = total L (cached + new), num_computed = L_prev
        seq_lens = torch.tensor(Ls, dtype=torch.int32, device=self.device)
        seq_lens_cpu = torch.tensor(Ls, dtype=torch.int32)
        num_computed_tokens_cpu = torch.tensor(per_req_start, dtype=torch.int32)
        # is_prefilling True for any req where this is the first call (L_prev=0)
        # OR there are still incremental tokens to add. Both cases: prefill-style.
        is_prefilling = torch.tensor(
            [True] * len(valid), dtype=torch.bool, device=self.device,
        )
        max_query_len_inc = max(per_req_inc_len) if per_req_inc_len else 1
        cad_prefill = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens, _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=len(valid), num_actual_tokens=sum_inc,
            max_query_len=max_query_len_inc, max_seq_len=max_L,
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping_prefill,
            causal=True, is_prefilling=is_prefilling,
        )

        per_layer_attn_metadata = self._build_drafter_layer_metadata(
            cad_prefill, len(valid))
        # GDN groups consume their group-local recurrent-state indices from
        # their metadata.  They must never receive token-KV slot mappings.
        drafter_layers = [
            n for n in self._draft_full_attn_layer_names
            if "draft_model." in n
        ] if self._hybrid_specsteer else [
            n for n in self._draft_attn_layer_names if "draft_model." in n
        ]
        slot_mapping_dict = {n: slot_mapping_prefill for n in drafter_layers}

        if _PROF: prof_evts.append(("rebuild_setup_done", self._profile_event()))
        # Compute multimodal embeddings for the prefill stage when present.
        _vlmm_embeds = self._vlmm_compute_aug_inputs_embeds(valid, all_input_ids)
        if getattr(self, "uses_mrope", False):
            mrope_pos = self._vlmm_get_aug_mrope_positions(valid, all_input_ids)
            if mrope_pos is not None:
                all_positions = mrope_pos
            elif all_positions.dim() == 1:
                all_positions = all_positions.unsqueeze(0).expand(3, -1).contiguous()
        with set_forward_context(
            per_layer_attn_metadata, self.vllm_config,
            num_tokens=sum_inc, slot_mapping=slot_mapping_dict,
        ):
            if _vlmm_embeds is not None:
                ret = self._orig_drafter_forward(
                    input_ids=None, positions=all_positions,
                    inputs_embeds=_vlmm_embeds,
                )
            else:
                ret = self._orig_drafter_forward(
                    input_ids=all_input_ids, positions=all_positions, inputs_embeds=None,
                )
            dh = ret[0] if isinstance(ret, tuple) else ret
        if _PROF: prof_evts.append(("rebuild_forward_done", self._profile_event()))

        # The last committed verifier token predicts draft position zero.
        last_indices = torch.tensor(
            [offsets_inc[i + 1] - 1 for i in range(len(valid))],
            dtype=torch.int64, device=self.device,
        )
        last_hidden = dh.index_select(0, last_indices)
        first_logits = self.model.compute_logits(last_hidden)
        first_drafts = first_logits.argmax(dim=-1)  # (N_valid,) int

        # Update _aug_prefilled high-water mark for next streaming chunk.
        # Store (L, hash) so the next call can verify the cached prefix
        # matches before trusting it (see comment at lookup site).
        for slot, req_idx, aug_ids in valid:
            rid = req_ids_batch[req_idx] if req_idx < len(req_ids_batch) else None
            if rid and not self._hybrid_specsteer:
                self._aug_prefilled[rid] = (
                    len(aug_ids), hash(tuple(aug_ids)),
                )
        if _PROF: prof_evts.append(("proposal_start", self._profile_event()))
        # The prefill produces draft zero.  Only later draft positions need a
        # one-token decode.  Keep GDN on the committed-prefix state block
        # throughout this inner loop; the outer round always re-prefills it.
        draft_token_ids_list = [first_drafts]
        self._draft_logits_per_pos.append(first_logits.detach())
        self._base_logits_per_pos.append(first_logits.detach())
        last_tokens = first_drafts
        gdn_anchor_seq_lens = torch.tensor(
            Ls, dtype=torch.int32, device=self.device,
        )

        for iter_idx in range(1, K):
            if _PROF: prof_evts.append((f"k{iter_idx}_start", self._profile_event()))
            # Per-req current position (L_i + iter_idx) and slot
            cur_positions = torch.tensor(
                [Ls[i] + iter_idx for i in range(len(valid))],
                dtype=torch.int64, device=self.device,
            )
            slot_pieces_dec = []
            for i in range(len(valid)):
                pos = Ls[i] + iter_idx
                row = block_table_tensor[i]
                bid = int(row[pos // block_size].item())
                slot_pieces_dec.append(bid * block_size + (pos % block_size))
            slot_mapping_dec = torch.tensor(
                slot_pieces_dec, dtype=torch.int64, device=self.device,
            )

            # CAD for decode iter
            qsl = torch.arange(0, len(valid) + 1, dtype=torch.int32, device=self.device)
            qsl_cpu = torch.arange(0, len(valid) + 1, dtype=torch.int32)
            seq_lens_iter = torch.tensor(
                [Ls[i] + iter_idx + 1 for i in range(len(valid))],
                dtype=torch.int32, device=self.device,
            )
            seq_lens_cpu_iter = torch.tensor(
                [Ls[i] + iter_idx + 1 for i in range(len(valid))],
                dtype=torch.int32,
            )
            num_computed_iter = torch.tensor(
                [Ls[i] + iter_idx for i in range(len(valid))],
                dtype=torch.int32,
            )
            is_prefilling_iter = torch.tensor(
                [False] * len(valid), dtype=torch.bool, device=self.device,
            )
            cad_dec = CommonAttentionMetadata(
                query_start_loc=qsl,
                query_start_loc_cpu=qsl_cpu,
                seq_lens=seq_lens_iter, _seq_lens_cpu=seq_lens_cpu_iter,
                _num_computed_tokens_cpu=num_computed_iter,
                num_reqs=len(valid), num_actual_tokens=len(valid),
                max_query_len=1, max_seq_len=int(seq_lens_cpu_iter.max()),
                block_table_tensor=block_table_tensor,
                slot_mapping=slot_mapping_dec,
                causal=True, is_prefilling=is_prefilling_iter,
            )
            per_layer_attn_metadata_dec = self._build_drafter_layer_metadata(
                cad_dec,
                len(valid),
                draft_index=iter_idx,
                gdn_state_seq_lens=gdn_anchor_seq_lens,
            )
            slot_mapping_dict_dec = {
                n: slot_mapping_dec for n in drafter_layers
            }

            if _PROF: prof_evts.append((f"k{iter_idx}_meta_done", self._profile_event()))
            input_ids_dec = last_tokens.to(torch.int32)
            # Use drafter-correct M-RoPE positions for K-token decoding.
            # cur_positions = [Ls[i] + iter_idx for i in range(N)] uses
            # drafter's own context length Ls[i] = len(aug_ids[i]). For mm reqs
            # we must shift by mrope_position_delta to get drafter's mrope axes.
            if getattr(self, "uses_mrope", False):
                import vllm.v1.core.kv_cache_utils as _kvu
                _mm_cache = getattr(_kvu, "_specsteer_aug_mm_data", {})
                _input_batch = self.runner.input_batch if self.runner else None
                # Build per-req shifts (mrope_max - (Ls[i] - 1)) to convert
                # linear text-end-pos to mrope_max-anchored pos. For text-only
                # reqs, shift = 0.
                shifts = []
                for i, (_, req_idx, _) in enumerate(valid):
                    rid = (_input_batch.req_ids[req_idx] if _input_batch and req_idx < len(_input_batch.req_ids) else None)
                    mm = _mm_cache.get(rid, {}) if rid else {}
                    mrope_max = mm.get("mrope_max", None)
                    if mrope_max is None:
                        shifts.append(0)
                    else:
                        # cur_positions[i] = Ls[i] + iter_idx → corresponds to
                        # drafter token at (Ls[i] + iter_idx). Drafter mrope
                        # for this token is mrope_max + iter_idx + 1.
                        # Shift = mrope_max + 1 - Ls[i]
                        shifts.append(mrope_max + 1 - Ls[i])
                shift_t = torch.tensor(shifts, dtype=cur_positions.dtype, device=cur_positions.device)
                cur_pos_drafter = cur_positions + shift_t
                pos_dec = cur_pos_drafter.unsqueeze(0).expand(3, -1).contiguous()
            else:
                pos_dec = cur_positions
            with set_forward_context(
                per_layer_attn_metadata_dec, self.vllm_config,
                num_tokens=len(valid), slot_mapping=slot_mapping_dict_dec,
            ):
                ret_dec = self._orig_drafter_forward(
                    input_ids=input_ids_dec,
                    positions=pos_dec,
                    inputs_embeds=None,
                )
                dh_dec = ret_dec[0] if isinstance(ret_dec, tuple) else ret_dec
            if _PROF: prof_evts.append((f"k{iter_idx}_fwd_done", self._profile_event()))

            # Sample one new draft per req
            draft_logits = self.model.compute_logits(dh_dec)
            # Append to _draft_logits_per_pos for downstream Path B/Gate A
            # consumption (gpu_model_runner asserts non-empty list).
            self._draft_logits_per_pos.append(draft_logits.detach())
            self._base_logits_per_pos.append(draft_logits.detach())  # placeholder
            new_drafts = draft_logits.argmax(dim=-1)  # (N_valid,) int
            draft_token_ids_list.append(new_drafts)
            last_tokens = new_drafts

        if _PROF: prof_evts.append(("proposal_done", self._profile_event()))

        # Stack: (K, N_valid) → transpose to (N_valid, K)
        draft_per_req_K = torch.stack(draft_token_ids_list, dim=0).t()

        # Map back to original (N, K) order — invalid items get zeros (will be
        # filtered downstream)
        draft_full = torch.zeros((N, K), dtype=torch.int32, device=self.device)
        for j, (slot, _, _) in enumerate(valid):
            draft_full[slot] = draft_per_req_K[j].to(torch.int32)

        if _PROF:
            # A profile window synchronizes only in _prof_report_if_due().
            # Keep CUDA events, plus CPU-side work counts, until then.
            prof_evts.append(("end", self._profile_event()))
            event_by_name = dict(prof_evts)
            start = event_by_name["start"]
            replay_end = event_by_name["rebuild_forward_done"]
            proposal_start = event_by_name["proposal_start"]
            proposal_end = event_by_name["proposal_done"]
            end = event_by_name["end"]
            # The target forward and strict sampler immediately precede this
            # proposal round.  They are recorded by GPUModelRunner; retain
            # their already-complete event pairs to make a logical outer-step
            # record without forcing a device synchronization here.
            target_pair = self._prof_events.get("verifier_forward", [])[-1]
            sampler_pair = self._prof_events.get("strict_sampler", [])[-1]
            def pair(name, first, second):
                self._prof_events.setdefault(name, []).append([first, second])
            pair("drafter_rebuild", start, replay_end)
            pair("drafter_proposal", proposal_start, proposal_end)
            pair("drafter_postprocess", proposal_end, end)
            committed_lengths = [int(input_batch.num_tokens_no_spec[r])
                                 for _, r, _ in valid]
            main_prompt_lengths = [int(input_batch.num_prompt_tokens_cpu_tensor[r])
                                   for _, r, _ in valid]
            repeated = sum(
                L for L, committed, prompt in zip(Ls, committed_lengths,
                                                   main_prompt_lengths)
                if committed > prompt
            )
            self._prof_steps.append({
                "kind": "drafter_step",
                "step": self._prof_step + 1,
                "requests": len(valid),
                "K": K,
                "committed_length_before": max(committed_lengths),
                "initial_prefill_tokens": sum(Ls) if repeated == 0 else 0,
                "replayed_history_tokens": repeated,
                "drafter_new_tokens": len(valid) * K,
                # Events are resolved when the window is flushed.
                "_events": [start, replay_end, proposal_start, proposal_end, end,
                            target_pair[0], target_pair[1],
                            sampler_pair[0], sampler_pair[1]],
            })

        return draft_full

    @torch.no_grad()
    def diagnostic_base_forward(self) -> dict:
        """Run ONE tiny base_model forward with synthetic inputs and return
        output shape + dtype. Used to verify the vLLM-native base forward
        pathway (paged attn + set_forward_context + base's attn_groups)
        mechanically works before committing to per-step integration.

        This bypasses slot_mapping/cache consistency — just checks the
        forward plumbing doesn't crash with our attn backend setup.
        """
        from vllm.forward_context import set_forward_context
        # Build the minimal viable per-layer attn metadata by picking up what
        # the parent set up for drafter and translating to base layers. For a
        # dispose-able dry-run, we borrow drafter's existing attn metadata
        # builder and just retarget layer names.
        if not self.base_attn_groups or not self.draft_attn_groups:
            return {"ok": False, "reason": "attn_groups not initialized"}

        # Create a minimal CommonAttentionMetadata — drafter's own helpers do
        # this via `build_for_drafting`. For diagnostic we just check that
        # base's metadata builder can instantiate something callable.
        try:
            _ = self.base_attn_groups[0].get_metadata_builder()
            return {
                "ok": True,
                "num_base_groups": len(self.base_attn_groups),
                "base_model_class": type(self.base_model).__name__,
                "vocab_size": getattr(self.base_model, "config", None).vocab_size
                              if hasattr(self.base_model, "config") else None,
            }
        except Exception as e:
            return {"ok": False, "reason": f"metadata builder: {e!r}"}

    @staticmethod
    def _vlmm_compute_mrope_positions_for_aug(
        aug_input_ids: list[int],
        image_grid_thw: torch.Tensor,
        config,
    ) -> torch.Tensor:
        """Compute Qwen-VL M-RoPE 3D positions for a full-context prompt
        with one image. Mirrors `_get_mrope_input_positions` static method
        from `qwen3_vl.py:2228` for the single-image case (no video, no
        multi-image, no EVS pruning).

        Returns: positions tensor shape (3, len(aug_input_ids)) on CPU.
        Caller must move to device.

        Algorithm:
          1. Find vision_start_token in aug_input_ids → text_len_before = position
          2. Image patch tokens: indices((1, llm_h, llm_w)).reshape(3, -1) where
             llm_h = h // spatial_merge_size, llm_w = w // spatial_merge_size
          3. Text after image: linear continuation from max(image_pos)+1
          4. All text positions broadcast across 3 axes (per Qwen-VL paper)
        """
        import numpy as np

        vision_start_id = config.vision_start_token_id
        spatial_merge = config.vision_config.spatial_merge_size

        ids = aug_input_ids if isinstance(aug_input_ids, list) else aug_input_ids.tolist()
        n = len(ids)

        # Locate the (single) vision_start token
        try:
            vs_idx = ids.index(vision_start_id)
        except ValueError:
            # No image — pure text aug prompt (shouldn't happen for VL drafter
            # but be safe). Broadcast linear positions.
            return torch.from_numpy(
                np.broadcast_to(np.arange(n, dtype=np.int64), (3, n)).copy()
            )

        # offset = position of FIRST image_pad token (right after vision_start)
        offset = vs_idx + 1

        # Image grid info
        t, h, w = image_grid_thw[0].tolist() if image_grid_thw.dim() == 2 \
            else image_grid_thw.tolist()
        assert t == 1, f"Smoke test only supports single-frame image, got t={t}"
        llm_h = h // spatial_merge
        llm_w = w // spatial_merge
        n_image_tokens = t * llm_h * llm_w

        # Build 3D position list piecewise
        pos_pieces = []

        # Section 1: text BEFORE image patches (positions 0..offset-1, includes
        # vision_start). 3 axes identical for text.
        text_pre_len = offset
        pos_pre = np.broadcast_to(
            np.arange(text_pre_len, dtype=np.int64), (3, text_pre_len)
        ).copy()
        pos_pieces.append(pos_pre)

        # Section 2: image patch tokens (axis 0 = constant, axes 1,2 = h,w spatial)
        st_idx = pos_pieces[-1].max() + 1
        grid_indices = np.indices((t, llm_h, llm_w)).reshape(3, -1).astype(np.int64)
        pos_pieces.append(grid_indices + st_idx)

        # Section 3: text AFTER image (vision_end + question + suffix). Linear
        # continuation from max(image_pos)+1, broadcast across 3 axes.
        after_image_offset = offset + n_image_tokens
        text_post_len = n - after_image_offset
        if text_post_len > 0:
            st_idx = pos_pieces[-1].max() + 1
            pos_post = np.broadcast_to(
                np.arange(text_post_len, dtype=np.int64), (3, text_post_len)
            ).copy() + st_idx
            pos_pieces.append(pos_post)

        positions = np.concatenate(pos_pieces, axis=1)  # (3, n)
        assert positions.shape == (3, n), \
            f"M-RoPE position shape mismatch: got {positions.shape}, want (3, {n})"
        return torch.from_numpy(positions)

    @torch.no_grad()
    def _vlmm_get_aug_mrope_positions(
        self, valid: list, all_input_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Get M-RoPE 3D positions for the concatenated full-context
        sequence. Also cache per-request mrope_max for
        K draft tokens).

        Returns: (3, sum_L) tensor on device, or None if no req has mm data.
        """
        import vllm.v1.core.kv_cache_utils as _kvu
        mm_cache = getattr(_kvu, "_specsteer_aug_mm_data", {})
        if not mm_cache:
            return None
        if self.runner is None:
            return None
        input_batch = self.runner.input_batch
        any_mm = False
        per_req_mm = []
        per_req_id = []
        for _, req_idx, _ in valid:
            req_id = input_batch.req_ids[req_idx] if req_idx < len(input_batch.req_ids) else None
            mm = mm_cache.get(req_id) if req_id else None
            per_req_mm.append(mm)
            per_req_id.append(req_id)
            if mm is not None:
                any_mm = True
        if not any_mm:
            return None

        Ls = [len(a) for _, _, a in valid]
        offsets = [0]
        for L in Ls:
            offsets.append(offsets[-1] + L)
        sum_L = offsets[-1]

        all_pos = torch.zeros((3, sum_L), dtype=torch.int64, device=self.device)
        for i, ((_, _, aug_ids), mm) in enumerate(zip(valid, per_req_mm)):
            if mm is None:
                Li = Ls[i]
                lin = torch.arange(Li, dtype=torch.int64, device=self.device)
                all_pos[0, offsets[i]:offsets[i+1]] = lin
                all_pos[1, offsets[i]:offsets[i+1]] = lin
                all_pos[2, offsets[i]:offsets[i+1]] = lin
                # Cache mrope_max for K-draft continuation
                if per_req_id[i] is not None:
                    mm_cache.setdefault(per_req_id[i], {})
                    mm_cache[per_req_id[i]]["mrope_max"] = int(Li - 1)
            else:
                pos_3d_cpu = self._vlmm_compute_mrope_positions_for_aug(
                    aug_ids if isinstance(aug_ids, list) else aug_ids.tolist(),
                    mm["image_grid_thw"],
                    self.model.config,
                )
                all_pos[:, offsets[i]:offsets[i+1]] = pos_3d_cpu.to(self.device)
                # Cache the drafter's maximum position so K draft
                # tokens get correct continuation positions (NOT verifier's positions)
                if per_req_id[i] is not None:
                    mm["mrope_max"] = int(pos_3d_cpu.max().item())
        return all_pos

    @torch.no_grad()
    def _vlmm_compute_aug_inputs_embeds(
        self, valid: list, all_input_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build input embeddings with image embeddings merged in.

        For each valid request that has cached pixel_values:
          1. Run drafter's vision tower → image_embeds
          2. Get text embeddings for ALL aug tokens (text + image_pad slots)
          3. Replace embeddings at image_pad slots with image_embeds
        Returns inputs_embeds tensor shape (sum_L, H), or None if no req has mm.

        valid: list[(slot, req_idx, aug_ids)] from forward sites
        all_input_ids: concatenated aug ids tensor (sum_L,)
        """
        import vllm.v1.core.kv_cache_utils as _kvu
        mm_cache = getattr(_kvu, "_specsteer_aug_mm_data", {})
        if not mm_cache:
            return None  # text-only path
        # Determine which valid reqs have mm data
        # We need req_id (string) but valid only has req_idx (int into input_batch)
        if self.runner is None:
            return None
        input_batch = self.runner.input_batch
        any_mm = False
        per_req_mm = []  # list of (mm_dict_or_none) aligned with valid
        for _, req_idx, _ in valid:
            req_id = input_batch.req_ids[req_idx] if req_idx < len(input_batch.req_ids) else None
            mm = mm_cache.get(req_id) if req_id else None
            per_req_mm.append(mm)
            if mm is not None:
                any_mm = True
        if not any_mm:
            return None  # no requests with mm
        # Get text embeddings for everyone first
        # Drafter model is self.model (Qwen3-VL); has embed_input_ids
        if not hasattr(self.model, "embed_input_ids"):
            logger.warning("AsymSpec: drafter has no embed_input_ids; using text only")
            return None
        # Compute multimodal embeddings (image features through vision tower)
        # Build a list of (slot_in_concat, mm_dict) for the concat'd buffer
        Ls = [len(a) for _, _, a in valid]
        offsets = [0]
        for L in Ls:
            offsets.append(offsets[-1] + L)
        # Find image_pad token id
        image_token_id = getattr(self.model.config, "image_token_id", 151655)
        # Run vision tower per request that has mm data, collect into a list
        mm_embeds_list = []
        is_mm = torch.zeros(all_input_ids.shape[0], dtype=torch.bool, device=self.device)
        for i, mm in enumerate(per_req_mm):
            if mm is None:
                continue
            # Cache image embeddings per request to avoid
            # re-running vision tower on every spec step (~50% SS slowdown).
            cached = mm.get("_image_embeds")
            if cached is not None:
                img_emb = cached
            else:
                pv = mm["pixel_values"].to(self.device, dtype=self.model.dtype) \
                    if hasattr(self.model, 'dtype') else mm["pixel_values"].to(self.device)
                grid = mm["image_grid_thw"].to(self.device)
                try:
                    visual = getattr(self.model, "visual", None) or \
                             getattr(self.model, "vision_tower", None)
                    if visual is None:
                        logger.warning("AsymSpec: no vision tower found")
                        return None
                    img_emb = visual(pv, grid_thw=grid)
                except Exception as e:
                    logger.warning(f"AsymSpec: vision tower failed: {e}")
                    return None
                mm["_image_embeds"] = img_emb  # cache for subsequent steps
            mm_embeds_list.append(img_emb)
            # Mark which positions in this req's slice are image_pad
            req_slice = all_input_ids[offsets[i]:offsets[i+1]]
            is_mm[offsets[i]:offsets[i+1]] = (req_slice == image_token_id)
        if not mm_embeds_list:
            return None
        # Build text embeddings via embed_input_ids; let it handle the merge
        try:
            inputs_embeds = self.model.embed_input_ids(
                all_input_ids,
                multimodal_embeddings=mm_embeds_list,
                is_multimodal=is_mm,
            )
        except Exception as e:
            logger.warning(f"AsymSpec: embed_input_ids failed: {e}")
            return None
        return inputs_embeds

    def set_inputs_first_pass(self, *args, **kwargs):
        """Write drafter-correct M-RoPE positions for K draft tokens.

        The inherited Triton kernel writes positions = `verifier_pos + j` to
        `self.positions` (1D). For asymmetric SS (drafter sees aug=image+text,
        verifier sees main=text-only OCR), drafter's positions live on a
        DIFFERENT axis than verifier's:
          - Verifier max position = len(main_prompt)         (text-only)
          - Drafter max position  = max(mrope[image+text])   (image-anchored)

        For the K draft tokens (text continuation post-image), drafter's
        positions should be `drafter_mrope_max + j` on all 3 axes.

        We use cached `mrope_max` from `_vlmm_get_aug_mrope_positions` to
        produce correct drafter positions, REPLACING the kernel's wrong
        verifier-derived values.
        """
        result = super().set_inputs_first_pass(*args, **kwargs)
        if not getattr(self, "uses_mrope", False):
            return result
        # Replace kernel-written positions with drafter-correct mrope.
        # Use per-request cached mrope_max if available.
        import vllm.v1.core.kv_cache_utils as _kvu
        mm_cache = getattr(_kvu, "_specsteer_aug_mm_data", {})
        if not mm_cache or self.runner is None:
            # No mm requests — kernel wrote 1D, broadcast to 3D (text-only path)
            n = self.positions.shape[0]
            pos_long = self.positions.to(torch.int64)
            self.mrope_positions[0, :n] = pos_long
            self.mrope_positions[1, :n] = pos_long
            self.mrope_positions[2, :n] = pos_long
            return result
        # For each output slot, lookup which request it belongs to and apply
        # drafter-correct positions. We don't have per-slot req_id directly
        # here (kernel writes by output_start), but for BS=1 all
        # slots map to req 0.
        input_batch = self.runner.input_batch
        # BS=1 smoke shortcut: lookup the single request's mrope_max
        if input_batch.num_reqs == 1:
            req_id = input_batch.req_ids[0]
            mm = mm_cache.get(req_id, {})
            mrope_max = mm.get("mrope_max", None)
            n = self.positions.shape[0]
            pos_long = self.positions.to(torch.int64)
            if mrope_max is not None:
                # Kernel wrote `verifier_start + j`. Replace with
                # `drafter_mrope_max + 1 + j_relative_to_first_draft`.
                # j_relative_to_first_draft = pos_long - pos_long.min() (for
                # the slots the kernel actually wrote to)
                j_rel = pos_long - pos_long.min().clamp(min=0)
                drafter_pos = (mrope_max + 1) + j_rel
                self.mrope_positions[0, :n] = drafter_pos
                self.mrope_positions[1, :n] = drafter_pos
                self.mrope_positions[2, :n] = drafter_pos
            else:
                self.mrope_positions[0, :n] = pos_long
                self.mrope_positions[1, :n] = pos_long
                self.mrope_positions[2, :n] = pos_long
        else:
            # BS>1 (TODO: per-slot lookup). For now broadcast text-only style.
            n = self.positions.shape[0]
            pos_long = self.positions.to(torch.int64)
            self.mrope_positions[0, :n] = pos_long
            self.mrope_positions[1, :n] = pos_long
            self.mrope_positions[2, :n] = pos_long
        return result

    @torch.no_grad()
    def forward_base(
        self,
        input_ids: torch.Tensor,         # [N] int64 - tokens to forward
        positions: torch.Tensor,         # [N] int64 - absolute positions for each
        common_attn_metadata,            # reuse drafter's, shares block_table
        slot_mapping: torch.Tensor,      # [N] int64 - cache slots to write
    ) -> torch.Tensor:
        """Forward SLM_base in parallel, returning logits [N, V].

        Uses vLLM-native paged attention + CUDA graphs via the layers
        registered under the 'specsteer_base' tag. Safe to share block_table
        and slot_mapping with drafter because base's layers have distinct
        names ('specsteer_base.layers.N.*') — reshape_and_cache writes into
        base's own per-layer cache tensor, no overlap with drafter's.
        """
        from vllm.forward_context import set_forward_context

        num_input_tokens = input_ids.shape[0]

        # Build per-layer attn metadata for base
        per_layer_attn_metadata = {}
        for attn_group in self.base_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=0,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        model_kwargs = {
            "input_ids": input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
            "positions": positions,
            "inputs_embeds": None,
        }
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            slot_mapping=slot_mapping,
        ):
            ret = self.base_model(**model_kwargs)
            hidden_states = ret[0] if isinstance(ret, tuple) else ret

        logits = self.base_model.compute_logits(hidden_states)
        return logits

    def _reset_draft_logits(self) -> None:
        self._draft_logits_per_pos = []
        self._base_logits_per_pos = []

    def _record_pathb_fallback(self, step: int, reason: str) -> None:
        """Hard-fail on PathB PV failure.

        Silently dropping back to base = drafter would zero out the
        context-gain delta and collapse SpecSteer into LLM-only gamma-rule
        rejection sampling. Result JSONs would look fine but the algorithm
        has changed — fail loudly instead.
        """
        raise RuntimeError(
            f"SpecSteer PathB base_logits unavailable at step {step}: "
            f"{reason}. Setting base = drafter would silently change the "
            f"algorithm (delta == 0 -> fused argmax = LLM argmax)."
        )

    @property
    def last_draft_logits(self) -> torch.Tensor | None:
        """Stacked [num_pos, batch_size, vocab_size] or None if no drafts
        have been produced yet."""
        if not self._draft_logits_per_pos:
            return None
        return torch.stack(self._draft_logits_per_pos, dim=0)

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retain drafter logits for fusion; Path B supplies base logits."""
        logits = self.model.compute_logits(hidden_states)
        self._draft_logits_per_pos.append(logits.detach())
        self._base_logits_per_pos.append(logits.detach())  # placeholder
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def _base_mirror_forward(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata,
    ) -> torch.Tensor | None:
        """Run SLM_base's parallel forward on the SAME inputs drafter is about
        to see (target_token_ids = committed tokens scheduled this step),
        populating base's KV cache identically and returning hidden_states.

        Per-step, mirror forward advances base cache by num_scheduled_tokens,
        matching drafter's first parallel forward. Base lags drafter by K-1
        tokens (autoregressive drafts) after this; to get p_base at drafted
        positions we run an extra forward — see _base_extra_forward_on_drafts.

        Returns None if attn groups aren't set up.
        """
        from vllm.forward_context import set_forward_context
        if not self.base_attn_groups:
            return None

        per_layer_attn_metadata = {}
        for attn_group in self.base_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common_attn_metadata, draft_index=0,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        num_tokens = target_token_ids.shape[0]
        # CRITICAL: parent's _get_slot_mapping returns a dict keyed by
        # DRAFTER's layer names — base's layers wouldn't find their
        # slot_mapping in set_forward_context and KV writes go to
        # garbage/nowhere. Build a slot_mapping dict keyed by BASE's layer
        # names using the same slot tensor (shared block_table at gid 0).
        drafter_slot_dict = self._get_slot_mapping(
            num_tokens, common_attn_metadata.slot_mapping,
        )
        # Extract the underlying slot tensor (all drafter entries share it).
        slot_tensor = next(iter(drafter_slot_dict.values()))
        slot_mapping = {
            layer_name: slot_tensor for layer_name in self._base_attn_layer_names
        }

        model_kwargs = {
            "input_ids": target_token_ids,
            "positions": target_positions,
            "inputs_embeds": None,
        }

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            slot_mapping=slot_mapping,
        ):
            ret = self.base_model(**model_kwargs)
            hidden_states = ret[0] if isinstance(ret, tuple) else ret

        # Cache hidden[-1] — its logit predicts the FIRST drafted token
        # (position L+0 = drafts[0]) under base's main ctx.
        self._last_mirror_tail_hidden = hidden_states[-1:].detach()
        return hidden_states

    @torch.no_grad()
    def _maybe_substitute_aug_tokens(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        common_attn_metadata=None,
    ) -> torch.Tensor:
        """Gate A: substitute main-ctx tokens with aug-ctx tokens for drafter
        at prefill positions (per-request for BS>=1).

        For BS>1, target_token_ids is concatenated across requests. Use
        common_attn_metadata.query_start_loc to locate each request's
        segment: positions [qsl[i], qsl[i+1]) in the flat tensor.

        Only substitutes for requests whose SamplingParams has
        `specsteer_aug_prompt_ids` set AND whose aug length matches its
        prompt length. Other requests pass through unchanged.
        """
        if self.runner is None:
            return target_token_ids
        input_batch = getattr(self.runner, "input_batch", None)
        if input_batch is None:
            return target_token_ids
        B = input_batch.num_reqs
        if B < 1:
            return target_token_ids
        req_ids = getattr(input_batch, "req_ids", None)
        if not req_ids:
            return target_token_ids

        # EARLY EXIT: collect aug per-req WITHOUT GPU→CPU sync. Only proceed
        # to positions.cpu() if at least one req has aug and is in prefill.
        # (Previously this method did positions.cpu() unconditionally EVERY
        # decode step → 14ms/step sync cost.)
        has_any_aug = False
        aug_per_req: list[tuple] = []  # (i, aug, num_prompt_tokens_i)
        for i in range(B):
            req_id_i = req_ids[i]
            req = self.runner.requests.get(req_id_i)
            if req is None:
                continue
            sp = getattr(req, "sampling_params", None)
            aug = None
            if sp is not None and sp.extra_args is not None:
                aug = sp.extra_args.get("specsteer_aug_prompt_ids")
            if not aug:
                continue
            num_prompt_tokens_i = req.num_prompt_tokens
            if len(aug) != num_prompt_tokens_i:
                continue
            # Check if this req's num_computed_tokens is already past prefill.
            # If all reqs are past prefill, no substitution needed.
            num_computed_i = int(self.runner.input_batch.num_tokens_no_spec[i])
            if num_computed_i >= num_prompt_tokens_i:
                # decode step for this req → no substitution needed
                continue
            aug_per_req.append((i, aug, num_prompt_tokens_i))
            has_any_aug = True

        if not has_any_aug:
            # All reqs are past prefill or no aug set — skip expensive path
            if not getattr(self, "_warned_no_aug", False):
                logger.info(
                    "SpecSteer: no req needs aug substitution (all past prefill "
                    "or no aug set)",
                )
                self._warned_no_aug = True
            return target_token_ids

        # Only now pay the sync cost to figure out positions per-req
        N = target_token_ids.shape[0]
        if B == 1:
            qsl = [0, N]
        elif common_attn_metadata is not None and hasattr(common_attn_metadata, 'query_start_loc_cpu'):
            qsl = common_attn_metadata.query_start_loc_cpu.tolist()
        else:
            return target_token_ids

        positions_cpu = target_positions.detach().cpu()
        result = None

        for i, aug, num_prompt_tokens_i in aug_per_req:
            req_id_i = req_ids[i]
            if not getattr(self, "_logged_aug", False):
                logger.info(
                    "SpecSteer: Gate A active — req[%d]=%s aug_len=%d",
                    i, req_id_i, len(aug),
                )
                self._logged_aug = True

            lo, hi = qsl[i], qsl[i + 1]
            seg_positions = positions_cpu[lo:hi]
            mask = seg_positions < num_prompt_tokens_i
            if not mask.any():
                continue  # this req has no prefill positions in current batch
            if result is None:
                result = target_token_ids.clone()
            aug_tensor = torch.tensor(aug, dtype=target_token_ids.dtype,
                                       device=target_token_ids.device)
            idx_in_seg = mask.nonzero(as_tuple=True)[0]
            for j in idx_in_seg.tolist():
                pos = int(seg_positions[j])
                result[lo + j] = aug_tensor[pos]

        if result is None:
            if not getattr(self, "_warned_no_aug", False):
                logger.info(
                    "SpecSteer: no req has specsteer_aug_prompt_ids — "
                    "skipping full-context substitution",
                )
                self._warned_no_aug = True
            return target_token_ids
        return result

    @override
    def propose(self, num_speculative_tokens, *args, **kwargs):
        self.num_speculative_tokens = num_speculative_tokens
        # Bind the 0.28 signature once, keeping the legacy context-swap code
        # keyword-based so it cannot confuse K with target_token_ids.
        import inspect
        bound = inspect.signature(super().propose).bind(
            num_speculative_tokens, *args, **kwargs)
        kwargs = dict(bound.arguments)
        args = ()
        self._sync_request_caches()

        self._reset_draft_logits()
        self._last_base_logits = None

        # Pull all the relevant args parent expects positionally:
        #   target_token_ids, target_positions, target_hidden_states,
        #   next_token_ids, token_indices_to_sample, common_attn_metadata, ...
        target_token_ids = kwargs.get("target_token_ids")
        target_positions = kwargs.get("target_positions")
        target_hidden_states = kwargs.get("target_hidden_states")
        next_token_ids = kwargs.get("next_token_ids")
        token_indices_to_sample = kwargs.get("token_indices_to_sample")
        common_attn_metadata = kwargs.get("common_attn_metadata")
        num_rejected_tokens_gpu = kwargs.get("num_rejected_tokens_gpu")
        if args:
            if target_token_ids is None: target_token_ids = args[0]
            if target_positions is None: target_positions = args[1]
            if target_hidden_states is None and len(args) > 2: target_hidden_states = args[2]
            if next_token_ids is None and len(args) > 3: next_token_ids = args[3]
            if token_indices_to_sample is None and len(args) > 4: token_indices_to_sample = args[4]
            if common_attn_metadata is None and len(args) > 5: common_attn_metadata = args[5]

        # Qwen3.5's GDN state is recurrent, not paged token KV.  The native
        # runner lifecycle deliberately owns verifier state only; letting the
        # two auxiliary 4B views enter ``super().propose`` leaves their state
        # after rejected drafts live into the next iteration.  This must run
        # for *every* hybrid request, including equal-length full/main views.
        if self._hybrid_specsteer and num_speculative_tokens > 0:
            if self.runner is None or next_token_ids is None:
                raise RuntimeError(
                    "Hybrid SpecSteer requires a runner and verifier next_token_ids"
                )
            input_batch = self.runner.input_batch
            prompt_counts = getattr(input_batch, "num_prompt_tokens_cpu_tensor", None)
            if prompt_counts is None:
                raise RuntimeError(
                    "Hybrid SpecSteer requires prompt-token bookkeeping"
                )
            hybrid_items: list[tuple[int, list[int]]] = []
            for req_idx, req_id in enumerate(input_batch.req_ids[:input_batch.num_reqs]):
                req = self.runner.requests.get(req_id)
                full_ids = None
                if req is not None and req.sampling_params is not None:
                    extra = req.sampling_params.extra_args
                    if extra is not None:
                        full_ids = extra.get("specsteer_aug_prompt_ids")
                # Preserve direct-engine compatibility: without the serving
                # extension, the compressed prompt is also the full view.
                if not full_ids:
                    main_prompt_len = int(prompt_counts[req_idx])
                    full_ids = input_batch.token_ids_cpu[
                        req_idx, :main_prompt_len
                    ].tolist()
                hybrid_items.append((req_idx, list(full_ids)))

            draft_token_ids = self._merged_aug_prefill_and_kdecode(
                hybrid_items,
                K=num_speculative_tokens,
                next_token_ids=next_token_ids,
            )
            if draft_token_ids is None:
                raise RuntimeError("Hybrid SpecSteer drafter rebuild failed")

            B, K = draft_token_ids.shape
            if self._strict_target_mode:
                # The verifier owns every committed token in strict mode. The
                # full-context 4B only proposes; no compressed 4B exists and
                # no delta/base forward may run.
                self._base_logits_per_pos = []
                self._prof_report_if_due()
                return draft_token_ids

            pv_logits = self._base_parallel_verify(
                draft_token_ids.reshape(-1), next_token_ids=next_token_ids,
            )
            if (pv_logits is None or pv_logits.dim() != 3
                    or pv_logits.shape[0] != B or pv_logits.shape[1] < K):
                raise RuntimeError(
                    "Hybrid SpecSteer base rebuild returned invalid logits"
                )
            self._base_logits_per_pos = [
                pv_logits[:, k, :].detach() for k in range(K)
            ]
            step = getattr(self, "_pv_step_counter", 0) + 1
            self._pv_step_counter = step
            if step <= 3 or step % 20 == 1:
                aug_l0 = self._draft_logits_per_pos[0][0].view(-1)
                base_l0 = pv_logits[0, 0]
                logger.info(
                    "Hybrid SpecSteer canonical step %d: B=%d K=%d aug=%d base=%d "
                    "|base-aug|=%.3f",
                    step, B, K, int(aug_l0.argmax().item()),
                    int(base_l0.argmax().item()),
                    float((aug_l0.float() - base_l0.float()).abs().max().item()),
                )
            return draft_token_ids

        # Stash token_indices_to_sample so _greedy_sample can correctly align
        # base_hidden slicing with drafter's sampled positions. Default (None
        # → last-of-each-request) is covered by the is_first_call fallback
        # to base_hidden[-n_sample:]. Non-default values (rare, but cause
        # bugs when base_hidden[-n] ≠ base_hidden[indices]) need this stash.
        # Derive the effective indices: if None, parent uses
        # cad.query_start_loc[1:] - 1 (BS=1 → [num_tokens - 1]).
        if token_indices_to_sample is not None:
            self._stashed_token_indices_to_sample = token_indices_to_sample
        elif common_attn_metadata is not None:
            self._stashed_token_indices_to_sample = (
                common_attn_metadata.query_start_loc[1:] - 1
            )
        else:
            self._stashed_token_indices_to_sample = None

        # Asymmetric-context setup: drafter in gid=1 with aug-space when
        # aug_ids longer than main. Swap:
        # (a) block_table_tensor + slot_mapping → drafter cache group;
        # (b) target_token_ids/positions/hidden_states → full-context length,
        #     ONLY on first propose call (prefill step). For decode steps,
        #     positions shift by offset but other buffers keep main shape.
        # Profile context swap and bonus correction.
        self._prof_swap_pair = self._prof_start("A_context_swap")
        # Per-req aug data: (req_idx, aug_ids, aug_offset, L_main, L_aug).
        # aug_offset = L_aug - L_main; positive → aug ctx longer than main.
        # Empty means the symmetric draft-model path is used.
        per_req_aug: list[tuple[int, list[int], int, int, int]] = []
        if (self.drafter_kv_cache_gid is not None
                and self.drafter_kv_cache_gid >= 0
                and common_attn_metadata is not None
                and self.runner is not None):
            input_batch = self.runner.input_batch
            req_ids_list = getattr(input_batch, "req_ids", None)
            if req_ids_list:
                for ri, rid in enumerate(req_ids_list[:input_batch.num_reqs]):
                    req = self.runner.requests.get(rid)
                    if req is None or req.sampling_params is None \
                            or req.sampling_params.extra_args is None:
                        continue
                    aug_i = req.sampling_params.extra_args.get(
                        "specsteer_aug_prompt_ids")
                    if not aug_i:
                        continue
                    L_main_i = req.num_prompt_tokens
                    L_aug_i = len(aug_i)
                    off_i = L_aug_i - L_main_i
                    # Allow either sign for multimodal full/main length offsets.
                    if off_i != 0:
                        per_req_aug.append((ri, aug_i, off_i, L_main_i, L_aug_i))

        if (self.drafter_kv_cache_gid is not None
                and self.drafter_kv_cache_gid >= 0
                and common_attn_metadata is not None
                and self.runner is not None):
            input_batch = self.runner.input_batch
            if (hasattr(input_batch, "block_table")
                    and len(input_batch.block_table.block_tables) > self.drafter_kv_cache_gid):
                from copy import copy as _shallow_copy
                drafter_bt = input_batch.block_table[self.drafter_kv_cache_gid]
                num_reqs = common_attn_metadata.num_reqs

                # Detect PREFILL step using CPU-side cam fields (no GPU→CPU sync).
                # Decode step has K+1 tokens per req via spec-decode parallel verify,
                # so max_query_len == K+1 and num_actual_tokens == num_reqs × (K+1).
                # Prefill has max_query_len == L_main (much larger).
                _cam_max_query_len = getattr(common_attn_metadata, "max_query_len", 0)
                _expected_decode_max_q = (self.num_speculative_tokens or 0) + 1
                is_prefill = _cam_max_query_len > _expected_decode_max_q

                # PREFILL aug-swap fires when batch is homogeneous (all N reqs
                # have aug_ids with positive offset). N=1 with single aug req
                # is the trivial case.
                # Accept off!=0 so multimodal AsymSpec
                # (where aug=image+short_query may be SHORTER than main=caption)
                # also triggers the context swap. Without this, the drafter falls back
                # to running on MAIN text and never sees the image.
                _homogeneous_aug_batch = (
                    len(per_req_aug) == num_reqs and num_reqs > 0
                    and all(off != 0 for _, _, off, _, _ in per_req_aug)
                )
                if _homogeneous_aug_batch and is_prefill:
                    # ===== Unified PREFILL swap (single code path, N≥1) =====
                    L_augs = [la for _, _, _, _, la in per_req_aug]
                    sum_L = sum(L_augs)
                    max_L = max(L_augs)
                    offsets = [0]
                    for la in L_augs:
                        offsets.append(offsets[-1] + la)
                    device = target_token_ids.device

                    # input tokens + positions: per-req tensors, concat. At N=1
                    # these lists have 1 entry; torch.cat of [t] returns t-shaped
                    # contiguous tensor with identical values → same kernels.
                    new_target_token_ids = torch.cat([
                        torch.tensor(aug_i, dtype=target_token_ids.dtype,
                                     device=device)
                        for _, aug_i, _, _, _ in per_req_aug
                    ], dim=0)
                    new_target_positions = torch.cat([
                        torch.arange(0, la, dtype=target_positions.dtype,
                                     device=device)
                        for la in L_augs
                    ], dim=0)

                    # Hidden states pad/truncate to sum_L. At N=1 sum_L==L_aug,
                    # pad if shorter and truncate if longer.
                    hs_shape = target_hidden_states.shape
                    if hs_shape[0] < sum_L:
                        pad = torch.zeros(
                            (sum_L - hs_shape[0],) + tuple(hs_shape[1:]),
                            dtype=target_hidden_states.dtype, device=device,
                        )
                        new_hidden_states = torch.cat(
                            [target_hidden_states, pad], dim=0)
                    else:
                        new_hidden_states = target_hidden_states[:sum_L]

                    # CAD construction: per-req lists. At N=1 these lists are
                    # length 1 produces a shape-(1,) tensor.
                    new_cam = _shallow_copy(common_attn_metadata)
                    new_cam.num_actual_tokens = sum_L
                    new_cam.max_query_len = max_L
                    new_cam.max_seq_len = max_L
                    new_cam.query_start_loc = torch.tensor(
                        offsets, dtype=torch.int32, device=device)
                    new_cam.query_start_loc_cpu = torch.tensor(
                        offsets, dtype=torch.int32)
                    new_cam.seq_lens = torch.tensor(
                        L_augs, dtype=torch.int32, device=device)
                    new_cam._seq_lens_cpu = torch.tensor(
                        L_augs, dtype=torch.int32)
                    new_cam._num_computed_tokens_cpu = torch.tensor(
                        [0] * num_reqs, dtype=torch.int32)
                    new_cam.block_table_tensor = drafter_bt.get_device_tensor(num_reqs)
                    block_size = drafter_bt.block_size

                    # slot_mapping: unified per-req loop. At N=1 the loop runs
                    # once with j=0, ri=0, la=L_aug, producing same arithmetic
                    # as the single-request block addressing expression.
                    slot_pieces = []
                    for j, (ri, _, _, _, la) in enumerate(per_req_aug):
                        row_j = new_cam.block_table_tensor[ri]
                        local_pos = torch.arange(0, la, dtype=torch.int64,
                                                 device=device)
                        pos_int32_j = local_pos.to(torch.int32)
                        block_ids_j = row_j[pos_int32_j // block_size]
                        slot_j = (block_ids_j.to(torch.int64) * block_size
                                  + (local_pos % block_size))
                        slot_pieces.append(slot_j)
                    new_cam.slot_mapping = torch.cat(slot_pieces, dim=0)
                    new_cam.is_prefilling = torch.tensor(
                        [True] * num_reqs, dtype=torch.bool, device=device)
                    # FA3 pre-computed scheduler_metadata is for L_main shape;
                    # after swap to L_aug it's stale and kernel asserts
                    # `metadata_size` mismatch. None → FA3 computes per-call.
                    # (removed: scheduler_metadata=None — not a cam field)

                    # Replace in kwargs + args
                    kwargs["target_token_ids"] = new_target_token_ids
                    kwargs["target_positions"] = new_target_positions
                    kwargs["target_hidden_states"] = new_hidden_states
                    kwargs["common_attn_metadata"] = new_cam
                    if args:
                        if len(args) > 0: args = (new_target_token_ids,) + args[1:]
                        if len(args) > 1: args = args[:1] + (new_target_positions,) + args[2:]
                        if len(args) > 2: args = args[:2] + (new_hidden_states,) + args[3:]
                        if len(args) > 5: args = args[:5] + (new_cam,) + args[6:]
                    # Per-req sample-last indices: position offsets[i+1]-1.
                    # At N=1 with offsets=[0, L_aug], this is [L_aug - 1].
                    self._stashed_token_indices_to_sample = torch.tensor(
                        [offsets[i + 1] - 1 for i in range(num_reqs)],
                        dtype=torch.int32, device=device,
                    )

                    # Forward-merged path: aug-prefill (incremental) + K
                    # decodes in one fused call. Replaces separate
                    # bonus + F2 prefill_(L+1) + (K-1) decodes] sequence.
                    # Only valid when spec-decode driver supplied LLM bonus
                    # in next_token_ids; otherwise this is the initial step
                    # and we fall through to super().propose() with the
                    # swapped (aug-shaped) cam.
                    if next_token_ids is not None:
                        items = [(ri, aug_i)
                                 for ri, aug_i, _, _, _ in per_req_aug]
                        merged_drafts = self._merged_aug_prefill_and_kdecode(
                            items,
                            K=self.num_speculative_tokens,
                            next_token_ids=next_token_ids,
                        )
                        if merged_drafts is None:
                            raise RuntimeError(
                                "SpecSteer merged aug-prefill+Kdec returned "
                                "None — single canonical path requires it to "
                                "succeed. Check drafter init / aug_ids."
                            )
                        self._prof_end(self._prof_swap_pair)
                        self._prof_swap_pair = None
                        return merged_drafts
                else:
                    # ===== Unified decode-shift OR aug==main path (N≥1) =====
                    # Hot path: runs every decode iteration. Batched ops only
                    # (no Python loops, no .item() syncs); N=1 degenerates to
                    # single-row tensor operations at BS=1.
                    new_cam = _shallow_copy(common_attn_metadata)
                    new_cam.block_table_tensor = drafter_bt.get_device_tensor(num_reqs)
                    block_size = drafter_bt.block_size
                    device = target_positions.device

                    per_req_offsets = [0] * num_reqs
                    for ri, _, off, _, _ in per_req_aug:
                        per_req_offsets[ri] = off
                    max_off = max(per_req_offsets)
                    any_offset = max_off > 0

                    # At decode step (is_prefill==False is the gate above),
                    # every req has cam.max_query_len tokens (K+1 for spec
                    # decode parallel verify).  Ordinary models expose
                    # positions as (N*tpr,), while Qwen3.5 text uses M-RoPE
                    # positions shaped (3, N*tpr).  Shift every M-RoPE axis,
                    # but derive cache slots from its first (text) axis.
                    tpr = common_attn_metadata.max_query_len
                    is_mrope = target_positions.ndim == 2
                    if is_mrope:
                        pos_by_axis = target_positions.view(
                            target_positions.shape[0], num_reqs, tpr)
                    else:
                        pos_by_axis = target_positions.view(num_reqs, tpr)

                    if any_offset:
                        seq_dt = common_attn_metadata.seq_lens.dtype
                        offsets_cpu = torch.tensor(per_req_offsets, dtype=seq_dt)
                        offsets_gpu = offsets_cpu.to(device, non_blocking=True)
                        # (N, tpr) + (N, 1) → broadcast → flatten to (N*tpr,)
                        # At N=1 reshapes are free views; final tensor is
                        # equivalent to adding one scalar offset at BS=1.
                        if is_mrope:
                            shifted_positions = (
                                pos_by_axis
                                + offsets_gpu.to(target_positions.dtype).view(
                                    1, num_reqs, 1)
                            ).reshape(target_positions.shape[0], -1)
                        else:
                            shifted_positions = (
                                pos_by_axis
                                + offsets_gpu.to(target_positions.dtype).view(
                                    num_reqs, 1)
                            ).reshape(-1)
                        new_cam.seq_lens = common_attn_metadata.seq_lens + offsets_gpu
                        if common_attn_metadata._seq_lens_cpu is not None:
                            offsets_cpu_seq = offsets_cpu.to(
                                common_attn_metadata._seq_lens_cpu.dtype)
                            new_cam._seq_lens_cpu = (
                                common_attn_metadata._seq_lens_cpu + offsets_cpu_seq)
                            # CPU max — no GPU sync.
                            new_cam.max_seq_len = int(new_cam._seq_lens_cpu.max())
                        else:
                            new_cam.max_seq_len = (
                                common_attn_metadata.max_seq_len + max_off)
                    else:
                        shifted_positions = target_positions

                    pos_int32 = (shifted_positions[0] if is_mrope
                                 else shifted_positions).to(torch.int32)

                    # slot_mapping: single batched gather, no Python loop.
                    # block_table_tensor shape (N, max_blocks); pos_2d (N, tpr).
                    # gather along dim=1 produces (N, tpr) of block_ids; slot =
                    # block_id * block_size + pos % block_size, all batched.
                    # At N=1 this is one (1, max_blocks).gather((1, tpr)) call,
                    # same arithmetic as `row[pos // block_size]` indexing.
                    pos_2d = pos_int32.view(num_reqs, tpr)
                    block_idx = (pos_2d // block_size).to(torch.int64)
                    block_ids = new_cam.block_table_tensor.gather(1, block_idx)
                    slot_2d = (block_ids.to(torch.int64) * block_size
                               + pos_2d.to(torch.int64).remainder(block_size))
                    new_cam.slot_mapping = slot_2d.reshape(-1)
                    kwargs["common_attn_metadata"] = new_cam
                    if len(args) > 5:
                        args = args[:5] + (new_cam,) + args[6:]
                    if any_offset:
                        kwargs["target_positions"] = shifted_positions
                        if len(args) > 1:
                            args = args[:1] + (shifted_positions,) + args[2:]

        # Gate A: substitute aug tokens for drafter at prefill positions.
        # ALSO stash the pre-substitution main version of the drafter's
        # first-forward input so dual_forward can swap it back for base
        # (base must see main ctx while drafter sees aug).
        aug_target = self._maybe_substitute_aug_tokens(
            target_token_ids, target_positions,
            common_attn_metadata=common_attn_metadata,
        )
        if aug_target is not target_token_ids:
            kwargs["target_token_ids"] = aug_target
            if args:
                args = (aug_target,) + args[1:]
            # Stash main version of target_token_ids for dual_forward swap.
            # dual_forward will compute main-version input_ids by reversing
            # the aug substitution on the shifted buffer.
            # Gate A fix (scheme 1): replace next_token_ids[i] (LLM's main-ctx
            # bonus at position L_i) with aug SLM's own prediction at L_i
            # for each request that has specsteer_aug_prompt_ids set and
            # hasn't had its bonus replaced yet.
            if not hasattr(self, "_aug_bonus_computed"):
                self._aug_bonus_computed: set[str] = set()
            input_batch = self.runner.input_batch
            req_ids = getattr(input_batch, "req_ids", None)
            if req_ids and next_token_ids is not None:
                # Streaming-session chunk-boundary detection: peek at each
                # rid's current aug_len; if it grew vs last seen, invalidate
                # _aug_bonus_computed + _base_prefilled for that rid so the
                # bonus is recomputed and base re-prefilled to cover the
                # chunk's new main tokens. Single-chunk requests see no
                # growth → no invalidation in the non-streaming path.
                aug_ids_by_idx: dict[int, list[int]] = {}
                for i, req_id_i in enumerate(req_ids[:input_batch.num_reqs]):
                    req = self.runner.requests.get(req_id_i)
                    if req is None or req.sampling_params is None \
                            or req.sampling_params.extra_args is None:
                        continue
                    a = req.sampling_params.extra_args.get(
                        "specsteer_aug_prompt_ids")
                    if a:
                        aug_ids_by_idx[i] = a
                self._maybe_invalidate_chunk_caches(
                    list(req_ids[:input_batch.num_reqs]), aug_ids_by_idx,
                )

                next_token_ids_mut = None
                # COLLECT pending requests needing aug-bonus replacement, then
                # invoke a SINGLE batched _compute_aug_first_bonus call. At
                # num_reqs==1 with one pending req, the batched call degenerates
                # to a single-request shape (see _compute_aug_first_bonus).
                pending: list[tuple[int, str, list[int]]] = []  # (slot_i, req_id, aug_ids)
                for i, req_id_i in enumerate(req_ids[:input_batch.num_reqs]):
                    if req_id_i in self._aug_bonus_computed:
                        continue
                    aug_ids_i = aug_ids_by_idx.get(i)
                    if aug_ids_i:
                        pending.append((i, req_id_i, aug_ids_i))
                    else:
                        # Mark as computed even if no aug_ids — don't retry next step.
                        self._aug_bonus_computed.add(req_id_i)
                if pending:
                    # Single batched drafter forward over all pending reqs.
                    bonuses = self._compute_aug_first_bonus(
                        [(i, aug) for i, _, aug in pending],
                    )
                    for (i, req_id_i, _), bonus in zip(pending, bonuses):
                        if bonus is not None:
                            if next_token_ids_mut is None:
                                next_token_ids_mut = next_token_ids.clone()
                            old_bonus = int(next_token_ids_mut.view(-1)[i].item())
                            next_token_ids_mut.view(-1)[i] = bonus
                            logger.info(
                                "SpecSteer Gate A bonus fix: req[%d]=%s "
                                "next_token_ids LLM=%d → aug=%d",
                                i, req_id_i, old_bonus, bonus,
                            )
                        self._aug_bonus_computed.add(req_id_i)
                if next_token_ids_mut is not None:
                    next_token_ids = next_token_ids_mut
                    kwargs["next_token_ids"] = next_token_ids
                    if len(args) > 3:
                        args = args[:3] + (next_token_ids,) + args[4:]
        # End context-swap/bonus profile segment.
        if self._profile_enabled and hasattr(self, "_prof_swap_pair"):
            self._prof_end(self._prof_swap_pair)
            self._prof_swap_pair = None
        # Inject multimodal embeddings so Eagle's drafter prefill
        # uses image embeddings (line 441 of eagle.py). Without this, drafter
        # treats image_pad tokens as ordinary text → wrong logits → KV pollution.
        # Only fires for prefill step with mm reqs; decode steps use cached KV.
        # Eagle defaults supports_mm_inputs=False for our drafter; we need it
        # True so eagle.py:438 branch fires and uses mm_embed_inputs.
        if per_req_aug and is_prefill:
            try:
                import vllm.v1.core.kv_cache_utils as _kvu
                mm_cache = getattr(_kvu, "_specsteer_aug_mm_data", {})
                image_token_id = getattr(
                    self.model.config, "image_token_id", 151655)
                # Concat input_ids matching the swapped target_token_ids order.
                # per_req_aug entries are (ri, aug_ids, off, L_main, L_aug)
                concat_ids = []
                for _, aug_i, _, _, _ in per_req_aug:
                    concat_ids.extend(aug_i)
                input_ids_t = torch.tensor(
                    concat_ids, dtype=torch.long, device=self.device)
                is_mm_mask = (input_ids_t == image_token_id)
                # Build per-req mm_embeds list via vision tower (with cache).
                mm_embeds_list_inj: list[torch.Tensor] = []
                visual = (getattr(self.model, "visual", None)
                          or getattr(self.model, "vision_tower", None))
                req_ids_list_inj = getattr(
                    self.runner.input_batch, "req_ids", None)
                for ri, _, _, _, _ in per_req_aug:
                    rid = req_ids_list_inj[ri]
                    mm = mm_cache.get(rid)
                    if mm is None:
                        mm_embeds_list_inj = []
                        break
                    cached = mm.get("_image_embeds")
                    if cached is None and visual is not None:
                        pv = mm["pixel_values"].to(
                            self.device,
                            dtype=getattr(self.model, "dtype", torch.bfloat16))
                        grid = mm["image_grid_thw"].to(self.device)
                        cached = visual(pv, grid_thw=grid)
                        mm["_image_embeds"] = cached
                    if cached is None:
                        mm_embeds_list_inj = []
                        break
                    mm_embeds_list_inj.append(cached)
                if mm_embeds_list_inj and int(is_mm_mask.sum()) > 0:
                    kwargs["mm_embed_inputs"] = (
                        mm_embeds_list_inj, is_mm_mask)
                    # Force Eagle into mm-aware embed path
                    self._saved_supports_mm = getattr(
                        self, "supports_mm_inputs", False)
                    self.supports_mm_inputs = True
                    logger.info(
                        "AsymSpec: injected mm_embed_inputs n_imgs=%d "
                        "is_mm_count=%d input_len=%d (forced supports_mm=True)",
                        len(mm_embeds_list_inj),
                        int(is_mm_mask.sum()),
                        input_ids_t.shape[0])
            except Exception as e:
                logger.warning(
                    "AsymSpec: mm_embed_inputs construction failed: %r", e)
        # PROFILE: super().propose() drafter K forwards (segment B)
        _prof_B = self._prof_start("B_super_propose")
        draft_token_ids = super().propose(*args, **kwargs)
        self._prof_end(_prof_B)
        # Restore supports_mm_inputs after Eagle prefill.
        if hasattr(self, "_saved_supports_mm"):
            self.supports_mm_inputs = self._saved_supports_mm
            del self._saved_supports_mm

        # PATH-B MODE: populate _base_logits_per_pos from PV (dual's base
        # was skipped). Also keep log diag for verification.
        if (getattr(self, "_pathb_skip_dual_base", False)
                and self.base_model is not None
                and self._draft_logits_per_pos):
            # draft_token_ids: [B, K]. Pass full flattened tensor.
            drafts_flat = draft_token_ids.view(-1)
            B, K = draft_token_ids.shape[:2] if draft_token_ids.dim() == 2 \
                else (1, int(drafts_flat.numel()))
            _prof_C = self._prof_start("C_base_pv")
            pv_logits = None
            pv_exc: Exception | None = None
            try:
                if getattr(self, "_use_fast_base_fwd", False):
                    pv_logits = self._base_forward_fast(
                        drafts_flat, next_token_ids=next_token_ids,
                    )
                else:
                    pv_logits = self._base_parallel_verify(
                        drafts_flat, next_token_ids=next_token_ids,
                    )
            except Exception as e:  # noqa: BLE001
                pv_exc = e
            self._prof_end(_prof_C)
            step = getattr(self, "_pv_step_counter", 0) + 1
            self._pv_step_counter = step
            # pv_logits shape: [B, K+1, V]. Take first K → [B, K, V].
            # Per-pos layout expected by _specsteer_sample: list of K
            # tensors, each [B, V]. Stack pv slices into that shape.
            if (pv_exc is None and pv_logits is not None
                    and pv_logits.dim() == 3
                    and pv_logits.shape[0] == B and pv_logits.shape[1] >= K):
                self._base_logits_per_pos = [
                    pv_logits[:, k, :].detach() for k in range(K)
                ]
                if step <= 3 or step % 20 == 1:
                    aug_l0 = self._draft_logits_per_pos[0][0].view(-1)
                    pv_l0 = pv_logits[0, 0]
                    diff_pa = (aug_l0.float() - pv_l0.float()).abs()
                    logger.info(
                        "SpecSteer PathB step %d: B=%d K=%d aug=%d pv=%d  "
                        "|pv-aug|=%.3f",
                        step, B, K,
                        int(aug_l0.argmax().item()),
                        int(pv_l0.argmax().item()),
                        float(diff_pa.max().item()),
                    )
            else:
                if pv_exc is not None:
                    reason = f"PV exception: {pv_exc!r}"
                else:
                    reason = (
                        "PV shape mismatch: got "
                        f"{None if pv_logits is None else tuple(pv_logits.shape)}, "
                        f"expected (B={B}, K>={K}, V)"
                    )
                self._record_pathb_fallback(step, reason)

        self._prof_report_if_due()
        return draft_token_ids
