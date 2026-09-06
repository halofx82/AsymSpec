"""Narrow hybrid-Mamba helpers used only by SpecSteer's verifier.

The normal vLLM helpers deliberately discover every Mamba cache group in a
model.  SpecSteer has three independent model views in one cache config, so
the target's request-level lifecycle must explicitly own only the verifier
groups.  This module reuses vLLM's buffer classes and kernels; it does not
implement a second state-copy algorithm.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace

import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker import mamba_utils
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch
from vllm.v1.core.sched.output import SchedulerOutput


def _validated_groups(
    kv_cache_config: KVCacheConfig, mamba_group_ids: Sequence[int]
) -> tuple[list[int], MambaSpec]:
    """Validate an explicit, non-empty, uniform set of Mamba group IDs."""
    gids = list(mamba_group_ids)
    if not gids or len(set(gids)) != len(gids):
        raise ValueError("mamba_group_ids must be a non-empty unique list")
    groups = kv_cache_config.kv_cache_groups
    if any(gid < 0 or gid >= len(groups) for gid in gids):
        raise ValueError("mamba_group_ids contains an invalid cache group")
    specs = [groups[gid].kv_cache_spec for gid in gids]
    if not all(isinstance(spec, MambaSpec) for spec in specs):
        raise ValueError("mamba_group_ids must select only Mamba cache groups")
    spec = specs[0]
    assert isinstance(spec, MambaSpec)
    if not all(candidate == spec for candidate in specs):
        raise ValueError("selected Mamba groups must have one uniform MambaSpec")
    return gids, spec


def create_for_groups(
    *,
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    copy_funcs: tuple[MambaStateCopyFunc, ...],
    make_buffer: Callable[..., CpuGpuBuffer],
    device: torch.device,
    mamba_group_ids: Sequence[int],
    with_postprocess_align: bool,
) -> mamba_utils.MambaBuffers:
    """Create native Mamba buffers constrained to explicit cache groups."""
    gids, spec = _validated_groups(kv_cache_config, mamba_group_ids)
    entries_per_req = sum(
        len(kv_cache_config.kv_cache_groups[gid].layer_names) for gid in gids
    ) * len(copy_funcs)
    n = max_num_reqs * entries_per_req
    preprocess = mamba_utils.MambaCopyBuffers(
        src_ptrs=make_buffer(n, dtype=torch.uint64),
        dst_ptrs=make_buffer(n, dtype=torch.uint64),
        sizes=make_buffer(n, dtype=torch.int32),
        mamba_group_ids=gids,
        mamba_spec=spec,
    )
    postprocess = None
    if with_postprocess_align:
        # The native constructor only uses kv_cache_groups during allocation.
        # Rebind the result to the original IDs before it sees the real config.
        filtered_config = SimpleNamespace(
            kv_cache_groups=[kv_cache_config.kv_cache_groups[gid] for gid in gids]
        )
        postprocess = mamba_utils.MambaSpecDecodeGPUContext.create(
            max_num_reqs=max_num_reqs,
            kv_cache_config=filtered_config,  # type: ignore[arg-type]
            num_state_types=len(copy_funcs),
            device=device,
            make_buffer=make_buffer,
        )
        postprocess.mamba_group_ids = gids
    buffers = mamba_utils.MambaBuffers(
        preprocess=preprocess, postprocess_align=postprocess
    )
    assert buffers.preprocess.mamba_group_ids == gids
    assert (buffers.postprocess_align is None
            or buffers.postprocess_align.mamba_group_ids == gids)
    return buffers


def preprocess_for_specsteer_verifier(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, object],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: mamba_utils.MambaCopyBuffers,
    align_ctx: mamba_utils.MambaSpecDecodeGPUContext | None,
) -> None:
    """Use vLLM align preprocessing without changing scheduler cache policy.

    vLLM's assertion couples align mode to prefix caching.  SpecSteer needs
    aligned recurrent-state commit semantics but intentionally has no prefix
    cache.  Pass a one-call config view solely to satisfy that invariant; the
    live CacheConfig is never mutated and scheduler prefix caching stays off.
    """
    if cache_config.enable_prefix_caching:
        mamba_utils.preprocess_mamba(
            scheduler_output, kv_cache_config, cache_config, mamba_state_idx,
            input_batch, requests, forward_context, mamba_state_copy_funcs,
            copy_bufs, align_ctx=align_ctx)
        return
    from dataclasses import replace
    align_config = replace(cache_config, enable_prefix_caching=True)
    mamba_utils.preprocess_mamba(
        scheduler_output, kv_cache_config, align_config, mamba_state_idx,
        input_batch, requests, forward_context, mamba_state_copy_funcs,
        copy_bufs, align_ctx=align_ctx)
