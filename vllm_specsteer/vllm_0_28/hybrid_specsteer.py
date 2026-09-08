# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-light invariants for hybrid SpecSteer models.

The worker-side implementation deliberately uses vLLM's GDN metadata and
state-copy machinery.  These helpers keep architecture detection and the
non-negotiable sharing/isolation checks out of the pure-attention proposer.
"""
from __future__ import annotations

from collections.abc import Iterable
import os


_AUXILIARY_PREFIXES = ("draft_model.", "specsteer_base.")


def is_strict_target_mode() -> bool:
    """Whether this worker is running target-authoritative SpecSteer.

    This remains environment-selected for compatibility with the existing
    sampler switch.  It is deliberately a narrow predicate: the normal JSD
    topology continues to require both 4B views.
    """
    return os.environ.get("ASYMSPEC_METHOD", "gamma_rule").lower() == "strict_target"


def uses_compressed_base(strict_target: bool) -> bool:
    """Whether a SpecSteer topology needs the compressed 4B model view."""
    return not strict_target


def is_auxiliary_state_layer(layer_name: str) -> bool:
    """Whether a cache layer belongs to a non-verifier SpecSteer view."""
    return layer_name.startswith(_AUXILIARY_PREFIXES)


def is_hybrid_model_config(model_config: object) -> bool:
    """Whether vLLM resolved this model to a hybrid attention/GDN model."""
    return bool(getattr(model_config, "is_hybrid", False))


def split_hybrid_layer_names(layer_names: Iterable[str]) -> tuple[set[str], set[str]]:
    """Separate full-attention and GDN names without model-name special cases."""
    names = set(layer_names)
    return (
        {name for name in names if ".self_attn" in name},
        {name for name in names if ".linear_attn" in name},
    )


def require_distinct_runtime_layer_sets(
    draft_layer_names: Iterable[str], base_layer_names: Iterable[str]
) -> None:
    """Fail before inference if the two Qwen3.5 views alias runtime layers."""
    draft = set(draft_layer_names)
    base = set(base_layer_names)
    if not draft or not base:
        raise RuntimeError("Hybrid SpecSteer did not register both 4B layer trees")
    if draft & base:
        raise RuntimeError("Hybrid SpecSteer drafter/base runtime layers overlap")
    for layers, label in ((draft, "draft_model"), (base, "specsteer_base")):
        full, gdn = split_hybrid_layer_names(layers)
        if not full or not gdn:
            raise RuntimeError(
                f"Hybrid SpecSteer {label} must contain both self_attn and "
                "linear_attn layers")


def require_strict_target_runtime_layer_set(
    draft_layer_names: Iterable[str], all_layer_names: Iterable[str],
) -> None:
    """Validate strict-target's two-view topology before inference.

    Strict target has a verifier and one full-context drafter.  In
    particular, a compressed ``specsteer_base`` layer must never have been
    registered: its presence means it can still consume KV/GDN resources.
    """
    draft = set(draft_layer_names)
    all_names = set(all_layer_names)
    if any(name.startswith("specsteer_base.") for name in all_names):
        raise RuntimeError("Strict-target registered specsteer_base runtime layers")
    full, gdn = split_hybrid_layer_names(draft)
    if draft and (not full or not gdn):
        raise RuntimeError(
            "Hybrid strict-target drafter must contain both self_attn and "
            "linear_attn layers")


def text_mrope_positions(positions):
    """Return Qwen3.5's native text-only three-axis position representation."""
    return positions.unsqueeze(0).expand(3, -1)
