# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-light invariants for hybrid SpecSteer models.

The worker-side implementation deliberately uses vLLM's GDN metadata and
state-copy machinery.  These helpers keep architecture detection and the
non-negotiable sharing/isolation checks out of the pure-attention proposer.
"""
from __future__ import annotations

from collections.abc import Iterable


_AUXILIARY_PREFIXES = ("draft_model.", "specsteer_base.")


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


def text_mrope_positions(positions):
    """Return Qwen3.5's native text-only three-axis position representation."""
    return positions.unsqueeze(0).expand(3, -1)
