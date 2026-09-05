# SPDX-License-Identifier: Apache-2.0
"""Small, renderer-agnostic helpers for transparent AsymSpec chat serving.

The OpenAI handler owns rendering.  This module deliberately operates on
messages only, so both full and compressed prompts pass through vLLM's normal
``OnlineRenderer`` (including tool schemas and chat-template kwargs).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


RESERVED_EXTRA_ARG = "specsteer_aug_prompt_ids"


class AsymSpecContextError(ValueError):
    """A request cannot be represented safely in the compressed view."""


def has_reserved_extra_arg(extra_args: Mapping[str, Any] | None) -> bool:
    """Whether a client attempts to override the server-owned full prompt."""
    return bool(extra_args and RESERVED_EXTRA_ARG in extra_args)


def is_text_only(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether messages need no multimodal rendering.

    ``None`` content is normal for assistant tool-call messages.  A list is
    OpenAI's content-parts form and may contain image/audio inputs, so it is
    intentionally rejected for the first text-only implementation.
    """
    return all(
        message.get("content") is None or isinstance(message.get("content"), str)
        for message in messages
    )


def split_recent_turns(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[list[Mapping[str, Any]]]]:
    """Keep leading instructions pinned and group each user interaction.

    Every message from a user through (but not including) the next user is an
    atomic group.  Consequently an assistant tool call and its tool result can
    never be separated by history compression.
    """
    pinned: list[Mapping[str, Any]] = []
    index = 0
    while index < len(messages) and messages[index].get("role") in {
        "system",
        "developer",
    }:
        pinned.append(messages[index])
        index += 1

    turns: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] | None = None
    for message in messages[index:]:
        if message.get("role") == "user":
            if current:
                turns.append(current)
            current = [message]
        elif current is not None:
            current.append(message)
        else:
            # Do not retain an orphaned assistant/tool message: without the
            # preceding user interaction it is not a safe recent-history view.
            raise AsymSpecContextError(
                "AsymSpec recent context requires messages after leading "
                "instructions to begin with a user message"
            )
    if current:
        turns.append(current)
    if not turns:
        raise AsymSpecContextError(
            "AsymSpec recent context requires a user message"
        )
    return pinned, turns


def select_recent_messages(
    messages: Sequence[Mapping[str, Any]],
    fits: Callable[[list[Mapping[str, Any]]], bool],
) -> list[Mapping[str, Any]]:
    """Return pinned instructions plus the newest whole turns that fit."""
    pinned, turns = split_recent_turns(messages)
    selected_start = len(turns)
    for index in range(len(turns) - 1, -1, -1):
        candidate = pinned + [message for turn in turns[index:] for message in turn]
        if fits(candidate):
            selected_start = index
        else:
            break
    if selected_start == len(turns):
        raise AsymSpecContextError(
            "Pinned instructions plus the newest user interaction exceed "
            "the AsymSpec compressed-context capacity"
        )
    return pinned + [message for turn in turns[selected_start:] for message in turn]
