#!/usr/bin/env python3
"""Compare standalone-target and strict-target generated token captures.

Each input is one JSON object (or a one-record JSONL file) containing generated
token IDs under one of ``generated_token_ids``, ``token_ids``, or
``output_token_ids``. Text is intentionally not used for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOKEN_KEYS = ("generated_token_ids", "token_ids", "output_token_ids")


def load_capture(path: Path) -> tuple[dict[str, Any], list[int]]:
    raw = path.read_text()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(records) != 1:
            raise ValueError(f"{path}: expected one JSON object or one JSONL record")
        value = records[0]
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError(f"{path}: expected a single capture object")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{path}: capture must be a JSON object")
    for key in TOKEN_KEYS:
        if key in value:
            ids = value[key]
            if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
                raise ValueError(f"{path}: {key} must be a list of integer token IDs")
            return value, ids
    raise ValueError(f"{path}: missing generated token IDs; expected one of {TOKEN_KEYS}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True,
                        help="Standalone Qwen3.8 greedy capture JSON")
    parser.add_argument("--asymspec", type=Path, required=True,
                        help="AsymSpec strict_target capture JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Compare only the first N generated tokens")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")

    _, target = load_capture(args.target)
    _, asymspec = load_capture(args.asymspec)
    if args.limit is not None:
        target, asymspec = target[:args.limit], asymspec[:args.limit]

    common = 0
    for expected, actual in zip(target, asymspec):
        if expected != actual:
            break
        common += 1
    differs = common < len(target) or common < len(asymspec)
    result = {
        "target_token_count": len(target),
        "asymspec_token_count": len(asymspec),
        "common_prefix_length": common,
        "first_differing_position": common if differs else None,
        "target_token_id_at_difference": target[common] if common < len(target) else None,
        "asymspec_token_id_at_difference": asymspec[common] if common < len(asymspec) else None,
        "exact_match": target == asymspec,
    }
    print(json.dumps(result, indent=2))
    return 0 if result["exact_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
