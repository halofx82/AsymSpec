#!/usr/bin/env python3
"""Summarize a buffered ``ASYMSPEC_PROFILE`` JSONL artifact on CPU."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def total(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--csv", type=Path,
                        help="optional compact per-outer-step CSV")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.profile.read_text().splitlines()
            if line.strip()]
    if not rows:
        raise SystemExit("profile contains no rows")
    if args.csv:
        fields = ["step", "committed_length_before", "K", "drafter_new_tokens",
                  "replayed_history_tokens", "replay_ms", "proposal_ms",
                  "verifier_ms", "sampler_ms"]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    replayed = total(rows, "replayed_history_tokens")
    drafted = total(rows, "drafter_new_tokens")
    ms = {key: total(rows, key) for key in
          ("replay_ms", "proposal_ms", "verifier_ms", "sampler_ms",
           "drafter_postprocess_ms")}
    covered = sum(ms.values())
    result = {
        "profiled_outer_steps": len(rows),
        "replay_invocations": len(rows),
        "historical_tokens_replayed": replayed,
        "mean_replay_tokens_per_outer_step": replayed / len(rows),
        "max_replay_tokens_per_outer_step": max(
            int(row.get("replayed_history_tokens", 0)) for row in rows),
        "new_draft_tokens": drafted,
        "replay_to_useful_draft_ratio": replayed / drafted if drafted else None,
        "gpu_region_ms": ms,
        "gpu_region_total_ms": covered,
        "region_percent": {key: value / covered * 100 if covered else 0
                           for key, value in ms.items()},
        "replay_ms_median": statistics.median(
            float(row["replay_ms"]) for row in rows),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
