#!/usr/bin/env python3
"""Run deterministic, conservative checks on strict-target quality captures."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def story_check(text: str) -> dict:
    words = re.findall(r"\S+", text.strip())
    lower = text.lower()
    meta = bool(re.search(r"\b(word count|words?\s*[:=]|counting|exactly\s+50)\b", lower))
    return {"word_count": len(words), "exactly_50_words": len(words) == 50,
            "obvious_counting_or_meta_commentary": meta}


def extract_function(text: str):
    match = re.search(r"(^|\n)(def\s+longest_unique\(s\):[\s\S]*?)(?=\n\S|\Z)", text)
    if not match:
        return None
    source = match.group(2).rstrip()
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return None
    fn = module.body[0]
    if fn.name != "longest_unique" or len(fn.args.args) != 1:
        return None
    # Strictly reject imports, calls to non-whitelisted builtins, and any
    # top-level executable statements.  ``seen.add/remove(...)`` are the two
    # method calls needed by the conventional set-based sliding-window fix.
    allowed_calls = {"set", "enumerate", "max", "len"}
    for node in ast.walk(fn):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda,
                             ast.ClassDef, ast.Global, ast.Nonlocal)):
            return None
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name)
                    and node.value.id == "seen"
                    and node.attr in {"add", "remove"}):
                return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in allowed_calls:
                    return None
            elif not (isinstance(node.func, ast.Attribute)
                      and isinstance(node.func.value, ast.Name)
                      and node.func.value.id == "seen"
                      and node.func.attr in {"add", "remove"}):
                return None
    return source


def coding_check(text: str) -> dict:
    source = extract_function(text)
    if source is None:
        return {"safe_function_extracted": False, "manual_review": True}
    namespace = {"__builtins__": {"set": set, "enumerate": enumerate,
                                    "max": max, "len": len}}
    try:
        exec(compile(source, "<model-output>", "exec"), namespace, namespace)
        fn = namespace["longest_unique"]
        cases = {"": 0, "abcabcbb": 3, "bbbbb": 1, "pwwkew": 3, "abba": 2}
        actual = {value: fn(value) for value in cases}
        return {"safe_function_extracted": True, "manual_review": False,
                "cases": actual, "all_cases_pass": actual == cases}
    except Exception as exc:  # generated code is diagnostic input, not trusted.
        return {"safe_function_extracted": True, "manual_review": True,
                "execution_error": f"{type(exc).__name__}: {exc}"}


def reasoning_check(text: str) -> dict:
    last = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    return {"last_nonempty_line": last,
            "contains_25": bool(re.search(r"(?<!\d)25(?!\d)", last)),
            "contains_50": bool(re.search(r"(?<!\d)50(?!\d)", last)),
            "contains_1250": "1250" in last}


def constraint_check(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    bullets = [line for line in lines if re.match(r"^\s*[-*]\s+", line)]
    prose = [line for line in lines if line not in bullets]
    verbs = []
    for line in bullets:
        rest = re.sub(r"^\s*[-*]\s+", "", line).strip()
        verbs.append(re.split(r"\s+", rest, maxsplit=1)[0].lower() if rest else "")
    return {
        "bullet_count": len(bullets), "exactly_four_bullets": len(bullets) == 4,
        "has_numbered_list": any(re.match(r"^\s*\d+[.)]\s+", line) for line in lines),
        "distinct_first_verbs": len(verbs) == 4 and len(set(verbs)) == 4,
        "semicolon_counts": [line.count(";") for line in bullets],
        "one_semicolon_each": len(bullets) == 4 and all(line.count(";") == 1 for line in bullets),
        "backup_occurrences": len(re.findall(r"backup", text, flags=re.I)),
        "exactly_two_backup_occurrences": len(re.findall(r"backup", text, flags=re.I)) == 2,
        "no_prose_outside_bullets": not prose,
    }


CHECKS = {"story-50": story_check, "coding-debugging": coding_check,
          "reasoning-pen": reasoning_check, "backup-constraints": constraint_check}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {}
    for path in args.captures:
        capture = json.loads(path.read_text())
        cases = capture.get("cases", [capture])
        report[capture.get("mode", path.stem)] = {
            case["id"]: CHECKS.get(case["id"], lambda _text: {})(case["generated_text"])
            for case in cases
        }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
