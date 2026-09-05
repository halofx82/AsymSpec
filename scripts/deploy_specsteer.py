#!/usr/bin/env python3
"""Deploy/revert the version- and hash-checked AsymSpec vLLM 0.28 payload."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "vllm_specsteer/vllm_0_28"
SUPPORTED_VLLM_VERSION = "0.28.0"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def package_root():
    dist = importlib.metadata.distribution("vllm")
    if dist.version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(f"Expected vLLM {SUPPORTED_VLLM_VERSION}; found {dist.version}")
    return Path(dist.locate_file("vllm")).resolve()


def deployment_map(pkg=None):
    pkg = package_root() if pkg is None else pkg
    manifest = json.loads((PATCH_DIR / "manifest.json").read_text())
    return [(PATCH_DIR / entry["source"], pkg / entry["target"], entry)
            for entry in manifest["files"]]


def apply(pkg, backup):
    pairs = deployment_map(pkg)
    state_path = backup / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state is not None and state["package"] != str(pkg):
        raise RuntimeError("Backups belong to a different installation")
    for src, dst, entry in pairs:
        if not src.is_file() or not dst.parent.is_dir():
            raise RuntimeError(f"Missing source or target directory: {src}, {dst.parent}")
        current = sha256(dst)
        allowed = {entry["upstream_sha256"], sha256(src)}
        if state is not None:
            allowed.add(state["deployed"].get(entry["target"]))
        if current not in allowed:
            raise RuntimeError(f"Unrecognized installed file; refusing overwrite: {dst}")
        if current is None and entry["upstream_sha256"] is not None:
            raise RuntimeError(f"Missing upstream file: {dst}")
        if state is None and current == sha256(src) and entry["upstream_sha256"] is not None:
            raise RuntimeError(f"Patched file without recoverable original: {dst}")
    if state is None:
        state = {"package": str(pkg), "original": {}, "deployed": {}}
        backup.mkdir(parents=True, exist_ok=True)
    # A payload can gain files after an earlier deployment. Record their
    # originals before copying so --revert remains complete and safe.
    added_originals = False
    for src, dst, entry in pairs:
        rel = entry["target"]
        if rel in state["original"]:
            continue
        current = sha256(dst)
        if current != entry["upstream_sha256"]:
            raise RuntimeError(
                "Cannot add a new payload entry after it has already been "
                f"modified; restore the upstream file first: {dst}"
            )
        state["original"][rel] = current
        if dst.exists():
            saved = backup / "original" / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, saved)
        added_originals = True
    if added_originals:
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    try:
        for src, dst, entry in pairs:
            shutil.copy2(src, dst)
            state["deployed"][entry["target"]] = sha256(src)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    except Exception:
        revert(pkg, backup, check_deployed=False)
        raise
    print(f"Deployed {len(pairs)} files into {pkg}")


def revert(pkg, backup, check_deployed=True):
    state_path = backup / "state.json"
    if not state_path.exists():
        raise RuntimeError(f"No deployment backup at {backup}")
    state = json.loads(state_path.read_text())
    if state["package"] != str(pkg):
        raise RuntimeError("Backups belong to a different installation")
    for rel, original_hash in state["original"].items():
        if original_hash is not None and sha256(backup / "original" / rel) != original_hash:
            raise RuntimeError(f"Missing/corrupt backup: {rel}")
        if check_deployed and sha256(pkg / rel) not in {
                original_hash, state["deployed"].get(rel)}:
            raise RuntimeError(f"Installed file changed since deployment: {rel}")
    for rel, original_hash in state["original"].items():
        dst = pkg / rel
        if original_hash is None:
            dst.unlink(missing_ok=True)
        else:
            shutil.copy2(backup / "original" / rel, dst)
    print("Restored original files; removed added modules. Backups retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ("check", "apply", "revert"):
        action.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    try:
        pkg = package_root()
        backup = ROOT / ".backups/vllm_0_28"
        if args.apply:
            apply(pkg, backup)
        elif args.revert:
            revert(pkg, backup)
        else:
            for src, dst, entry in deployment_map(pkg):
                current = sha256(dst)
                status = ("deployed" if current == sha256(src) else
                          "upstream" if current == entry["upstream_sha256"] else "MODIFIED")
                print(f"{status:10} {dst}")
    except (RuntimeError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
