"""Centralized path and model configuration for AsymSpec scripts.

Override via env vars (all optional, sensible defaults):

  ASYMSPEC_REPO_ROOT        — repo root         (default: dir of this file)
  ASYMSPEC_MODEL_LLM        — verifier HF id    (default: Qwen/Qwen3-32B)
  ASYMSPEC_MODEL_SLM_PREFIX — drafter prefix    (default: Qwen/Qwen3-)
  ASYMSPEC_MODEL_VL         — VL drafter HF id  (default: Qwen/Qwen3-VL-2B-Instruct)
  ASYMSPEC_DATA_DIR         — dataset root      (default: REPO_ROOT/data)
  ASYMSPEC_DOTENV           — secrets file      (default: REPO_ROOT/.env)

Model "paths" are HF model IDs (e.g. ``Qwen/Qwen3-32B``); transformers /
vLLM resolve them via the HF cache, so no absolute paths are baked in.
Control where weights live by setting ``HF_HOME`` / ``HF_HUB_CACHE``.

Dataset paths point at gitignored files; populate them via
``scripts/download_datasets.sh`` or override ``ASYMSPEC_DATA_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get(
    "ASYMSPEC_REPO_ROOT",
    Path(__file__).resolve().parent,
))

# Keep runtime artifacts local to this checkout, including spawned TP workers.
os.environ.setdefault("VLLM_CACHE_ROOT", str(REPO_ROOT / ".cache/vllm"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(REPO_ROOT / ".cache/inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(REPO_ROOT / ".cache/triton"))

# ── Models ────────────────────────────────────────────────────────────
# HF IDs — resolved via HF cache, no machine-specific absolute paths.
LLM_MODEL = os.environ.get("ASYMSPEC_MODEL_LLM", "Qwen/Qwen3-32B")
VL_MODEL = os.environ.get("ASYMSPEC_MODEL_VL", "Qwen/Qwen3-VL-2B-Instruct")


def slm_model(size: str) -> str:
    """Drafter HF id, e.g. ``slm_model("0.6B") -> "Qwen/Qwen3-0.6B"``."""
    prefix = os.environ.get("ASYMSPEC_MODEL_SLM_PREFIX", "Qwen/Qwen3-")
    return f"{prefix}{size}"


# Back-compat aliases — minimizes diff in existing scripts.
LLM_PATH = LLM_MODEL
VL_PATH = VL_MODEL

# ── Data ──────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("ASYMSPEC_DATA_DIR", REPO_ROOT / "data"))
LB_RAW = DATA_DIR / "longbench" / "raw"
LB_SUMMARIES = DATA_DIR / "longbench" / "summaries.jsonl"
MC_QUESTIONS = DATA_DIR / "multi-challenge" / "data" / "benchmark_questions.jsonl"
MC_SUMMARIES = DATA_DIR / "multi-challenge" / "summaries.json"
MV_CAPTIONS = DATA_DIR / "mathvista" / "captions_bard.json"
MV_OCRS = DATA_DIR / "mathvista" / "ocrs_easyocr.json"
GAIA_DIR = DATA_DIR / "gaia"
APIBANK_DIR = DATA_DIR / "DAMO-ConvAI" / "api-bank"


# ── vLLM patch deploy targets ─────────────────────────────────────────
def vllm_specsteer_targets() -> tuple[Path, Path]:
    """Where ``vllm_specsteer/<ver>/*.py`` should be deployed inside
    the currently-installed vLLM package.

    Returns ``(specsteer_model_path, specsteer_sampler_path)``.
    Dynamic — works regardless of Python version or install location.
    """
    try:
        import vllm
    except ImportError as e:
        raise RuntimeError(
            "vllm not installed in current env. "
            "Run `uv pip install vllm==0.28.0` first."
        ) from e
    vllm_pkg = Path(vllm.__file__).parent
    return (
        vllm_pkg / "v1" / "spec_decode" / "specsteer_model.py",
        vllm_pkg / "v1" / "sample" / "specsteer_sampler.py",
    )


# ── GAIA snapshot (pinned to paper's revision) ────────────────────────
GAIA_REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"


def gaia_validation_dir() -> Path:
    """Path to GAIA 2023/validation directory.

    Downloads the pinned snapshot if missing; subsequent calls are O(1) cache
    lookups. Pinned to the paper's revision for reproducibility.
    """
    from huggingface_hub import snapshot_download
    repo_dir = Path(snapshot_download(
        "gaia-benchmark/GAIA",
        repo_type="dataset",
        revision=GAIA_REVISION,
        allow_patterns=["2023/validation/*"],
    ))
    return repo_dir / "2023" / "validation"


# ── Secrets ───────────────────────────────────────────────────────────
def _read_env_file() -> dict:
    env_path = Path(os.environ.get("ASYMSPEC_DOTENV", REPO_ROOT / ".env"))
    if not env_path.exists():
        return {}
    out = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_tavily_key() -> str:
    """Read TAVILY_API_KEY from .env file or environment."""
    if k := os.environ.get("TAVILY_API_KEY"):
        return k
    env = _read_env_file()
    if k := env.get("TAVILY_API_KEY"):
        return k
    env_path = Path(os.environ.get("ASYMSPEC_DOTENV", REPO_ROOT / ".env"))
    raise RuntimeError(
        f"TAVILY_API_KEY not found in {env_path} or environment. "
        f"Add 'TAVILY_API_KEY=tvly-xxxxx' to {env_path}."
    )


def get_hf_token() -> str | None:
    """Read HF_TOKEN from environment or .env file (None if unset)."""
    if k := os.environ.get("HF_TOKEN"):
        return k
    return _read_env_file().get("HF_TOKEN")
