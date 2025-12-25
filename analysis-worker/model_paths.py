"""
Model Volume Pattern (shared between local dev and production containers).

Intent:
- Keep *all* model weights and ML caches out of the git repo and out of Docker images.
- In Docker/production, mount a persistent volume at `/models`.
- In local dev, keep a symlink at repo-root `./models` -> a persistent folder (created by
  `../setup_local.sh`).

This module centralizes:
- How we resolve the "models directory"
- Which environment variables we set to route weights/caches there

Why environment variables?
- HuggingFace, transformers, and torch all respect env vars for cache dirs.
- OceanAI uses `path_to_save_` plus our `OCEANAI_CACHE_DIR` env.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_models_dir() -> Path:
    """
    Resolve the models directory with a deterministic priority order.

    Priority:
    1) MODELS_DIR env var (explicit override)
    2) /models (container mount point)
    3) <repo_root>/models (local symlink created by setup_local.sh)
    """

    env = os.getenv("MODELS_DIR")
    if env:
        return Path(env).expanduser()

    # In production containers we mount /models.
    p = Path("/models")
    if p.exists():
        return p

    # In local dev we create repo-root ./models as a symlink to a persistent folder.
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "models"


def configure_model_caches(models_dir: Path | None = None) -> Path:
    """
    Set (default) env vars so ML frameworks store weights in the models directory.

    We use os.environ.setdefault so callers can still override explicitly.
    Returns the resolved models_dir for convenience/logging.
    """

    d = (models_dir or resolve_models_dir()).expanduser()
    d.mkdir(parents=True, exist_ok=True)

    # HuggingFace / transformers cache
    hf_home = d / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))

    # torch cache (torch.hub and some model downloads)
    torch_home = d / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    # A catch-all for libraries honoring XDG cache dir
    xdg_cache = d / "xdg-cache"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    # OceanAI cache (Big Five weights)
    oceanai_cache = d / "oceanai-cache"
    oceanai_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OCEANAI_CACHE_DIR", str(oceanai_cache))

    # Also expose the resolved models dir for other components.
    os.environ.setdefault("MODELS_DIR", str(d))
    return d



