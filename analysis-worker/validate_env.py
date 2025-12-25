"""
Environment validation for the analysis-worker.

Intent:
- Fail fast with human-readable errors (missing keys, missing binaries, unwritable model volume).
- Make it obvious how to run locally vs in Docker/serverless.

We keep this as a script AND an importable helper so:
- gunicorn/Flask startup can validate immediately
- local developers can run `python validate_env.py` to debug config
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

from model_paths import configure_model_caches


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def validate_env_or_raise(*, mode: str) -> Path:
    """
    Validate environment for a given mode.

    Modes:
    - "handler": serverless container handler (HTTP entrypoint)
    - "worker": polling worker (main.py loop)
    - "sandbox": local one-off scripts
    """

    # Ensure model cache env vars are set and the directory exists.
    models_dir = configure_model_caches()

    # Ensure we can write to the models dir (serverless volumes can be read-only in some setups).
    try:
        test_dir = models_dir / ".validate"
        test_dir.mkdir(parents=True, exist_ok=True)
        probe = test_dir / "probe.txt"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"Model volume is not writable: MODELS_DIR={models_dir}\n"
            f"Fix: mount a writable volume at /models (Docker) or set MODELS_DIR to a writable path.\n"
            f"Underlying error: {e}"
        ) from e

    # ffmpeg is required by OceanAI / media processing.
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available on PATH.\n"
            "Fix (macOS): `brew install ffmpeg`\n"
            "Fix (Docker): ensure ffmpeg is installed in the image (we do this in analysis-worker/Dockerfile)."
        )

    # S3 is required for production modes.
    if mode in ("handler", "worker"):
        _require_env("S3_BUCKET")
        _require_env("S3_ACCESS_KEY")
        _require_env("S3_SECRET")

    # ChatGPT/OpenAI is optional (we allow turning it off explicitly).
    enable_chatgpt = not _is_truthy(os.getenv("DISABLE_CHATGPT"))
    if mode in ("handler", "worker") and enable_chatgpt:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        relay_token = os.getenv("OPENAI_RELAY_TOKEN") or os.getenv("RELAY_TOKEN")

        if openai_base_url:
            # Proxy/relay mode
            if not relay_token:
                raise RuntimeError(
                    "OPENAI_BASE_URL is set but RELAY_TOKEN/OPENAI_RELAY_TOKEN is missing.\n"
                    "Fix: set RELAY_TOKEN (recommended) or OPENAI_RELAY_TOKEN."
                )
        else:
            # Direct mode
            if not openai_api_key:
                raise RuntimeError(
                    "ChatGPT analysis is enabled but OPENAI_API_KEY is missing.\n"
                    "Fix: set OPENAI_API_KEY, or set OPENAI_BASE_URL + RELAY_TOKEN to use your relay, "
                    "or set DISABLE_CHATGPT=true to disable transcript analysis."
                )

    return models_dir


def main(argv: list[str]) -> int:
    mode = "handler"
    if len(argv) > 1:
        mode = argv[1].strip()

    try:
        models_dir = validate_env_or_raise(mode=mode)
    except Exception as e:
        msg = textwrap.dedent(
            f"""
            validate_env failed (mode={mode})
            --------------------------------
            {e}
            """
        ).strip()
        print(msg, file=sys.stderr)
        return 1

    print(f"validate_env OK (mode={mode}) MODELS_DIR={models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))



