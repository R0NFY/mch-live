#!/usr/bin/env python3
"""
GPT-only local smoke test (no Docker, no amd64 emulation).

What it validates:
- We can reach OpenAI (or your proxy) using OPENAI_BASE_URL
- Auth headers (OPENAI_API_KEY / OPENAI_RELAY_TOKEN) are wired correctly
- The model responds end-to-end

Usage:
  uv run --with openai --with python-dotenv python analysis-worker/test_gpt_local.py \
    --env-file /Users/maximkolomiets/code/mch/.env.local

Relay default (recommended):
  uv run --with openai --with python-dotenv --with httpx python analysis-worker/test_gpt_local.py \
    --env-file /Users/maximkolomiets/code/mch/.env.local \
    --base-url https://openai-relay.mch.expert
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
import requests
import httpx


def _env_present(name: str) -> bool:
    v = os.getenv(name)
    return bool(v and v.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", required=True, help="Path to .env file (secrets are not printed)")
    ap.add_argument(
        "--base-url",
        default=None,
        help="Override OPENAI_BASE_URL (useful if your env-file points to a non-public address). Example: https://openai-relay.mch.expert",
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout seconds (kept small to avoid hanging).",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Max retries for OpenAI client (0 avoids long hangs).",
    )
    ap.add_argument(
        "--health-timeout",
        type=float,
        default=5.0,
        help="Timeout for the initial GET /v1/models connectivity check.",
    )
    args = ap.parse_args()

    load_dotenv(args.env_file, override=False)

    # We only print variable NAMES/presence, never values.
    # If OPENAI_BASE_URL is set, we assume it's your relay. That relay expects:
    #   Authorization: Bearer RELAY_TOKEN
    # and uses its own upstream OPENAI_API_KEY server-side.
    optional = ["OPENAI_BASE_URL", "OPENAI_RELAY_TOKEN", "RELAY_TOKEN", "OPENAI_API_KEY"]
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    relay = os.getenv("OPENAI_RELAY_TOKEN") or os.getenv("RELAY_TOKEN")
    if base_url:
        # Make it resilient to forgetting `/v1`
        base_url = base_url.strip().rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
    if base_url:
        required = ["OPENAI_BASE_URL", "RELAY_TOKEN"]
        # Accept OPENAI_RELAY_TOKEN as an alias for RELAY_TOKEN.
        if not relay:
            missing = ["RELAY_TOKEN"]
        else:
            missing = []
    else:
        required = ["OPENAI_API_KEY"]
        missing = [k for k in required if not _env_present(k)]

    if missing:
        print(f"Missing required env vars: {missing}", file=sys.stderr)
        return 2

    present = {k: _env_present(k) for k in required + optional}
    print("Env presence:", json.dumps(present, ensure_ascii=False))

    timeout_s = args.timeout
    max_retries = args.max_retries

    client_kwargs: dict[str, Any] = {
        # If using a relay, the caller's "API key" is actually the RELAY_TOKEN.
        "api_key": (relay if base_url else os.environ["OPENAI_API_KEY"]),
        "timeout": timeout_s,
        "max_retries": max_retries,
        # Avoid being affected by local HTTP(S)_PROXY env vars.
        "http_client": httpx.Client(timeout=timeout_s, trust_env=False),
    }

    if base_url:
        client_kwargs["base_url"] = base_url

    # Fast connectivity check (helps diagnose "hangs" quickly).
    if base_url:
        models_url = base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {relay}"}
        try:
            s = requests.Session()
            s.trust_env = False
            r = s.get(models_url, headers=headers, timeout=args.health_timeout)
            print("GET /v1/models:", r.status_code)
            if r.status_code >= 400:
                # Don't dump body (could contain details); show only first bytes.
                print("Body head:", (r.text or "")[:200])
        except Exception as e:  # noqa: BLE001
            print(f"GET /v1/models failed: {e}", file=sys.stderr)
            print("Hint: OPENAI_BASE_URL should be your public relay URL, e.g. https://openai-relay.mch.expert (the script appends /v1).", file=sys.stderr)
            return 3

    client = OpenAI(**client_kwargs)

    # Keep this simple: small prompt, low temperature, request JSON-ish response.
    messages = [
        {
            "role": "system",
            "content": "Верни ТОЛЬКО валидный JSON с полями: ok (bool), echo (string). Без markdown.",
        },
        {"role": "user", "content": "Скажи 'привет'."},
    ]

    resp = client.chat.completions.create(model=args.model, messages=messages, temperature=0.0)
    text = resp.choices[0].message.content or ""

    # Print a short, safe summary.
    print("Model:", args.model)
    print("Response chars:", len(text))
    try:
        parsed = json.loads(text)
        print("Parsed JSON keys:", list(parsed.keys()))
        print("ok:", parsed.get("ok"))
        print("echo:", parsed.get("echo"))
    except json.JSONDecodeError:
        print("Response (non-JSON):")
        print(text[:500])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


