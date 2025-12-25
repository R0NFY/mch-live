"""
Flask wrapper to run `handler.handler(...)` as an HTTP service.

This matches the expectations of "serverless container" style platforms where the
platform sends an HTTP request containing an event payload.

Why keep this as a real file (not `echo ... > app.py` in Docker)?
- Easier to test and lint
- Easier to extend with structured logging, request IDs, etc.
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from validate_env import validate_env_or_raise

# Fail fast on container startup (missing keys, missing ffmpeg, unwritable /models, etc.)
validate_env_or_raise(mode="handler")

from handler import handler

app = Flask(__name__)


@app.route("/", methods=["POST"])
def invoke():
    event = request.get_json(force=True, silent=False)
    result = handler(event, None)
    status_code = int(result.get("statusCode", 200)) if isinstance(result, dict) else 200
    return jsonify(result), status_code


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200



