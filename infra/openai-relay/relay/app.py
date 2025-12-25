from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com")
RELAY_TOKEN = _env("OPENAI_RELAY_TOKEN") or _env("RELAY_TOKEN")

# Keep conservative defaults; streaming endpoints can run for a while.
TIMEOUT = httpx.Timeout(
    connect=float(_env("OPENAI_CONNECT_TIMEOUT", "10")),
    read=float(_env("OPENAI_READ_TIMEOUT", "300")),
    write=float(_env("OPENAI_WRITE_TIMEOUT", "300")),
    pool=float(_env("OPENAI_POOL_TIMEOUT", "10")),
)

HOP_BY_HOP_HEADERS: set[str] = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=TIMEOUT)
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="OpenAI Relay", version="1.0", lifespan=_lifespan)


def _require_relay_token(request: Request) -> None:
    if not RELAY_TOKEN:
        return

    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != RELAY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid relay token")


def _build_upstream_headers(request: Request) -> dict[str, str]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    headers: dict[str, str] = {}
    for k, v in request.headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        # Never forward the inbound Host to upstream (it would be 127.0.0.1/...).
        if lk == "host":
            continue
        # Let httpx compute Content-Length for the forwarded body.
        if lk == "content-length":
            continue
        # Avoid forwarding Accept-Encoding: httpx transparently decompresses by default,
        # so we'd end up with a Content-Encoding header that doesn't match the body.
        if lk == "accept-encoding":
            continue
        # Never trust inbound Authorization; we always use server key upstream.
        if lk == "authorization":
            continue
        headers[k] = v

    headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
    headers["Accept-Encoding"] = "identity"
    return headers


def _filter_response_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers:
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS:
            continue
        # Let the server/framework manage these.
        if lk in {"content-length"}:
            continue
        # We stream decompressed bytes; never claim they're gzip.
        if lk == "content-encoding":
            continue
        # Cookies from upstream (Cloudflare/OpenAI) are useless for callers and can confuse clients.
        if lk == "set-cookie":
            continue
        out[k] = v
    return out


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def relay(path: str, request: Request) -> Response:
    _require_relay_token(request)

    upstream_url = str(httpx.URL(OPENAI_BASE_URL).join(path))
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"

    headers = _build_upstream_headers(request)
    body = await request.body()

    client: httpx.AsyncClient = request.app.state.http
    req = client.build_request(
        request.method,
        upstream_url,
        headers=headers,
        content=body if body else None,
    )
    upstream_resp = await client.send(req, stream=True)

    resp_headers = _filter_response_headers(upstream_resp.headers.items())
    media_type = upstream_resp.headers.get("content-type")

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        stream(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=media_type,
    )


