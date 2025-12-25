#!/usr/bin/env python3
"""
End-to-end local smoke test for the Serverless `analysis-worker` container.
Runs the container locally (optionally under linux/amd64 on Apple Silicon),
uploads a sample recording + transcript to S3 (Yandex Object Storage),
invokes the HTTP handler, and fetches the generated report.

Why this exists:
- `handler.py` expects a real S3 object key and downloads the video from S3.
- We want a realistic check that BOTH OceanAI and GPT(OpenAI) work together.

Prerequisites:
- Docker running
- S3 credentials + bucket configured via env vars (see below)
- OpenAI configured via env vars (recommended: relay; fallback: direct OpenAI key)

Required env vars (same as handler):
- S3_BUCKET
- S3_ACCESS_KEY
- S3_SECRET
Optional:
- S3_ENDPOINT (default: https://storage.yandexcloud.net)
- S3_REGION (default: ru-central1)
- TRANSCRIPTS_PREFIX (default: transcripts/)
- OPENAI_API_KEY (direct OpenAI mode)
- OPENAI_BASE_URL + RELAY_TOKEN (relay/proxy mode; you don't need to expose OpenAI key locally)

Usage:
  python analysis-worker/test_e2e_local_container.py \
    --image cr.yandex/crpv9gnnri1vqg1cof2b/analysis-worker:latest \
    --port 9090 \
    --platform linux/amd64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import boto3
import requests


# Prefer raw.githubusercontent.com: it's less prone to redirect/404 issues than github.com/.../raw/...
DEFAULT_SAMPLE_VIDEO_URL = (
    "https://raw.githubusercontent.com/DmitryRyumin/OCEANAI.github.io/main/media/1.mp4"
)


@dataclass(frozen=True)
class Env:
    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str
    s3_secret: str
    s3_region: str
    transcripts_prefix: str
    openai_api_key: str | None
    openai_base_url: str | None
    openai_relay_token: str | None
    relay_token: str | None


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def load_env_file(path: Path) -> None:
    """
    Load KEY=VALUE pairs into os.environ (without overriding existing vars).

    Supported line formats:
    - KEY=value
    - export KEY=value
    - comments starting with '#'
    """
    if not path.exists():
        raise SystemExit(f"--env-file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env() -> Env:
    env = Env(
        s3_endpoint=os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net"),
        s3_bucket=_require_env("S3_BUCKET"),
        s3_access_key=_require_env("S3_ACCESS_KEY"),
        s3_secret=_require_env("S3_SECRET"),
        s3_region=os.getenv("S3_REGION", "ru-central1"),
        transcripts_prefix=os.getenv("TRANSCRIPTS_PREFIX", "transcripts/"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        openai_relay_token=os.getenv("OPENAI_RELAY_TOKEN"),
        relay_token=os.getenv("RELAY_TOKEN"),
    )
    # Validate OpenAI config:
    # - direct mode: OPENAI_API_KEY
    # - relay mode:  OPENAI_BASE_URL + (OPENAI_RELAY_TOKEN or RELAY_TOKEN)
    if env.openai_base_url:
        if not (env.openai_relay_token or env.relay_token):
            raise SystemExit("Missing relay token: set RELAY_TOKEN (or OPENAI_RELAY_TOKEN) when OPENAI_BASE_URL is set")
    else:
        if not env.openai_api_key:
            raise SystemExit("Missing OPENAI_API_KEY (or set OPENAI_BASE_URL + RELAY_TOKEN for relay mode)")
    return env


def get_s3_client(env: Env):
    return boto3.client(
        service_name="s3",
        endpoint_url=env.s3_endpoint,
        aws_access_key_id=env.s3_access_key,
        aws_secret_access_key=env.s3_secret,
        region_name=env.s3_region,
    )


def download_sample_video(local_path: Path, url: str) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    print(f"Downloading sample video → {local_path}")
    urllib.request.urlretrieve(url, str(local_path))
    return local_path


def put_object_file(s3_client, bucket: str, key: str, path: Path) -> None:
    print(f"Uploading to s3://{bucket}/{key} ({path.stat().st_size / (1024 * 1024):.1f} MB)")
    s3_client.upload_file(str(path), bucket, key)


def put_object_json(s3_client, bucket: str, key: str, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    print(f"Uploading transcript to s3://{bucket}/{key} ({len(body)} bytes)")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def get_object_json(s3_client, bucket: str, key: str) -> dict:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return json.loads(data)


def wait_http_ok(url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return
            last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as e:  # noqa: BLE001 - best-effort polling
            last_err = str(e)
        time.sleep(1.0)
    raise SystemExit(f"Container did not become healthy: {last_err}")


def docker_rm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def docker_run(
    *,
    name: str,
    image: str,
    platform: str | None,
    port: int,
    env: Env,
) -> None:
    docker_rm(name)

    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{port}:8080",
        "-e",
        f"S3_ENDPOINT={env.s3_endpoint}",
        "-e",
        f"S3_BUCKET={env.s3_bucket}",
        "-e",
        f"S3_ACCESS_KEY={env.s3_access_key}",
        "-e",
        f"S3_SECRET={env.s3_secret}",
        "-e",
        f"S3_REGION={env.s3_region}",
        "-e",
        f"TRANSCRIPTS_PREFIX={env.transcripts_prefix}",
    ]
    if env.openai_api_key:
        cmd += ["-e", f"OPENAI_API_KEY={env.openai_api_key}"]
    if env.openai_base_url:
        base = env.openai_base_url.strip().rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        cmd += ["-e", f"OPENAI_BASE_URL={base}"]
    if env.openai_relay_token:
        cmd += ["-e", f"OPENAI_RELAY_TOKEN={env.openai_relay_token}"]
    if env.relay_token:
        cmd += ["-e", f"RELAY_TOKEN={env.relay_token}"]
    if platform:
        cmd += ["--platform", platform]
    cmd.append(image)

    # Never print secrets into logs/CI output.
    redacted = []
    for part in cmd:
        if (
            part.startswith("S3_ACCESS_KEY=")
            or part.startswith("S3_SECRET=")
            or part.startswith("OPENAI_API_KEY=")
            or part.startswith("OPENAI_RELAY_TOKEN=")
            or part.startswith("RELAY_TOKEN=")
        ):
            k = part.split("=", 1)[0]
            redacted.append(f"{k}=<redacted>")
        else:
            redacted.append(part)
    print("Starting container (redacted):")
    print("  " + " ".join(redacted))
    subprocess.run(cmd, check=True)


def invoke_handler(endpoint: str, bucket: str, object_key: str) -> dict:
    event = {
        "messages": [
            {
                "details": {
                    "bucket_id": bucket,
                    "object_id": object_key,
                }
            }
        ]
    }
    r = requests.post(endpoint, json=event, timeout=3600)
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {"raw": r.text}
    if r.status_code != 200:
        raise SystemExit(f"Handler failed: HTTP {r.status_code}: {body}")
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="cr.yandex/crpv9gnnri1vqg1cof2b/analysis-worker:latest")
    ap.add_argument("--name", default="analysis-worker-amd64-smoke")
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--platform", default=None, help="e.g. linux/amd64 (useful on Apple Silicon)")
    ap.add_argument(
        "--env-file",
        default=None,
        help="Path to a local .env file with S3/OpenAI credentials (will not be committed)",
    )
    ap.add_argument(
        "--video",
        default=None,
        help="Path to a local video file to upload instead of downloading a sample",
    )
    ap.add_argument(
        "--video-url",
        default=DEFAULT_SAMPLE_VIDEO_URL,
        help="URL to download a sample video from (used when --video is not provided)",
    )
    ap.add_argument("--keep", action="store_true", help="Do not remove the container after the test")
    args = ap.parse_args()

    if args.env_file:
        load_env_file(Path(args.env_file))

    env = load_env()
    s3 = get_s3_client(env)

    room_name = f"localtest-{int(time.time())}"
    transcript_key = f"{env.transcripts_prefix}{room_name}_transcript.json"

    # Minimal transcript format that `extract_conversation()` understands.
    transcript_payload = {
        "transcript": {
            "chat_history": {
                "items": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"text": "Привет! Подскажи, пожалуйста, сколько стоит доставка?"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"text": "Здравствуйте! Доставка по городу — 500₽. Куда везём?"}],
                    },
                ]
            }
        }
    }

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"--video not found: {video_path}")
    else:
        video_path = download_sample_video(
            Path("/tmp") / "analysis-worker-smoke" / "sample.mp4",
            url=args.video_url,
        )

    # Keep the real extension because the handler derives processing behavior from it.
    # This also makes it easier to correlate the uploaded object with the local file.
    ext = video_path.suffix.lower() or ".mp4"
    recording_key = f"recordings/{room_name}_sample{ext}"
    put_object_file(s3, env.s3_bucket, recording_key, video_path)
    put_object_json(s3, env.s3_bucket, transcript_key, transcript_payload)

    docker_run(name=args.name, image=args.image, platform=args.platform, port=args.port, env=env)
    try:
        wait_http_ok(f"http://localhost:{args.port}/health", timeout_s=180)
        resp = invoke_handler(f"http://localhost:{args.port}/", env.s3_bucket, recording_key)
        print("Handler response:", json.dumps(resp, ensure_ascii=False, indent=2))

        report_key = resp.get("report")
        if not report_key:
            # Handler returns {"status": "ok", "report": "..."} inside a JSON string sometimes,
            # but our Flask wrapper returns JSON-decoded already. Still, keep a fallback.
            raw_body = resp.get("body")
            if isinstance(raw_body, str):
                try:
                    raw_body_json = json.loads(raw_body)
                    report_key = raw_body_json.get("report")
                except Exception:  # noqa: BLE001
                    report_key = None

        if not report_key:
            raise SystemExit(f"Could not determine report key from handler response: {resp}")

        report = get_object_json(s3, env.s3_bucket, report_key)
        print("\n=== Report summary ===")
        ta = report.get("transcript_analysis", {})
        print("GPT success:", ta.get("success"))
        print("GPT error:", ta.get("error"))
        if ta.get("success"):
            print("GPT keys:", list((ta.get("insights") or {}).keys()))

        pers = report.get("personality", {})
        print("OceanAI success:", pers.get("success"))
        print("OceanAI error:", pers.get("error"))
        if pers.get("success"):
            print("OceanAI scores:", pers.get("scores"))

        if not ta.get("success"):
            raise SystemExit("GPT path did not succeed — check OPENAI_API_KEY / OPENAI_BASE_URL and container logs.")

    finally:
        if not args.keep:
            docker_rm(args.name)


if __name__ == "__main__":
    main()


