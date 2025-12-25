"""
Fast inner-loop sandbox:

Given a static JSON "interview transcript", run ONLY the Big Five (OceanAI) portion.

Why this exists:
- Debugging Big Five parsing + scoring should take seconds, not minutes.
- You should not need a real 15-minute recording just to iterate on analysis logic.

How it works:
- If you pass --video, we use it.
- If you don't, we generate a tiny 1-second dummy mp4 with ffmpeg and run OceanAI
  with ASR disabled (since we have a transcript), which is usually enough to exercise
  the core logic quickly while still keeping the "Big Five via OceanAI" codepath.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from bigfive import analyze_with_oceanai, extract_conversation
from validate_env import validate_env_or_raise


def _make_dummy_video(tmp_dir: Path) -> Path:
    """
    Create a tiny mp4 (1s black video + silent audio) so OceanAI can run without
    needing a real recording.
    """

    out = tmp_dir / "dummy.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        # video: 1 second black
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=25:d=1",
        # audio: 1 second silence
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Big Five (OceanAI) on a transcript JSON")
    parser.add_argument("transcript_json", help="Path to transcript JSON (LiveKit session report export)")
    parser.add_argument("--video", help="Optional path to a video (mp4). If omitted, we generate a tiny dummy mp4.")
    parser.add_argument("--print-conversation", action="store_true", help="Print extracted conversation text.")
    args = parser.parse_args()

    validate_env_or_raise(mode="sandbox")

    transcript_path = Path(args.transcript_json).expanduser()
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    conversation = extract_conversation(transcript)
    if args.print_conversation:
        print(conversation)

    if args.video:
        video_path = Path(args.video).expanduser()
        result = analyze_with_oceanai(str(video_path), transcript)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 2

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        dummy = _make_dummy_video(tmp_dir)
        result = analyze_with_oceanai(str(dummy), transcript)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())



