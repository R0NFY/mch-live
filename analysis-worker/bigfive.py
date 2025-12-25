"""
OceanAI Big Five analysis helpers.

This module is intentionally focused:
- transcript parsing / normalization
- running OceanAI Big Five inference

It does NOT contain S3 logic or ChatGPT/OpenAI logic, so it can be imported from
fast local scripts (like analyze_sandbox.py) without requiring production env vars.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Optional

from model_paths import configure_model_caches

logger = logging.getLogger("analysis-worker.bigfive")

# Ensure caches are routed to the model volume as early as possible.
_MODELS_DIR = configure_model_caches()

# OceanAI Configuration (can be overridden via env)
OCEANAI_LANG = os.getenv("OCEANAI_LANG", "ru")
OCEANAI_CORPUS = os.getenv("OCEANAI_CORPUS") or (
    "mupta" if OCEANAI_LANG.lower().startswith("ru") else "fi"
)
OCEANAI_DISK = os.getenv("OCEANAI_DISK", "googledisk")
OCEANAI_CACHE_DIR = os.getenv("OCEANAI_CACHE_DIR", str(_MODELS_DIR / "oceanai-cache"))
OCEANAI_FORCE_ASR = os.getenv("OCEANAI_FORCE_ASR", "false").lower() in ("1", "true", "yes")

_OCEANAI_LOCK = threading.RLock()
_OCEANAI_RUNNER = None
_OCEANAI_CONFIG = None


def _get_transcript_payload(transcript: dict) -> dict:
    if not isinstance(transcript, dict):
        return {}
    payload = transcript.get("transcript")
    return payload if isinstance(payload, dict) else transcript


def _normalize_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("value") or ""
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()
    if content is None:
        return ""
    return str(content).strip()


def extract_conversation(transcript: dict) -> str:
    """
    Extract only the conversation messages from the transcript.
    Returns a clean text format: 'role: content' for each message.
    """

    conversation_lines: list[str] = []

    payload = _get_transcript_payload(transcript)
    if not payload:
        return ""

    # Preferred format: chat_history.items (LiveKit Agents report)
    chat_history = payload.get("chat_history", {})
    items = chat_history.get("items", [])

    for item in items:
        if item.get("type") == "message":
            role = item.get("role", "unknown")
            content = _normalize_content(item.get("content", []))
            content = content.replace("<|im_end|>", "").strip()
            if content:
                role_display = "Пользователь" if role == "user" else "Ассистент"
                conversation_lines.append(f"{role_display}: {content}")

    # Fallback: messages list
    if not conversation_lines:
        messages = payload.get("messages", [])
        for msg in messages:
            role = msg.get("role", "unknown")
            content = _normalize_content(msg.get("content", ""))
            content = content.replace("<|im_end|>", "").strip()
            if content:
                role_display = "Пользователь" if role == "user" else "Ассистент"
                conversation_lines.append(f"{role_display}: {content}")

    # Fallback: events
    if not conversation_lines:
        events = payload.get("events", [])
        for event in events:
            if event.get("type") == "user_input_transcribed":
                text = event.get("transcript", "").strip()
                if text:
                    conversation_lines.append(f"Пользователь: {text}")
            elif event.get("type") == "conversation_item_added":
                item = event.get("item", {})
                if item.get("type") == "message":
                    role = item.get("role", "unknown")
                    content = _normalize_content(item.get("content", []))
                    content = content.replace("<|im_end|>", "").strip()
                    if content:
                        role_display = "Пользователь" if role == "user" else "Ассистент"
                        conversation_lines.append(f"{role_display}: {content}")

    return "\n".join(conversation_lines)


def _safe_float(value: Optional[object]) -> Optional[float]:
    try:
        if value is None:
            return None
        num = float(value)
        if math.isnan(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _load_oceanai_models(ocean, corpus: str, disk: str, lang: str) -> None:
    """
    Load the OceanAI models and weights required for AVT Big Five prediction.

    Note: This intentionally does NOT force-download anything at build-time.
    All downloads should go to OCEANAI_CACHE_DIR (which is routed to /models).
    """

    ocean.load_audio_model_hc(out=False)
    ocean.load_audio_model_nn(out=False)
    ocean.load_audio_model_weights_hc(
        url=ocean.weights_for_big5_["audio"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_audio_model_weights_nn(
        url=ocean.weights_for_big5_["audio"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_video_model_hc(lang=lang, out=False)
    ocean.load_video_model_deep_fe(out=False)
    ocean.load_video_model_nn(out=False)
    ocean.load_video_model_weights_hc(
        url=ocean.weights_for_big5_["video"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_video_model_weights_deep_fe(
        url=ocean.weights_for_big5_["video"][corpus]["fe"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_video_model_weights_nn(
        url=ocean.weights_for_big5_["video"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_text_features(out=False)
    ocean.setup_translation_model(out=False)
    ocean.setup_bert_encoder(force_reload=False, out=False)
    ocean.load_text_model_hc(corpus=corpus, out=False)
    ocean.load_text_model_weights_hc(
        url=ocean.weights_for_big5_["text"][corpus]["hc"][disk],
        force_reload=False,
        out=False,
    )
    ocean.load_text_model_nn(corpus=corpus, out=False)
    ocean.load_text_model_weights_nn(
        url=ocean.weights_for_big5_["text"][corpus]["nn"][disk],
        force_reload=False,
        out=False,
    )

    ocean.load_avt_model_b5(out=False)
    ocean.load_avt_model_weights_b5(
        url=ocean.weights_for_big5_["avt"][corpus]["b5"][disk],
        force_reload=False,
        out=False,
    )


def _get_oceanai_runner():
    """
    Singleton-ish OceanAI runner with thread safety (expensive to init).
    """

    global _OCEANAI_RUNNER, _OCEANAI_CONFIG
    config = (OCEANAI_LANG, OCEANAI_CORPUS, OCEANAI_DISK, OCEANAI_CACHE_DIR)
    with _OCEANAI_LOCK:
        if _OCEANAI_RUNNER is not None and _OCEANAI_CONFIG == config:
            return _OCEANAI_RUNNER

        from oceanai.modules.lab.build import Run

        os.makedirs(OCEANAI_CACHE_DIR, exist_ok=True)
        ocean = Run(lang=OCEANAI_LANG, metadata=False)
        ocean.path_to_save_ = OCEANAI_CACHE_DIR
        ocean.path_to_logs_ = os.path.join(OCEANAI_CACHE_DIR, "logs")

        _load_oceanai_models(ocean, OCEANAI_CORPUS, OCEANAI_DISK, OCEANAI_LANG)

        _OCEANAI_RUNNER = ocean
        _OCEANAI_CONFIG = config
        return _OCEANAI_RUNNER


def analyze_with_oceanai(video_path: str, transcript: Optional[dict] = None) -> dict:
    """
    Analyze video with OceanAI to get Big Five personality traits.

    Returns dict with OCEAN scores (0-1 scale):
    - openness
    - conscientiousness
    - extraversion
    - agreeableness
    - non_neuroticism (inverse of Neuroticism)
    """

    transcript_text = extract_conversation(transcript) if transcript else ""
    text_path: Path | None = None

    # If we have transcript text, avoid ASR to make runs faster + deterministic.
    use_asr = OCEANAI_FORCE_ASR or not transcript_text

    try:
        with _OCEANAI_LOCK:
            ocean = _get_oceanai_runner()

            ocean.path_to_dataset_ = str(Path(video_path).parent)
            ocean.ext_ = [Path(video_path).suffix.lower()]

            if transcript_text:
                text_path = Path(video_path).with_suffix(".txt")
                text_path.write_text(transcript_text, encoding="utf-8")

            logger.info("Running OceanAI Big Five (AVT) analysis... asr=%s", use_asr)
            ok = ocean.get_avt_predictions_gradio(
                paths=[video_path],
                lang=OCEANAI_LANG,
                asr=use_asr,
                accuracy=False,
                logs=False,
                out=False,
                runtime=False,
            )

        if not ok:
            return {"success": False, "error": "OceanAI predictions failed", "scores": None}

        df = ocean.df_files_
        if df is None or df.empty:
            return {"success": False, "error": "OceanAI produced no scores", "scores": None}

        row = df.iloc[0].to_dict()
        scores = {
            "openness": _safe_float(row.get("Openness")),
            "conscientiousness": _safe_float(row.get("Conscientiousness")),
            "extraversion": _safe_float(row.get("Extraversion")),
            "agreeableness": _safe_float(row.get("Agreeableness")),
            "non_neuroticism": _safe_float(row.get("Non-Neuroticism")),
        }

        if not any(v is not None for v in scores.values()):
            return {"success": False, "error": "OceanAI produced empty scores", "scores": None}

        return {"success": True, "scores": scores, "raw_result": None}
    except Exception as e:
        return {"success": False, "error": str(e), "scores": None}
    finally:
        if text_path and text_path.exists():
            text_path.unlink()



