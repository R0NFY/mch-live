from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


# Allow importing the handler module directly from the analysis-worker folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _ensure_stub_module(name: str, **attrs) -> None:
    """Create a lightweight stub module if the dependency is missing."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _reload_handler(monkeypatch: pytest.MonkeyPatch, **env: str | None):
    """Reload the handler module with specific environment overrides applied."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    _ensure_stub_module("boto3", client=lambda *args, **kwargs: None)
    _ensure_stub_module("dotenv", load_dotenv=lambda *args, **kwargs: None)
    _ensure_stub_module("openai", OpenAI=object)

    import handler

    return importlib.reload(handler)


def test_default_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _reload_handler(monkeypatch, OCEANAI_CACHE_DIR=None)

    assert handler.OCEANAI_CACHE_DIR == "/app/oceanai-cache"


def test_normalize_openai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _reload_handler(monkeypatch)

    assert (
        handler._normalize_openai_base_url("https://api.example.com")
        == "https://api.example.com/v1"
    )
    assert (
        handler._normalize_openai_base_url("https://api.example.com/v1")
        == "https://api.example.com/v1"
    )
    assert (
        handler._normalize_openai_base_url("https://api.example.com/v1/")
        == "https://api.example.com/v1"
    )
    assert handler._normalize_openai_base_url(None) is None


def test_extract_conversation_prefers_chat_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _reload_handler(monkeypatch)
    transcript = {
        "transcript": {
            "chat_history": {
                "items": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"text": "Привет!"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"text": "Здравствуйте!"}],
                    },
                ]
            },
            # These should be ignored because chat_history exists.
            "messages": [{"role": "user", "content": "fallback"}],
        }
    }

    conversation = handler.extract_conversation(transcript)

    assert conversation.splitlines() == [
        "Пользователь: Привет!",
        "Ассистент: Здравствуйте!",
    ]


def test_extract_conversation_falls_back_to_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _reload_handler(monkeypatch)
    transcript = {
        "transcript": {
            "events": [
                {
                    "type": "user_input_transcribed",
                    "transcript": "Можно узнать статус заказа?",
                },
                {
                    "type": "conversation_item_added",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"text": "Конечно!"}],
                    },
                },
            ]
        }
    }

    conversation = handler.extract_conversation(transcript)

    assert conversation.splitlines() == [
        "Пользователь: Можно узнать статус заказа?",
        "Ассистент: Конечно!",
    ]
