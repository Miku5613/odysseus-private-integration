from __future__ import annotations

import asyncio
import json

import pytest

import src.llm_core as llm_core


CANDIDATES = [
    ("http://primary.example/v1/chat/completions", "primary", {}),
    ("http://fallback.example/v1/chat/completions", "fallback", {}),
]


def _validator(response):
    if not isinstance(response, str) or not response.strip():
        raise ValueError("structured response is empty")
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("structured response is invalid JSON") from exc
    if payload != {"ok": True}:
        raise ValueError("structured response violates contract")
    return payload


def test_async_validated_fallback_rejects_empty_primary(monkeypatch):
    calls = []

    async def fake_call(url, model, messages, headers=None, **kwargs):
        calls.append(model)
        return "" if model == "primary" else '{"ok": true}'

    monkeypatch.setattr(llm_core, "llm_call_async", fake_call)

    result = asyncio.run(
        llm_core.llm_call_async_with_fallback(
            CANDIDATES,
            [{"role": "user", "content": "Return JSON."}],
            response_validator=_validator,
        )
    )

    assert result == '{"ok": true}'
    assert calls == ["primary", "fallback"]


def test_async_validated_fallback_rejects_malformed_primary(monkeypatch):
    calls = []

    async def fake_call(url, model, messages, headers=None, **kwargs):
        calls.append(model)
        return "{not-json" if model == "primary" else '{"ok": true}'

    monkeypatch.setattr(llm_core, "llm_call_async", fake_call)

    result = asyncio.run(
        llm_core.llm_call_async_with_fallback(
            CANDIDATES,
            [{"role": "user", "content": "Return JSON."}],
            response_validator=_validator,
        )
    )

    assert result == '{"ok": true}'
    assert calls == ["primary", "fallback"]


def test_async_validated_fallback_raises_after_all_candidates_invalid(
    monkeypatch,
):
    async def fake_call(url, model, messages, headers=None, **kwargs):
        return ""

    monkeypatch.setattr(llm_core, "llm_call_async", fake_call)

    with pytest.raises(
        ValueError,
        match="structured response is empty",
    ):
        asyncio.run(
            llm_core.llm_call_async_with_fallback(
                CANDIDATES,
                [{"role": "user", "content": "Return JSON."}],
                response_validator=_validator,
            )
        )


def test_async_fallback_without_validator_preserves_chat_semantics(
    monkeypatch,
):
    calls = []

    async def fake_call(url, model, messages, headers=None, **kwargs):
        calls.append(model)
        return ""

    monkeypatch.setattr(llm_core, "llm_call_async", fake_call)

    result = asyncio.run(
        llm_core.llm_call_async_with_fallback(
            CANDIDATES,
            [{"role": "user", "content": "Say hello."}],
        )
    )

    assert result == ""
    assert calls == ["primary"]


def test_sync_validated_fallback_uses_same_acceptance_contract(monkeypatch):
    calls = []

    def fake_call(url, model, messages, headers=None, **kwargs):
        calls.append(model)
        return "{bad" if model == "primary" else '{"ok": true}'

    monkeypatch.setattr(llm_core, "llm_call", fake_call)

    result = llm_core.llm_call_with_fallback(
        CANDIDATES,
        [{"role": "user", "content": "Return JSON."}],
        response_validator=_validator,
    )

    assert result == '{"ok": true}'
    assert calls == ["primary", "fallback"]
