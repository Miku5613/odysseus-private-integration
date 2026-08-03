import asyncio

import httpx
import pytest

from src import llm_core


class _FakeAsyncClient:
    def __init__(self, payload):
        self.payload = payload
        self.request_json = None

    async def post(self, *args, **kwargs):
        self.request_json = kwargs.get("json")
        request = httpx.Request("POST", "http://local/v1/chat/completions")
        return httpx.Response(200, request=request, json=self.payload)


def _call(monkeypatch, payload, **kwargs):
    llm_core._response_cache.clear()
    client = _FakeAsyncClient(payload)
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)
    result = asyncio.run(llm_core.llm_call_async(
        "http://local/v1",
        "qwen3.5:9b",
        [{"role": "user", "content": "Return JSON."}],
        max_retries=1,
        structured_output=True,
        **kwargs,
    ))
    return result, client.request_json


def _sync_call(monkeypatch, payload, **kwargs):
    llm_core._response_cache.clear()
    captured = {}

    def fake_post(url, headers, *, json, timeout):
        captured["json"] = json
        request = httpx.Request(
            "POST",
            "http://local/v1/chat/completions",
        )
        return httpx.Response(200, request=request, json=payload)

    monkeypatch.setattr(
        llm_core,
        "httpx_post_kimi_aware",
        fake_post,
    )
    result = llm_core.llm_call(
        "http://local/v1",
        "qwen3.5:9b",
        [{"role": "user", "content": "Return JSON."}],
        structured_output=True,
        **kwargs,
    )
    return result, captured["json"]


def test_structured_request_uses_json_format_and_non_thinking_effort(monkeypatch):
    result, request_json = _call(
        monkeypatch,
        {
            "choices": [{
                "message": {"content": '{"ok": true}', "reasoning": ""},
                "finish_reason": "stop",
            }]
        },
        reasoning_effort="none",
    )

    assert result == '{"ok": true}'
    assert request_json["response_format"] == {"type": "json_object"}
    assert request_json["reasoning_effort"] == "none"


def test_structured_response_extracts_only_complete_reasoning_json(monkeypatch):
    result, _ = _call(
        monkeypatch,
        {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning": 'analysis before {"ok": true} after',
                },
                "finish_reason": "stop",
            }]
        },
    )

    assert result == '{"ok": true}'


def test_sync_structured_request_uses_non_thinking_json_mode(monkeypatch):
    result, request_json = _sync_call(
        monkeypatch,
        {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning": 'analysis {"ok": true}',
                },
                "finish_reason": "stop",
            }]
        },
        reasoning_effort="none",
    )

    assert result == '{"ok": true}'
    assert request_json["response_format"] == {"type": "json_object"}
    assert request_json["reasoning_effort"] == "none"


def test_structured_response_rejects_truncated_reasoning(monkeypatch):
    with pytest.raises(
        llm_core.StructuredOutputIncomplete,
        match="truncated",
    ):
        _call(
            monkeypatch,
            {
                "choices": [{
                    "message": {
                        "content": "",
                        "reasoning": '{"ok": tru',
                    },
                    "finish_reason": "length",
                }]
            },
        )


def test_normal_response_does_not_promote_message_reasoning(monkeypatch):
    llm_core._response_cache.clear()
    client = _FakeAsyncClient({
        "choices": [{
            "message": {
                "content": "",
                "reasoning": "private chain of thought",
            },
            "finish_reason": "stop",
        }]
    })
    monkeypatch.setattr(llm_core, "_get_http_client", lambda: client)

    result = asyncio.run(llm_core.llm_call_async(
        "http://local/v1",
        "qwen3.5:9b",
        [{"role": "user", "content": "Answer."}],
        max_retries=1,
    ))

    assert result == ""
