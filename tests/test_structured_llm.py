from __future__ import annotations

import asyncio

from src.agent_runtime.structured_llm import (
    StructuredControlFailure,
    StructuredResponseRejected,
    invoke_structured_model_async,
    invoke_validated_fallback_async,
)


def _validator(value):
    if value != "valid":
        raise ValueError("invalid structured response")
    return {"parsed": value}


class ValidatedModel:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        raise AssertionError("validated seam should be preferred")

    def invoke_validated(self, prompt, validator):
        self.calls.append((prompt, validator))
        return "valid"


class PlainAsyncModel:
    def invoke(self, prompt):
        async def response():
            return "valid"

        return response()


def test_structured_invocation_prefers_validated_model_seam():
    model = ValidatedModel()

    result = asyncio.run(
        invoke_structured_model_async(
            model,
            "prompt",
            _validator,
        )
    )

    assert result == {"parsed": "valid"}
    assert model.calls == [("prompt", _validator)]


def test_structured_invocation_preserves_plain_model_compatibility():
    result = asyncio.run(
        invoke_structured_model_async(
            PlainAsyncModel(),
            "prompt",
            _validator,
        )
    )

    assert result == {"parsed": "valid"}


def test_validated_fallback_retries_chain_once_with_distinct_repair_prompt():
    calls = []
    kwargs_by_call = []

    async def fake_caller(
        candidates,
        *,
        messages,
        response_validator,
        **kwargs,
    ):
        calls.append(messages[0]["content"])
        kwargs_by_call.append(kwargs)
        response = "invalid" if len(calls) == 1 else "valid"
        response_validator(response)
        return response

    result = asyncio.run(
        invoke_validated_fallback_async(
            fake_caller,
            [("url", "model", {})],
            "Return the required JSON.",
            _validator,
            max_rounds=2,
        )
    )

    assert result == "valid"
    assert len(calls) == 2
    assert calls[0] == "Return the required JSON."
    assert calls[1] != calls[0]
    assert calls[1].startswith("Return the required JSON.")
    assert "STRUCTURED OUTPUT RETRY:" in calls[1]
    assert kwargs_by_call[0]["structured_output"] is True
    assert kwargs_by_call[0]["reasoning_effort"] == "none"
    assert kwargs_by_call[1]["structured_output"] is True
    assert kwargs_by_call[1]["reasoning_effort"] == "none"


def test_validated_fallback_stops_after_bounded_repair_round():
    calls = []

    async def fake_caller(
        candidates,
        *,
        messages,
        response_validator,
        **kwargs,
    ):
        calls.append(messages[0]["content"])
        response_validator("invalid")
        return "invalid"

    try:
        asyncio.run(
            invoke_validated_fallback_async(
                fake_caller,
                [("url", "model", {})],
                "Return the required JSON.",
                _validator,
                max_rounds=2,
            )
        )
    except StructuredControlFailure as exc:
        assert exc.code == "structured_output_failed"
        pass
    else:
        raise AssertionError(
            "Expected bounded structured fallback exhaustion"
        )

    assert len(calls) == 2
    assert calls[0] != calls[1]


def test_validated_fallback_does_not_retry_transport_failure():
    calls = []

    async def fake_caller(
        candidates,
        *,
        messages,
        response_validator,
        **kwargs,
    ):
        calls.append(messages[0]["content"])
        raise RuntimeError("provider unavailable")

    try:
        asyncio.run(
            invoke_validated_fallback_async(
                fake_caller,
                [("url", "model", {})],
                "Return the required JSON.",
                _validator,
                max_rounds=2,
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "provider unavailable"
    else:
        raise AssertionError("Expected provider failure")

    assert len(calls) == 1
