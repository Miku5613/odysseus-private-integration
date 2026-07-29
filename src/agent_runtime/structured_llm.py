from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from src.llm_core import StructuredOutputIncomplete


T = TypeVar("T")


class StructuredModel(Protocol):
    """Minimal model seam for validated control-plane output."""

    def invoke(self, prompt: str) -> Any | Awaitable[Any]:
        ...


StructuredValidator = Callable[[Any], T]
StructuredFallbackCaller = Callable[..., Any | Awaitable[Any]]


class StructuredResponseRejected(ValueError):
    """One candidate chain completed without an acceptable structured result."""


class StructuredControlFailure(RuntimeError):
    """A bounded structured control-plane request exhausted its retry."""

    code = "structured_output_failed"


def _validated_candidate_acceptor(
    validator: StructuredValidator[Any],
) -> StructuredValidator[Any]:
    """Wrap caller-owned validation so chain exhaustion is distinguishable."""

    def validate(response: Any) -> Any:
        try:
            return validator(response)
        except StructuredResponseRejected:
            raise
        except Exception as exc:
            raise StructuredResponseRejected(
                f"{type(exc).__name__}: {exc}"
            ) from exc

    return validate


def build_structured_retry_prompt(
    original_prompt: str,
    error: Exception,
) -> str:
    """Build one bounded repair prompt with a distinct cache key."""

    reason = (
        f"{type(error).__name__}: {error}"
    ).strip()
    if len(reason) > 800:
        reason = reason[:800] + "...<truncated>"

    return "\n".join([
        original_prompt.rstrip(),
        "",
        "STRUCTURED OUTPUT RETRY:",
        "Your previous response could not be validated.",
        "Return a fresh response that follows the exact output contract above.",
        "Return only the required structured payload, with no commentary outside it.",
        f"Validation failure: {reason}",
    ])


async def invoke_validated_fallback_async(
    llm_caller: StructuredFallbackCaller,
    candidates: Sequence[Any],
    prompt: str,
    validator: StructuredValidator[T],
    *,
    call_kwargs: Mapping[str, Any] | None = None,
    max_rounds: int = 2,
) -> Any:
    """Run a validated candidate chain with one bounded repair round.

    The canonical fallback executor still owns candidate ordering and advances
    from the selected model to configured chat fallbacks when one response is
    invalid. This helper only starts a fresh structured round after the entire
    candidate chain was rejected by the caller-owned validation contract.

    The repair round uses a distinct prompt so an invalid cached response from
    the first round cannot be replayed as the retry.
    """

    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    kwargs = dict(call_kwargs or {})
    acceptor = _validated_candidate_acceptor(
        validator
    )
    current_prompt = prompt

    for round_index in range(max_rounds):
        round_kwargs = dict(kwargs)
        round_kwargs["structured_output"] = True
        round_kwargs["reasoning_effort"] = "none"
        try:
            response = llm_caller(
                list(candidates),
                messages=[
                    {
                        "role": "user",
                        "content": current_prompt,
                    }
                ],
                response_validator=acceptor,
                **round_kwargs,
            )
            if inspect.isawaitable(response):
                response = await response
            return response
        except (StructuredResponseRejected, StructuredOutputIncomplete) as exc:
            if round_index + 1 >= max_rounds:
                raise StructuredControlFailure(
                    "Structured output remained invalid after one "
                    "non-thinking retry"
                ) from exc
            current_prompt = build_structured_retry_prompt(
                prompt,
                exc,
            )

    raise AssertionError(
        "Structured fallback loop exited without a result"
    )


async def invoke_structured_model_async(
    model: StructuredModel,
    prompt: str,
    validator: StructuredValidator[T],
    *,
    max_rounds: int = 2,
) -> T:
    """Invoke one structured model call through its strongest available seam.

    Fallback-backed production adapters may expose ``invoke_validated`` so the
    response contract participates in candidate selection. Simpler injected or
    offline models keep the existing ``invoke`` seam and are validated after
    their one response.
    """

    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")

    validated_invoke = getattr(
        model,
        "invoke_validated",
        None,
    )

    if callable(validated_invoke):
        response = validated_invoke(
            prompt,
            validator,
        )
        if inspect.isawaitable(response):
            response = await response
        return validator(response)

    current_prompt = prompt
    last_error: Exception | None = None
    for round_index in range(max_rounds):
        response = model.invoke(current_prompt)
        if inspect.isawaitable(response):
            response = await response
        try:
            return validator(response)
        except Exception as exc:
            last_error = exc
            if round_index + 1 >= max_rounds:
                raise
            current_prompt = build_structured_retry_prompt(
                prompt,
                exc,
            )

    raise AssertionError(
        f"Structured model loop exited without a result: {last_error}"
    )


__all__ = [
    "StructuredFallbackCaller",
    "StructuredControlFailure",
    "StructuredModel",
    "StructuredResponseRejected",
    "StructuredValidator",
    "build_structured_retry_prompt",
    "invoke_structured_model_async",
    "invoke_validated_fallback_async",
]
