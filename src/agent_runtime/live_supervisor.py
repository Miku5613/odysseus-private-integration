from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from src.agent_runtime.supervisor_decision import parse_supervisor_decision
from src.agent_runtime.supervisor_reasoner import build_supervisor_prompt
from src.agent_runtime.worker_dispatcher import (
    DispatchResult,
    dispatch_supervisor_decision_async,
)
from src.llm_core import llm_call_async


RUNTIME_VERSION = "general-supervisor-v1"

AsyncLlmCaller = Callable[..., Awaitable[Any]]
AsyncDispatcher = Callable[..., Awaitable[DispatchResult]]


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if str(message.get("role") or "").lower() != "user":
            continue

        content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("content", "response", "output", "text"):
            item = value.get(key)
            if isinstance(item, str):
                return item

    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content

    raise ValueError("LLM response did not contain text")


def _frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _dispatch_payload(dispatch: DispatchResult) -> dict[str, Any]:
    return {
        "requested_capabilities": list(dispatch.requested_capabilities),
        "completed_capabilities": list(dispatch.completed_capabilities),
        "undispatched_capabilities": list(dispatch.undispatched_capabilities),
        "worker_results": [
            {
                "capability": result.capability,
                "status": result.status,
                "summary": result.summary,
                "data": result.data,
                "evidence": list(result.evidence),
                "missing_information": list(result.missing_information),
                "error": result.error,
            }
            for result in dispatch.worker_results
        ],
    }


def _fallback_summary(
    dispatch: DispatchResult,
    reasoning_summary: str,
) -> str:
    summaries = [
        result.summary.strip()
        for result in dispatch.worker_results
        if result.summary and result.summary.strip()
    ]

    if summaries:
        return "\n\n".join(summaries)

    if dispatch.undispatched_capabilities:
        names = ", ".join(dispatch.undispatched_capabilities)
        return f"The Supervisor selected capabilities that are not executable yet: {names}."

    return reasoning_summary or "The Supervisor completed without producing a final response."


async def stream_general_supervisor_sse(
    endpoint_url: str,
    model: str,
    messages: list[dict[str, Any]],
    headers: dict[str, Any] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    prompt_type: str | None = None,
    session_id: str | None = None,
    owner: str | None = None,
    workspace: str | None = None,
    active_document: Any = None,
    llm_caller: AsyncLlmCaller = llm_call_async,
    dispatcher: AsyncDispatcher = dispatch_supervisor_decision_async,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Run the model-backed General Supervisor through the existing SSE seam."""

    started_at = time.time()
    query = _latest_user_message(messages)

    if not query:
        raise ValueError("General Supervisor could not find a user message")

    decision_prompt = build_supervisor_prompt(query)

    decision_response = await llm_caller(
        endpoint_url,
        model,
        [{"role": "user", "content": decision_prompt}],
        headers=headers,
        temperature=0.0,
        max_tokens=min(max_tokens, 1400),
        prompt_type=prompt_type,
    )

    decision = parse_supervisor_decision(_text(decision_response).strip())

    if decision.clarification_needed:
        missing = ", ".join(decision.missing_information)
        clarification = (
            f"I need one detail before I can continue: {missing}"
            if missing
            else "I need one clarification before I can continue safely."
        )

        yield _frame({"delta": clarification})
        yield _frame({
            "type": "metrics",
            "data": {
                "runtime": RUNTIME_VERSION,
                "model": model,
                "elapsed_s": round(time.time() - started_at, 3),
                "selected_capabilities": list(decision.selected_capabilities),
                "clarification_needed": True,
                "langgraph_runtime": True,
            },
        })
        yield "data: [DONE]\n\n"
        return

    supervisor_state = {
        "missing_information": list(decision.missing_information),
        "worker_context": {
            "query": query,
            "topic": query,
            "research_question": query,
            "requested_capability": query,
            "arguments": {},
            "owner": owner,
            "workspace": workspace,
            "active_doc": active_document,
            "root_id": kwargs.get("root_id") or "ideaverse_vault",
        },
    }

    dispatch = await dispatcher(
        decision,
        query,
        supervisor_state=supervisor_state,
        session_id=session_id,
    )

    evidence_payload = _dispatch_payload(dispatch)

    synthesis_prompt = "\n".join([
        "You are the final-response stage of the Odysseus General Supervisor.",
        "",
        "Answer the user using only the worker results and evidence below.",
        "Do not claim an operation succeeded unless a worker status says completed.",
        "Clearly state when confirmation, clarification, or additional execution is required.",
        "Do not mention internal class names, routing code, or JSON schemas.",
        "",
        "USER REQUEST:",
        query,
        "",
        "SUPERVISOR RATIONALE:",
        decision.reasoning_summary,
        "",
        "WORKER RESULTS:",
        json.dumps(evidence_payload, ensure_ascii=False, default=str),
    ])

    try:
        final_response = await llm_caller(
            endpoint_url,
            model,
            [{"role": "user", "content": synthesis_prompt}],
            headers=headers,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_type=prompt_type,
        )
        final_text = _text(final_response).strip()
    except Exception:
        final_text = _fallback_summary(
            dispatch,
            decision.reasoning_summary,
        )

    if not final_text:
        final_text = _fallback_summary(
            dispatch,
            decision.reasoning_summary,
        )

    yield _frame({"delta": final_text})
    yield _frame({
        "type": "metrics",
        "data": {
            "runtime": RUNTIME_VERSION,
            "model": model,
            "elapsed_s": round(time.time() - started_at, 3),
            "selected_capabilities": list(decision.selected_capabilities),
            "completed_capabilities": list(dispatch.completed_capabilities),
            "undispatched_capabilities": list(dispatch.undispatched_capabilities),
            "clarification_needed": False,
            "langgraph_runtime": True,
        },
    })
    yield "data: [DONE]\n\n"


__all__ = ["RUNTIME_VERSION", "stream_general_supervisor_sse"]
