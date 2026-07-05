from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Optional, Set

from src.agent_runtime.legacy_sse_adapter import to_legacy_sse_frame
from src.agent_runtime.streaming import run_stream


RUNTIME_VERSION = "v0.10.2"


def _extract_latest_user_message(messages: list[dict[str, Any]] | None) -> str:
    """Find the latest user-visible message from an OpenAI-style message list."""
    if not messages:
        return ""

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text

        # Basic support for content parts.
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    value = part.get("text") or part.get("content")
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                elif isinstance(part, str) and part.strip():
                    parts.append(part.strip())
            if parts:
                return "\n".join(parts).strip()

    return ""


def _legacy_frame(payload: dict[str, Any] | str) -> str:
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return "data: " + json.dumps(payload, ensure_ascii=False, default=str) + "\n\n"


def _metrics_payload(
    *,
    model: str,
    requested_model: str | None,
    started_at: float,
    agent_rounds: int,
    agent_tool_calls: int,
    final_chars: int,
) -> dict[str, Any]:
    elapsed = max(0.0, time.time() - started_at)
    return {
        "type": "metrics",
        "data": {
            "runtime": RUNTIME_VERSION,
            "model": model,
            "requested_model": requested_model or model,
            "elapsed_s": round(elapsed, 3),
            "agent_rounds": agent_rounds,
            "agent_tool_calls": agent_tool_calls,
            "final_chars": final_chars,
            "langgraph_runtime": True,
        },
    }


async def stream_langgraph_agent_legacy_sse(
    endpoint_url: str,
    model: str,
    messages: list[dict[str, Any]],
    headers: Optional[dict[str, Any]] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    prompt_type: Optional[str] = None,
    max_rounds: int = 8,
    max_tool_calls: int = 0,
    context_length: int = 0,
    active_document: Any = None,
    active_email: Optional[dict[str, str]] = None,
    session_id: Optional[str] = None,
    disabled_tools: Optional[Set[str]] = None,
    owner: Optional[str] = None,
    relevant_tools: Optional[Set[str]] = None,
    fallbacks: Optional[list[tuple]] = None,
    plan_mode: bool = False,
    approved_plan: Optional[str] = None,
    tool_policy: Any = None,
    workspace: Optional[str] = None,
    agent_profile: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Compatibility wrapper for future chat_routes.py integration.

    This intentionally mirrors stream_agent_loop's broad call signature so the
    route can later switch runtimes behind a feature flag with minimal code.

    Current v0.8 limitations:
    - read-only file-inspection runtime only
    - uses latest user message as the question
    - does not yet honor the full Odysseus tool universe
    - does not save responses; chat_routes.py remains responsible for saving
    """

    started_at = time.time()
    requested_model = model
    question = _extract_latest_user_message(messages)

    agent_rounds = 0
    agent_tool_calls = 0
    final_answer = ""

    if not question:
        yield _legacy_frame({
            "delta": "I could not find a user message to process with the LangGraph runtime."
        })
        yield _legacy_frame(_metrics_payload(
            model=model,
            requested_model=requested_model,
            started_at=started_at,
            agent_rounds=agent_rounds,
            agent_tool_calls=agent_tool_calls,
            final_chars=0,
        ))
        yield "data: [DONE]\n\n"
        return

    # v0.8 hard safety rail: only use this wrapper for file/workspace inspection.
    # Full agent/memory/MCP/write integration comes later.
    disabled = set(disabled_tools or set())
    unsafe_requested = any(
        name not in disabled
        for name in {
            "write_file",
            "edit_file",
            "bash",
            "python",
            "create_document",
            "update_document",
            "edit_document",
            "manage_atlas",
            "manage_memory",
            "send_email",
        }
    )

    if active_document is not None or active_email is not None:
        yield _legacy_frame({
            "delta": (
                "LangGraph runtime v0.8 is currently limited to read-only file inspection. "
                "This request includes active document/email context, so it should use the legacy agent loop for now."
            )
        })
        yield _legacy_frame(_metrics_payload(
            model=model,
            requested_model=requested_model,
            started_at=started_at,
            agent_rounds=agent_rounds,
            agent_tool_calls=agent_tool_calls,
            final_chars=0,
        ))
        yield "data: [DONE]\n\n"
        return

    # The runtime itself currently accepts question/workspace through run_stream.
    # If workspace is None, run_stream falls back to its default configured workspace.
    async for _frame in _run_stream_as_legacy_frames(question, workspace=workspace):
        if _frame.startswith("data: ") and not _frame.startswith("data: [DONE]"):
            try:
                payload = json.loads(_frame[6:])
                if payload.get("type") == "agent_step":
                    data = payload.get("data") or {}
                    round_value = data.get("tool_rounds") or data.get("round")
                    if isinstance(round_value, int):
                        agent_rounds = max(agent_rounds, round_value)
                elif payload.get("type") == "tool_start":
                    agent_tool_calls += 1
                elif "delta" in payload:
                    final_answer += payload.get("delta") or ""
            except Exception:
                pass

        # Intercept DONE so we can emit metrics first, matching chat_routes.py expectations.
        if _frame == "data: [DONE]\n\n":
            yield _legacy_frame(_metrics_payload(
                model=model,
                requested_model=requested_model,
                started_at=started_at,
                agent_rounds=agent_rounds,
                agent_tool_calls=agent_tool_calls,
                final_chars=len(final_answer),
            ))
            yield _frame
        else:
            yield _frame


async def _run_stream_as_legacy_frames(
    question: str,
    *,
    workspace: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Async facade over run_stream(), which currently yields from a thread.

    run_stream itself is synchronous because it bridges LangGraph execution into
    a queue. This async wrapper keeps the public integration shape compatible
    with FastAPI StreamingResponse and stream_agent_loop.
    """

    # Import here so future workspace-aware runtime changes stay localized.
    from src.agent_runtime.streaming import run_stream as _run_stream

    for event in _run_stream(question, workspace=workspace):
        frame = to_legacy_sse_frame(event)
        if frame is not None:
            yield frame
