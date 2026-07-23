from __future__ import annotations

from typing import Any


BRIDGE_VERSION = "knowledge_llm_response_bridge_v1"

SYSTEM_PROMPT = '''You are the LangGraph knowledge-runtime response writer for Odysseus.

The runtime, not you, owns permissions and tool safety.

You must:
- answer only from the runtime evidence/context provided
- respect blocked/allowed graph decisions
- never claim you edited, deleted, renamed, moved, or wrote a source note
- never invent vault contents
- be natural and helpful
- when evidence is missing, say what is missing and ask for the smallest useful clarification
- when the runtime blocked an action, explain the block naturally and offer safe alternatives

Return only the final user-facing assistant message.
'''


def _short(value: Any, limit: int = 5000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def knowledge_llm_config_from_runtime_locals(scope: dict[str, Any]) -> dict[str, Any]:
    def pick_attr(obj: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            try:
                value = getattr(obj, name, None)
            except Exception:
                value = None
            if value:
                return value
        return None

    url = ""
    model = ""
    headers = None

    for name in ("endpoint_url", "chat_url", "model_url", "url"):
        if scope.get(name):
            url = str(scope.get(name) or "").strip()
            break

    for name in ("model", "selected_model", "model_name"):
        if scope.get(name):
            model = str(scope.get(name) or "").strip()
            break

    for name in ("headers", "auth_headers", "request_headers"):
        if isinstance(scope.get(name), dict):
            headers = scope.get(name)
            break

    ctx = scope.get("ctx") or scope.get("chat_context")
    if ctx is not None:
        url = url or str(pick_attr(ctx, ("endpoint_url", "chat_url", "url", "model_url")) or "").strip()
        model = model or str(pick_attr(ctx, ("model", "selected_model", "model_name")) or "").strip()
        headers = headers or pick_attr(ctx, ("headers", "auth_headers"))

    sess = scope.get("sess") or scope.get("session")
    if sess is not None:
        model = model or str(pick_attr(sess, ("model", "selected_model", "model_name")) or "").strip()
        headers = headers or pick_attr(sess, ("headers", "auth_headers"))

    candidates = (
        scope.get("candidates")
        or scope.get("fallback_candidates")
        or scope.get("fallbacks")
        or scope.get("_fallback_candidates")
    )
    if not url and not model and isinstance(candidates, (list, tuple)) and candidates:
        first = candidates[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            url = str(first[0] or "").strip()
            model = str(first[1] or "").strip()
            if len(first) >= 3 and isinstance(first[2], dict):
                headers = first[2]

    return {
        "url": url,
        "model": model,
        "headers": headers if isinstance(headers, dict) else None,
        "fallback_candidates": (
            list(candidates)
            if isinstance(candidates, (list, tuple))
            else []
        ),
    }


def build_knowledge_response_messages(
    *,
    user_message: str,
    deterministic_answer: str,
    runtime_decision: dict[str, Any],
    evidence_summary: dict[str, Any],
    hitl_policy: dict[str, Any],
    legacy_fallback_contract: dict[str, Any],
) -> list[dict[str, str]]:
    context = {
        "runtime_decision": runtime_decision,
        "evidence_summary": evidence_summary,
        "hitl_policy": hitl_policy,
        "legacy_fallback_contract": legacy_fallback_contract,
        "deterministic_runtime_answer": _short(deterministic_answer, 7000),
    }

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "User request:\n"
                f"{_short(user_message, 2000)}\n\n"
                "Runtime context and evidence:\n"
                f"{_short(context, 12000)}\n\n"
                "Write a natural final answer. Do not expose raw JSON. "
                "Do not claim actions that runtime did not allow."
            ),
        },
    ]


def try_knowledge_llm_response(
    *,
    llm_config: dict[str, Any] | None,
    user_message: str,
    deterministic_answer: str,
    runtime_decision: dict[str, Any],
    evidence_summary: dict[str, Any],
    hitl_policy: dict[str, Any],
    legacy_fallback_contract: dict[str, Any],
    llm_call_func: Any | None = None,
) -> dict[str, Any]:
    config = llm_config or {}
    url = str(config.get("url") or "").strip()
    model = str(config.get("model") or "").strip()
    headers = config.get("headers") if isinstance(config.get("headers"), dict) else None

    result = {
        "version": BRIDGE_VERSION,
        "used_llm": False,
        "fallback_used": True,
        "status": "skipped",
        "url_present": bool(url),
        "model": model,
        "final_answer": deterministic_answer,
        "error": "",
    }

    if not url or not model:
        result["status"] = "skipped_missing_selected_model"
        return result

    messages = build_knowledge_response_messages(
        user_message=user_message,
        deterministic_answer=deterministic_answer,
        runtime_decision=runtime_decision,
        evidence_summary=evidence_summary,
        hitl_policy=hitl_policy,
        legacy_fallback_contract=legacy_fallback_contract,
    )

    try:
        if llm_call_func is None:
            from src.llm_core import llm_call_with_fallback

            runtime_llm_candidates = [
                (url, model, headers),
                *list(config.get("fallback_candidates") or ()),
            ]

            response = llm_call_with_fallback(
                runtime_llm_candidates,
                messages=messages,
                temperature=0.2,
                max_tokens=1200,
            )
        else:
            response = llm_call_func(
                url,
                model,
                messages,
                temperature=0.2,
                max_tokens=1200,
                headers=headers,
            )
        text = str(response or "").strip()
        if not text:
            result["status"] = "empty_llm_response"
            return result

        result.update({
            "used_llm": True,
            "fallback_used": False,
            "status": "ok",
            "final_answer": text,
        })
        return result
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        result.update({
            "status": "llm_error",
                "error_type": error_type,
                "error_message": error_message[:600],
            "error": f"{type(exc).__name__}: {exc}",
        })
        return result


__all__ = [
    "BRIDGE_VERSION",
    "build_knowledge_response_messages",
    "knowledge_llm_config_from_runtime_locals",
    "try_knowledge_llm_response",
]


def enforce_knowledge_llm_final_answer(
    *,
    final_answer: str,
    llm_response: dict[str, Any],
    llm_config: dict[str, Any] | None,
) -> str:
    """Ensure the user-facing answer is honest about LLM synthesis.

    Raw read/search output may exist as tool evidence, but the final assistant
    answer should be LLM-authored when the LLM bridge succeeds. If the bridge is
    skipped or errors, return a clear diagnostic final answer instead of silently
    presenting deterministic tool output as the assistant's final response.
    """

    candidate = str(llm_response.get("final_answer") or "").strip()
    used_llm = bool(llm_response.get("used_llm"))
    status = str(llm_response.get("status") or "unknown")
    model = str(llm_response.get("model") or "")

    if used_llm and candidate:
        return candidate

    config_present = bool(llm_config and (llm_config.get("model") or llm_config.get("url")))

    if config_present:
        model_part = f" using model {model}" if model else ""
        return (
            "I read the requested knowledge source, but the LLM finalizer did not "
            f"produce the final assistant answer{model_part}. "
            f"Bridge status: {status}. The raw tool result is available above as evidence."
        )

    return (
        "I read the requested knowledge source, but the selected model configuration "
        "was not available to the LLM finalizer. "
        f"Bridge status: {status}. The raw tool result is available above as evidence."
    )

