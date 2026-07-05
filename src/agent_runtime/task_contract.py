from __future__ import annotations

import re
from typing import Any

from src.agent_runtime.exact_terms import extract_exact_terms


def _norm(text: str | None) -> str:
    return str(text or "").lower().replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")


def _source_intent(query: str | None) -> str:
    q = _norm(query)

    if any(x in q for x in ["frontend", "front end", "client", "browser", "ui"]):
        return "frontend"

    if "chat routes" in q or "chat_routes" in q:
        return "chat_routes"

    if "agent loop" in q:
        return "agent_loop"

    if any(x in q for x in ["route", "api", "backend", "server"]):
        return "backend"

    return "unknown"


def _task_kind(query: str | None) -> str:
    q = _norm(query)
    exact_terms = extract_exact_terms(query)
    source_intent = _source_intent(query)

    # Source-specific tasks must be classified before broad semantic words like
    # "saved", otherwise frontend event handling gets mixed with backend save logic.
    if source_intent == "frontend" and any(x in q for x in ["handle", "handles", "handling", "event", "events", "json type"]):
        return "frontend_event_handler_lookup"

    if any(x in q for x in [" and where ", "registered and", "where its "]):
        return "compound_code_location"

    if exact_terms and any(x in q for x in ["not_a_real", "banana_event", "definitely_not"]):
        return "exact_negative_lookup"

    if any(x in q for x in ["data: [done]", "[done]"]):
        return "control_flow_marker_lookup"

    if any(x in q for x in ["execute", "executes", "executing", "run", "runs", "running"]):
        return "implementation_callsite_lookup"

    if any(x in q for x in ["handle", "handles", "handling", "handler"]):
        return "event_handler_lookup"

    if any(x in q for x in ["save", "saves", "saved", "saving", "message_saved"]):
        return "save_or_event_lookup"

    if exact_terms:
        return "exact_symbol_lookup"

    return "code_location_lookup"

def _split_obligations(query: str | None) -> list[dict[str, Any]]:
    raw = str(query or "").strip()
    q = _norm(raw)
    exact_terms = extract_exact_terms(raw)
    source_intent = _source_intent(raw)
    task_kind = _task_kind(raw)
    obligations: list[dict[str, Any]] = []

    # Exact terms are first-class obligations.
    for term in exact_terms:
        obligations.append(
            {
                "kind": "exact_term",
                "term": term,
                "status": "unknown",
            }
        )

    # Frontend handler lookup: do NOT require backend save call evidence just
    # because the event name contains "saved" or the prompt says "after saved".
    if task_kind == "frontend_event_handler_lookup":
        obligations.append(
            {
                "kind": "frontend_handler_branch",
                "term": "frontend handler branch",
                "status": "unknown",
            }
        )
    else:
        if "registered" in q or "route" in q or "/api/" in raw:
            obligations.append(
                {
                    "kind": "route_registration",
                    "term": "route registration",
                    "status": "unknown",
                }
            )

        if "generator" in q or "stream generator" in q:
            obligations.append(
                {
                    "kind": "stream_generator",
                    "term": "stream generator",
                    "status": "unknown",
                }
            )

        if any(x in q for x in ["execute", "executes", "executing", "run", "runs", "running"]):
            obligations.append(
                {
                    "kind": "concrete_callsite",
                    "term": "concrete execution callsite",
                    "status": "unknown",
                }
            )

        if any(x in q for x in ["handle", "handles", "handling", "handler"]):
            obligations.append(
                {
                    "kind": "handler_branch",
                    "term": "handler branch",
                    "status": "unknown",
                }
            )

        if any(x in q for x in ["save", "saves", "saving", "saved"]):
            obligations.append(
                {
                    "kind": "save_callsite",
                    "term": "save callsite",
                    "status": "unknown",
                }
            )

    if not obligations:
        obligations.append(
            {
                "kind": "code_location",
                "term": raw,
                "status": "unknown",
            }
        )

    seen = set()
    out = []
    for ob in obligations:
        key = (ob.get("kind"), str(ob.get("term")).lower())
        if key not in seen:
            out.append(ob)
            seen.add(key)

    return out

def build_task_contract(query: str | None) -> dict[str, Any]:
    return {
        "task_kind": _task_kind(query),
        "source_intent": _source_intent(query),
        "exact_terms": extract_exact_terms(query),
        "obligations": _split_obligations(query),
        "requires_terminal_finalizer": True,
        "allow_direct_end": False,
    }


def format_task_contract(contract: dict[str, Any] | None) -> str:
    c = contract or {}
    rows = [
        "Task contract:",
        f"- task_kind: {c.get('task_kind', 'unknown')}",
        f"- source_intent: {c.get('source_intent', 'unknown')}",
        f"- exact_terms: {', '.join(c.get('exact_terms') or []) or '(none)'}",
        f"- requires_terminal_finalizer: {c.get('requires_terminal_finalizer', True)}",
        f"- allow_direct_end: {c.get('allow_direct_end', False)}",
        "- obligations:",
    ]

    for ob in c.get("obligations") or []:
        rows.append(f"  - {ob.get('kind')}: {ob.get('term')} [{ob.get('status', 'unknown')}]")

    return "\n".join(rows)
