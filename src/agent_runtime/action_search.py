from __future__ import annotations

import re

from src.agent_runtime.exact_terms import extract_exact_terms
from typing import Any

from src.agent_runtime.evidence_ledger import rank_observations


def _norm(text: str | None) -> str:
    return str(text or "").lower().replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")


def search_assist_needed(query: str | None, claim_support: dict[str, Any] | None) -> bool:
    support = (claim_support or {}).get("support_level")
    if support == "strong":
        return False

    q = _norm(query)
    action_words = [
        "execute", "executes", "executing", "execution",
        "save", "saves", "saving", "saved",
        "handle", "handles", "handling", "handler",
        "block", "blocks", "blocking",
        "configure", "config", "setting",
        "stream", "streaming",
    ]

    return any(word in q for word in action_words)


def _best_source_path(query: str | None, observations: list[dict[str, Any]] | None) -> str:
    q = _norm(query)

    if "agent loop" in q:
        return "src\\agent_loop.py"

    if "chat_routes" in q or "chat routes" in q or "/api/" in str(query or "").lower() or "chat stream" in q or "chat_stream" in q:
        return "routes\\chat_routes.py"

    if any(word in q for word in ["frontend", "front end", "client", "ui", "browser"]):
        return "static\\js"

    ranked = rank_observations(observations or [])
    for obs in ranked:
        path = str(obs.get("source_path") or "")
        norm_path = path.replace("/", "\\")
        if norm_path.startswith("src\\agent_runtime\\"):
            # Runtime internals should not become the search target unless
            # the user specifically asked about the runtime.
            if not any(x in q for x in ["runtime", "langgraph", "claim support", "evidence ledger", "task contract", "finalizer"]):
                continue
        if path and obs.get("source_authority") == "source_code":
            return path

    return "."


def assist_patterns(query: str | None) -> list[str]:
    q = _norm(query)
    patterns: list[str] = []

    # Exact user-specified targets first. This prevents prompts like
    # "data: [DONE]" from drifting into generic json.type/agent_step handlers.
    for term in extract_exact_terms(query):
        if term.lower().startswith("data:") and "[done]" in term.lower():
            patterns.extend([r"data:\s*\[DONE\]", r"\[DONE\]", r"data:\s*\[done\]"])
        elif term.startswith("/"):
            patterns.append(re.escape(term))
        else:
            patterns.append(re.escape(term))

    if "tool call" in q or "tool calls" in q or "execute" in q or "executes" in q or "executing" in q:
        patterns.extend(
            [
                r"return\s+await\s+execute_tool_block",
                r"await\s+execute_tool_block",
                r"execute_tool_block\s*\(",
                r"\.invoke\(",
                r"tool_obj\.invoke",
                r"tool_blocks?",
                r"tool_calls?",
                r"ToolMessage",
            ]
        )

    if "save" in q or "saves" in q or "saving" in q or "saved" in q:
        patterns.extend(
            [
                r"save_assistant_response",
                r"message_saved",
                r"full_response",
                r"data:\s*\[DONE\]",
                r"\[DONE\]",
                r"stream_with_save",
                r"commit\(",
            ]
        )

    if "handle" in q or "handles" in q or "handling" in q or "handler" in q:
        patterns.extend(
            [
                r"json\.type",
                r"type.*==",
                r"elif.*type",
                r"if.*type",
                r"case\s+",
                r"handler",
            ]
        )

    if "stream" in q or "streaming" in q or "generator" in q:
        patterns.extend(
            [
                r"@router\.post\(.*chat_stream",
                r"async\s+def\s+chat_stream",
                r"StreamingResponse",
                r"yield\s+f?['\"]data:",
                r"yield\s+['\"]data:",
                r"data:\s*\[DONE\]",
                r"\[DONE\]",
            ]
        )

    # De-duplicate while preserving order.
    seen = set()
    out = []
    for p in patterns:
        if p not in seen:
            out.append(p)
            seen.add(p)

    return out[:10]

def build_assist_searches(
    *,
    query: str | None,
    observations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    path = _best_source_path(query, observations)
    searches = []

    for pattern in assist_patterns(query):
        searches.append(
            {
                "tool": "grep",
                "args": {
                    "pattern": pattern,
                    "path": path,
                    "max_matches": 30,
                },
                "reason": "targeted_action_search",
            }
        )

    return searches[:5]
