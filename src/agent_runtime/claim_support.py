from __future__ import annotations

import re
from typing import Any

from src.agent_runtime.exact_terms import missing_exact_terms

from src.agent_runtime.evidence_ledger import rank_observations


STOP_WORDS = {
    "the", "and", "for", "where", "find", "show", "main", "project",
    "file", "code", "use", "tools", "needed", "guess", "does", "this",
    "that", "with", "from", "into", "current", "after", "before",
    "around", "implemented", "implementation",
}


def _norm(text: str | None) -> str:
    return str(text or "").lower().replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")


def _tokens(text: str | None) -> set[str]:
    raw = _norm(text)
    return {
        t for t in re.findall(r'[a-zA-Z][a-zA-Z0-9]{2,}', raw)
        if t not in STOP_WORDS
    }


def _contains_term(text: str, term: str) -> bool:
    normalized = _norm(term)
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])",
            text,
        )
    )


def _code_like_terms(query: str | None) -> set[str]:
    raw = str(query or "")
    terms = set()

    for match in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', raw):
        if "_" in match and len(match) >= 4:
            terms.add(match.lower())

    for match in re.findall(r'[A-Za-z0-9_./\\-]+\.(?:py|js|jsx|ts|tsx|css|html|json|md)', raw):
        terms.add(match.lower().replace("/", "\\"))

    return terms


def _required_semantic_groups(query: str | None) -> list[tuple[str, set[str]]]:
    q = _norm(query)
    groups: list[tuple[str, set[str]]] = []

    if "tool calls" in q or "tool call" in q:
        groups.append(("tool-call concept", {"tool call", "tool_calls", "tool block", "tool_blocks", "toolmessage", "tool message"}))

    if any(_contains_term(q, word) for word in ["save", "saves", "saving", "saved"]):
        groups.append(("save action", {"save", "saves", "saving", "saved", "save_assistant_response", "commit", "message_saved"}))

    if "assistant response" in q or "assistant" in q and "response" in q:
        groups.append(("assistant response", {"assistant", "response", "full_response", "reply"}))

    if _requires_execution_call_site(query):
        groups.append(("execution action", {"execute", "executes", "executing", "execution", "invoke", "invoke(", ".invoke(", "await", "tool_obj", "execute_tool_block", "call"}))

    if any(
        phrase in q
        for phrase in [
            "handling branch",
            "event handler",
            "handler branch",
            "switch branch",
        ]
    ):
        groups.append(("handling branch", {"handle", "handler", "if", "elif", "case", "switch", "event", "type", "json type"}))

    if any(_contains_term(q, word) for word in ["stream", "streaming", "streamed"]):
        groups.append(("streaming context", {"stream", "streaming", "chunk", "delta", "[done]", "data: [done]", "yield", "stream_with_save"}))

    return groups


def _requires_execution_call_site(query: str | None) -> bool:
    """Recognize an explicit call-site obligation, not a generic code lookup."""

    q = _norm(query)
    if any(
        phrase in q
        for phrase in [
            "call site",
            "callsite",
            "execution flow",
            "execution path",
            "invocation site",
            "where it is invoked",
            "where this is invoked",
            "where it is called",
            "where this is called",
        ]
    ):
        return True
    return bool(
        re.search(
            r"\bhow\b.{0,80}\b(?:execute|executes|run|runs)\b",
            q,
        )
        or re.search(
            r"\bwhere\b.{0,80}\b(?:execute|executes|executed|called|invoked)\b",
            q,
        )
    )


def _path_obligation_ok(path: str | None, query: str | None) -> tuple[bool, str]:
    p = str(path or "").lower().replace("/", "\\")
    q = _norm(query)

    if any(_contains_term(q, word) for word in ["frontend", "front end", "client", "ui", "browser"]):
        if p.startswith("static\\") or "\\static\\" in p or p.endswith((".js", ".jsx", ".ts", ".tsx", ".css", ".html")):
            return True, "frontend/client source path matches query"
        return False, "query asks for frontend/client handling but best evidence is not in frontend/client source"

    if "chat routes" in q or "chat_routes" in q:
        if "routes\\chat_routes.py" in p:
            return True, "chat_routes path matches query"
        return False, "query asks for chat_routes but best evidence is not routes\\chat_routes.py"

    if "agent loop" in q:
        if "src\\agent_loop.py" in p:
            return True, "agent_loop path matches query"
        return False, "query asks for agent loop but best evidence is not src\\agent_loop.py"

    return True, "no path-specific obligation"


def _context_text(obs: dict[str, Any]) -> str:
    return _norm(
        " ".join(
            [
                str(obs.get("source_path", "")),
                str(obs.get("content", "")),
                str((obs.get("metadata") or {}).get("raw", "")),
            ]
        )
    )


def _is_probably_prompt_or_header(obs: dict[str, Any]) -> bool:
    content = str(obs.get("content") or "")
    source_text = re.sub(
        r"(?m)^[^\r\n]*?:\d+:\s?",
        "",
        content,
    )
    text = _norm(source_text)
    has_implementation = bool(
        re.search(
            r"\b(?:async\s+def|def|class)\s+[a-z_][a-z0-9_]*",
            source_text,
            flags=re.IGNORECASE,
        )
        and (
            "await " in source_text
            or re.search(r"\breturn\b", source_text)
            or re.search(r"\b(?:if|for|while|try)\b", source_text)
        )
    )

    # Import-only windows remain headers regardless of their line number.
    if (
        not has_implementation
        and (
            re.search(
                r"(?m)^\s*(?:\d+:\s*)?(?:from\s+\S+\s+import|import\s+\S+)",
                source_text,
            )
            or "streamingresponse" in text
        )
    ):
        return True

    # Large system prompt/rules blocks are usually instructions about behavior,
    # not the implementation site of that behavior.
    if (
        not has_implementation
        and (
            "_agent_rules" in text
            or "you declare when the job is done" in text
            or "after a tool succeeds" in text
            or (
                re.search(r'(?:"""|\'\'\')', source_text)
                and re.search(r"\b(?:prompt|rules|instructions)\b", text)
            )
        )
    ):
        return True

    return False


def is_implementation_source_window(obs: dict[str, Any]) -> bool:
    return bool(
        obs.get("source_authority") == "source_code"
        and obs.get("kind") == "context_window"
        and not _is_probably_prompt_or_header(obs)
    )



def _context_has_concrete_call_site(obs: dict[str, Any], query: str | None) -> bool:
    text = str(obs.get("content") or "")
    lowered = text.lower()
    q = _norm(query)

    if _requires_execution_call_site(query):
        patterns = [
            r'return\s+await\s+execute_tool_block\s*\(',
            r'await\s+execute_tool_block\s*\(',
            r'execute_tool_block\s*\(',
            r'\.invoke\s*\(',
            r'tool_obj\.invoke\s*\(',
            r'\b(?:return\s+)?await\s+[A-Za-z_][A-Za-z0-9_\.]*\s*\(',
        ]
        return any(re.search(p, text) for p in patterns)

    if any(_contains_term(q, word) for word in ["save", "saves", "saving", "saved"]):
        return bool(
            re.search(r'save_assistant_response\s*\(', text)
            and any(term in lowered for term in ["full_response", "message_saved", "data: [done]", "[done]", "stream"])
        )

    if any(_contains_term(q, word) for word in ["handle", "handles", "handling", "handler"]):
        return bool(
            "tool_output" in lowered
            and any(term in lowered for term in ["json.type", "else if", "elif", "if (", "handler", "event"])
        )

    return True


def _obligation_report(obs: dict[str, Any], query: str) -> dict[str, Any]:
    text = _context_text(obs)
    q_tokens = _tokens(query)
    code_terms = _code_like_terms(query)
    semantic_groups = _required_semantic_groups(query)

    matched_query_terms = sorted(t for t in q_tokens if t in text)
    missing_code_terms = missing_exact_terms(obs.get("content", ""), query)

    group_results = []
    for label, group in semantic_groups:
        ok = any(term in text for term in group)
        group_results.append(
            {
                "label": label,
                "ok": ok,
                "expected_any": sorted(group),
            }
        )

    path_ok, path_reason = _path_obligation_ok(obs.get("source_path"), query)
    header_or_prompt = _is_probably_prompt_or_header(obs)

    q_norm = _norm(query)
    done_marker_required = "[done]" in q_norm or "data: [done]" in q_norm
    done_marker_ok = True
    if done_marker_required:
        done_marker_ok = "[done]" in text or "data: [done]" in text

    enough_terms = len(matched_query_terms) >= 2 or bool(code_terms and not missing_code_terms)
    code_terms_ok = not missing_code_terms
    groups_ok = all(item["ok"] for item in group_results)

    concrete_call_site_ok = _context_has_concrete_call_site(obs, query)

    ok = bool(path_ok and code_terms_ok and groups_ok and enough_terms and not header_or_prompt and concrete_call_site_ok and done_marker_ok)

    reasons = []
    if not path_ok:
        reasons.append(path_reason)
    if missing_code_terms:
        reasons.append("missing exact code/event/function terms: " + ", ".join(missing_code_terms))
    if not groups_ok:
        failed = [g["label"] for g in group_results if not g["ok"]]
        reasons.append("missing required semantic group(s): " + ", ".join(failed))
    if not enough_terms:
        reasons.append("insufficient query-term coverage in expanded context")
    if header_or_prompt:
        reasons.append("context appears to be header/import/prompt-rules, not implementation evidence")
    if not concrete_call_site_ok:
        if _requires_execution_call_site(query):
            reasons.append(
                "missing execution call site explicitly required by the query"
            )
        else:
            reasons.append(
                "expanded context does not contain the required save/handler site"
            )
    if not done_marker_ok:
        reasons.append("query asks for data: [DONE] / [DONE], but expanded context does not contain that exact marker")

    return {
        "ok": ok,
        "path_ok": path_ok,
        "path_reason": path_reason,
        "matched_query_terms": matched_query_terms,
        "missing_code_terms": missing_code_terms,
        "group_results": group_results,
        "header_or_prompt": header_or_prompt,
        "concrete_call_site_ok": concrete_call_site_ok,
        "done_marker_ok": done_marker_ok,
        "reason": "; ".join(reasons) or "expanded context satisfies query obligations",
    }


def evaluate_code_location_support(
    *,
    query: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = rank_observations(observations)

    explicit_file_targets = {
        value.lower().replace("/", "\\")
        for value in re.findall(
            r"[A-Za-z0-9_.\\/-]+\.(?:py|js|jsx|ts|tsx|css|html|json|"
            r"md|txt|toml|yaml|yml)",
            str(query or ""),
            flags=re.IGNORECASE,
        )
    }
    direct_document_reads = [
        observation
        for observation in ranked
        if observation.get("kind") == "context_window"
        and observation.get("source_tool") == "read_file"
        and observation.get("source_authority") == "document"
        and str(observation.get("content") or "").strip()
        and any(
            str(observation.get("source_path") or "").lower().endswith(
                target
            )
            for target in explicit_file_targets
        )
    ]
    if direct_document_reads:
        best_document = direct_document_reads[0]
        return {
            "claim_type": "file_contents",
            "support_level": "strong",
            "supported_by": [best_document.get("observation_id")],
            "reason": (
                "An exact requested workspace document was read directly."
            ),
            "best_observation": best_document,
            "obligation_report": {
                "ok": True,
                "reason": "exact requested file contents are available",
            },
        }

    source_contexts = [
        o for o in ranked
        if o.get("source_authority") == "source_code"
        and o.get("kind") == "context_window"
    ]

    source_lines = [
        o for o in ranked
        if o.get("source_authority") == "source_code"
        and o.get("kind") == "code_line"
    ]

    passing_contexts = []
    failing_contexts = []

    for obs in source_contexts:
        report = _obligation_report(obs, query)
        decorated = dict(obs)
        decorated["obligation_report"] = report
        if report["ok"]:
            passing_contexts.append(decorated)
        else:
            failing_contexts.append(decorated)

    if passing_contexts:
        best_context = sorted(
            passing_contexts,
            key=lambda o: (
                -int(o.get("confidence", 0)),
                -int(o.get("query_relevance", 0)),
                -int(o.get("call_site_score", 0)),
                str(o.get("source_path", "")),
                int(o.get("line_start") or 0),
            ),
        )[0]

        related_lines = [
            o for o in source_lines
            if o.get("source_path") == best_context.get("source_path")
            and int(best_context.get("line_start") or 0) <= int(o.get("line_start") or 0) <= int(best_context.get("line_end") or 0)
        ][:6]

        supported = [best_context.get("observation_id")]
        supported.extend(o.get("observation_id") for o in related_lines if o.get("observation_id"))

        return {
            "claim_type": "code_location",
            "support_level": "strong",
            "supported_by": supported,
            "reason": (
                "A source-code context window satisfies the query obligations: "
                + best_context["obligation_report"]["reason"]
            ),
            "best_observation": best_context,
            "obligation_report": best_context["obligation_report"],
        }

    relevant_lines = [
        o for o in source_lines
        if int(o.get("query_relevance", 0)) >= 3
        and not o.get("weak_line")
    ]

    if relevant_lines:
        best_line = relevant_lines[0]
        reason = (
            "A query-relevant source-code line was observed, but no expanded context "
            "has satisfied the required target/action/path obligations yet."
        )
        if failing_contexts:
            reason += " Best expanded context failed because: " + failing_contexts[0]["obligation_report"]["reason"]

        return {
            "claim_type": "code_location",
            "support_level": "partial",
            "supported_by": [best_line.get("observation_id")],
            "reason": reason,
            "best_observation": best_line,
            "obligation_report": failing_contexts[0]["obligation_report"] if failing_contexts else {},
        }

    if failing_contexts:
        best_context = sorted(
            failing_contexts,
            key=lambda o: (
                -int(o.get("confidence", 0)),
                -int(o.get("query_relevance", 0)),
                str(o.get("source_path", "")),
                int(o.get("line_start") or 0),
            ),
        )[0]
        report = best_context["obligation_report"]
        return {
            "claim_type": "code_location",
            "support_level": "insufficient",
            "supported_by": [],
            "reason": "Best source window was rejected because: " + report["reason"],
            "best_observation": best_context,
            "obligation_report": report,
        }

    return {
        "claim_type": "code_location",
        "support_level": "insufficient",
        "supported_by": [],
        "reason": "No query-relevant source-code observation is currently available.",
        "best_observation": None,
        "obligation_report": {},
    }


def has_strong_claim_support(claim_support: dict[str, Any] | None) -> bool:
    return bool((claim_support or {}).get("support_level") == "strong")


def format_claim_support(claim_support: dict[str, Any] | None) -> str:
    cs = claim_support or {}
    best = cs.get("best_observation") or {}
    report = cs.get("obligation_report") or {}

    rows = [
        f"Claim support: {cs.get('support_level', 'unknown')}",
        f"Claim type: {cs.get('claim_type', 'unknown')}",
        f"Reason: {cs.get('reason', '')}",
        f"Supported by: {', '.join(cs.get('supported_by') or []) or '(none)'}",
    ]

    if report:
        rows.append(f"Obligation check: {report.get('reason', '')}")
        if report.get("matched_query_terms"):
            rows.append("Matched query terms: " + ", ".join(report.get("matched_query_terms") or []))
        if report.get("missing_code_terms"):
            rows.append("Missing code terms: " + ", ".join(report.get("missing_code_terms") or []))
        if report.get("group_results"):
            for group in report.get("group_results") or []:
                rows.append(f"Group {group.get('label')}: {'ok' if group.get('ok') else 'missing'}")

    if best:
        loc = best.get("source_path") or "(unknown)"
        if best.get("line_start"):
            if best.get("line_end") and best.get("line_end") != best.get("line_start"):
                loc += f":{best.get('line_start')}-{best.get('line_end')}"
            else:
                loc += f":{best.get('line_start')}"
        rows.append(f"Best observation: {best.get('observation_id')} {loc}")
        rows.append(f"Best observation content: {str(best.get('content', '')).strip()[:700]}")

    return "\n".join(rows)
