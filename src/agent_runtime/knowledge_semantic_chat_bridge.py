
from __future__ import annotations

import json
import re
import time
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any

from .knowledge_graph_router import (
    build_human_clarification_prompt,
    evaluate_knowledge_request,
)
from .runtime_router import (
    is_casual_conversation_request,
    route_agent_runtime,
)
from .knowledge_vault_semantic import (
    get_semantic_vault_stats,
    search_semantic_vault_index,
)
from .daily_driver_preface import generate_intent_preface
from .knowledge_worker import run_knowledge_worker
from .knowledge_llm_response_bridge import try_knowledge_llm_response
from .supervisor_contracts import WorkerHandoff


def sse_frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def is_semantic_vault_search_request(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False

    # Do not steal exact/active/path reads from the existing read graph.
    exact_read_markers = [
        "read the active note",
        "active note",
        "list vault folders",
        "list my vault folders",
        "vault folders",
        "read note ",
        "read file ",
        "read path ",
    ]
    if any(marker in text for marker in exact_read_markers):
        return False

    semantic_markers = [
        "search my notes",
        "search the notes",
        "search notes",
        "search my vault",
        "find my notes",
        "find notes",
        "notes about",
        "note about",
        "what did i write about",
        "what have i written about",
        "what do my notes say about",
        "in my notes",
        "from my notes",
        "within my notes",
        "based on my notes",
        "ideaverse about",
        "knowledge about",
    ]

    return any(marker in text for marker in semantic_markers)


def should_intercept_workspace_knowledge(message: str) -> bool:
    """Treat an active workspace as authoritative context for safe questions.

    The workspace is request-scoped context, so users should not have to repeat
    "in my notes" on every follow-up. Clear code inspection, current-world
    research, and short social turns remain with their dedicated runtimes.
    """

    text = str(message or "").strip()
    if not text:
        return False

    decision = evaluate_knowledge_request(text)
    if decision.intent != "not_knowledge":
        return True

    if is_casual_conversation_request(text):
        return False

    runtime = route_agent_runtime(text)
    if runtime.target_runtime in {"contract", "general"}:
        return False

    low = text.lower()
    if re.search(
        r"\b(?:research|latest|news|web|online|browser|google|search online|look up online)\b",
        low,
    ):
        return False

    if re.search(
        r"\b(?:backend|frontend|source code|codebase|implementation|route|function|class|module|callsite|handler)\b|"
        r"(?:^|\s)(?:src|routes|static/js)/|\.(?:py|js|ts|tsx|jsx)\b",
        low,
    ):
        return False

    # An active workspace is available context, not evidence that every factual
    # question is about that workspace. Require the user to refer to the
    # workspace/project/files (or an equivalent contextual pronoun) before RAG
    # can intercept an otherwise ordinary question.
    return bool(re.search(
        r"\b(?:this|current|selected|active|my|our)\s+"
        r"(?:workspace|project|repository|repo|codebase|files?|documents?|notes?)\b"
        r"|\b(?:in|from|within|about)\s+(?:the|this|my|our)\s+"
        r"(?:workspace|project|repository|repo|codebase|files?|documents?|notes?)\b",
        low,
    ))


def extract_semantic_query(message: str) -> str:
    raw = str(message or "").strip()
    text = raw

    patterns = [
        r"(?i)^\s*search\s+(?:my\s+)?(?:vault|notes)\s+(?:for|about)\s+",
        r"(?i)^\s*find\s+(?:my\s+)?notes\s+(?:for|about|on|related to)\s+",
        r"(?i)^\s*what\s+(?:did|have)\s+i\s+(?:write|written)\s+about\s+",
        r"(?i)^\s*what\s+do\s+my\s+notes\s+say\s+about\s+",
        r"(?i)\s+in\s+my\s+notes\s*$",
        r"(?i)\s+from\s+my\s+notes\s*$",
        r"(?i)\s+based\s+on\s+my\s+notes\s*$",
    ]

    for pat in patterns:
        text = re.sub(pat, "", text).strip()

    text = text.strip(" .?!\"'")
    return text or raw


def _is_broad_workspace_planning_request(message: str) -> bool:
    """Identify requests that need coverage across several workspace facets."""

    low = str(message or "").strip().lower()
    if not low:
        return False

    planning_signal = re.search(
        r"\b(?:what should i do|what to do|do next|focus on|prioriti(?:y|es)|"
        r"most important|top\s+(?:\w+\s+)?things?|next actions?|action items?)\b",
        low,
    )
    if not planning_signal:
        return False

    # A planning word alone is not enough: a request with an explicit subject
    # should use normal topic retrieval. Whole-workspace planning is reserved
    # for genuinely subject-free requests such as "what should I do next?".
    generic_terms = {
        "a", "about", "active", "action", "actions", "and", "are", "based",
        "current", "do", "figure", "focus", "for", "from", "give", "help",
        "i", "important", "in", "is", "item", "items", "me", "most", "my",
        "next", "note", "notes", "of", "on", "out", "overall", "please",
        "priorities", "priority", "review", "should", "suggest", "tell", "the",
        "thing", "things", "this", "these", "three", "to", "top", "what",
        "workspace",
        # Presentation/style modifiers do not create a subject scope.
        "brief", "briefly", "clear", "clearly", "concise", "concisely",
        "language", "plain", "simple", "simply", "terms",
    }
    subject_terms = [
        token
        for token in re.findall(r"[a-z][a-z0-9_-]*", low)
        if token not in generic_terms
    ]
    return not subject_terms


def _planning_retrieval_queries(message: str) -> list[str]:
    """Expand broad planning into stable, domain-neutral retrieval facets."""

    if not _is_broad_workspace_planning_request(message):
        return []

    return [
        "deadlines due dates and time-sensitive obligations",
        "bills payments and administrative commitments",
        "unfinished tasks next actions and explicit priorities",
        "current goals important work and active projects",
    ]


def _build_currency_total_fact(
    query: str,
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compute a requested currency total from explicitly named evidence."""

    low = str(query or "").lower()
    if not re.search(r"\b(?:together|total|sum|combined|add(?:ed)?|altogether)\b", low):
        return None

    stop = {
        "what", "when", "where", "which", "much", "many", "are", "is",
        "the", "and", "or", "for", "from", "with", "bill", "bills",
        "payment", "payments", "amount", "amounts", "cost", "costs",
        "together", "total", "sum", "combined", "added", "altogether",
        "how", "does", "do", "my", "your", "their", "these", "those",
    }
    entities = [
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", low)
        if token not in stop
    ]
    if len(set(entities)) < 2:
        return None

    amount_pattern = re.compile(
        r"(?P<currency>[₹$€£])\s*(?P<amount>\d[\d,]*(?:\.\d{1,2})?)"
    )
    operands: dict[str, dict[str, Any]] = {}

    for result in results:
        preview = str(result.get("preview") or "")
        for line in preview.splitlines():
            line_low = line.lower()
            matched_entities = [entity for entity in entities if entity in line_low]
            if not matched_entities:
                continue
            amount_match = amount_pattern.search(line)
            if not amount_match:
                continue
            try:
                value = Decimal(amount_match.group("amount").replace(",", ""))
            except InvalidOperation:
                continue
            for entity in matched_entities:
                operands.setdefault(entity, {
                    "entity": entity,
                    "currency": amount_match.group("currency"),
                    "value": value,
                    "source_path": result.get("path"),
                    "source_line": line.strip(),
                })

    selected = [operands[entity] for entity in dict.fromkeys(entities) if entity in operands]
    if len(selected) < 2 or len({item["currency"] for item in selected}) != 1:
        return None

    total = sum((item["value"] for item in selected), Decimal("0"))
    plain = format(total, "f").rstrip("0").rstrip(".") if "." in format(total, "f") else format(total, "f")
    formatted = f"{total:,.2f}".rstrip("0").rstrip(".")
    return {
        "operation": "currency_sum",
        "currency": selected[0]["currency"],
        "operands": [
            {
                **item,
                "value": format(item["value"], "f"),
            }
            for item in selected
        ],
        "total_plain": plain,
        "total_formatted": formatted,
    }


def _build_planning_agenda(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract compact deadline and next-action facts from retrieved notes."""

    facts: list[tuple[int, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    date_pattern = re.compile(
        r"\b(?:\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
        r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
        re.IGNORECASE,
    )
    action_pattern = re.compile(
        r"\b(?:due|deadline|submit|return|pay|finish|complete|priority|"
        r"next step|next action|renew|prepare|schedule|send|back up|follow up)\b",
        re.IGNORECASE,
    )

    for result in results:
        path = str(result.get("path") or "")
        context = ""
        for raw_line in str(result.get("preview") or "").splitlines():
            line = raw_line.strip().strip("#").strip()
            if not line:
                continue
            clean = re.sub(r"^[-*\d.()\s]+", "", line).strip()
            if not clean:
                continue
            if clean.endswith(":") and len(clean) <= 80:
                context = clean[:-1].strip()
                continue
            has_date = bool(date_pattern.search(clean))
            has_action = bool(action_pattern.search(clean))
            if not (has_date or has_action):
                continue
            text = f"{context}: {clean}" if context else clean
            key = (path.casefold(), text.casefold())
            if key in seen:
                continue
            seen.add(key)
            score = 0
            if has_date:
                score += 6
            if re.search(r"\b(?:due|submit|return|pay|deadline)\b", clean, re.I):
                score += 5
            if re.search(r"\b(?:priority|next step|next action)\b", clean, re.I):
                score += 4
            if re.search(r"\b(?:finish|complete|prepare|renew|back up)\b", clean, re.I):
                score += 3
            facts.append((score, {
                "path": path,
                "fact": text,
                "dated": has_date,
            }))

    facts.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in facts[:24]]


def _select_balanced_planning_priorities(
    message: str,
    agenda: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select broad priorities without filling every slot from one category."""

    low = str(message or "").lower()
    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }
    requested_count = 0
    number_match = re.search(
        r"\b(one|two|three|four|five|[1-5])\b(?=\s+(?:most\s+)?important\b|\s+things?\b|\s+priorit)",
        low,
    )
    if number_match:
        token = number_match.group(1)
        requested_count = int(token) if token.isdigit() else word_numbers[token]
    if requested_count < 2:
        return []

    months = {
        name: index
        for index, names in enumerate((
            ("january", "jan"), ("february", "feb"), ("march", "mar"),
            ("april", "apr"), ("may",), ("june", "jun"),
            ("july", "jul"), ("august", "aug"),
            ("september", "sep"), ("october", "oct"),
            ("november", "nov"), ("december", "dec"),
        ), start=1)
        for name in names
    }

    def date_key(item: dict[str, Any]) -> tuple[int, int, int]:
        match = re.search(
            r"\b(\d{1,2})\s+([a-z]+)\s+(\d{4})\b",
            str(item.get("fact") or "").lower(),
        )
        if not match:
            return (9999, 12, 31)
        return (
            int(match.group(3)),
            months.get(match.group(2), 12),
            int(match.group(1)),
        )

    selected: list[dict[str, Any]] = []
    dated = [item for item in agenda if item.get("dated")]
    if dated:
        selected.append(min(dated, key=date_key))

    used_paths = {str(item.get("path") or "").casefold() for item in selected}
    action_candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(agenda):
        fact = str(item.get("fact") or "")
        if item.get("dated"):
            continue
        if re.search(r"\b(?:is|was|are|were)\s+complete\b", fact, re.I):
            continue
        if not re.search(
            r"\b(?:finish|complete|priority|next step|next action|submit|return|pay|prepare|back up)\b",
            fact,
            re.I,
        ):
            continue
        score = 0
        if re.search(r"\b(?:priority|next step|next action)\b", fact, re.I):
            score += 6
        if re.search(r"\b(?:finish|complete)\b", fact, re.I):
            score += 4
        if re.search(r"\b(?:submit|return|pay|prepare)\b", fact, re.I):
            score += 3
        if re.search(r"\bback up\b", fact, re.I):
            score += 1
        action_candidates.append((score, -index, item))

    action_candidates.sort(reverse=True, key=lambda row: (row[0], row[1]))
    for _score, _index, item in action_candidates:
        path_key = str(item.get("path") or "").casefold()
        if path_key in used_paths:
            continue
        selected.append(item)
        used_paths.add(path_key)
        if len(selected) >= requested_count:
            break

    return selected[:requested_count]


def _format_balanced_priority_answer(
    priorities: list[dict[str, Any]],
) -> str:
    lines = [
        "Based on the workspace's dated obligations and explicit next actions, "
        "these are the category-balanced priorities:",
        "",
    ]
    for index, item in enumerate(priorities, start=1):
        lines.append(f"{index}. **{item['fact']}**")
        if item.get("path"):
            lines.append(f"   Source: `{item['path']}`")
    lines.extend([
        "",
        "This ranking was produced read-only from the active workspace; no files were changed.",
    ])
    return "\n".join(lines)


def _format_comprehensive_planning_answer(
    agenda: list[dict[str, Any]],
) -> str:
    """Render broad planning coverage without model-selected omissions."""

    months = {
        name: index
        for index, names in enumerate((
            ("january", "jan"), ("february", "feb"), ("march", "mar"),
            ("april", "apr"), ("may",), ("june", "jun"),
            ("july", "jul"), ("august", "aug"),
            ("september", "sep"), ("october", "oct"),
            ("november", "nov"), ("december", "dec"),
        ), start=1)
        for name in names
    }

    def date_key(item: dict[str, Any]) -> tuple[int, int, int]:
        match = re.search(
            r"\b(\d{1,2})\s+([a-z]+)\s+(\d{4})\b",
            str(item.get("fact") or "").lower(),
        )
        if not match:
            return (9999, 12, 31)
        return (
            int(match.group(3)),
            months.get(match.group(2), 12),
            int(match.group(1)),
        )

    dated = sorted(
        (item for item in agenda if item.get("dated")),
        key=date_key,
    )[:8]
    action_rows: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(agenda):
        if item.get("dated"):
            continue
        fact = str(item.get("fact") or "")
        if re.search(r"\b(?:is|was|are|were)\s+complete\b", fact, re.I):
            continue
        if not re.search(
            r"\b(?:priority|next step|next action|finish|complete|prepare|back up)\b",
            fact,
            re.I,
        ):
            continue
        score = 0
        if re.search(r"\b(?:priority|next step|next action)\b", fact, re.I):
            score += 6
        if re.search(r"\b(?:finish|complete)\b", fact, re.I):
            score += 4
        if re.search(r"\b(?:prepare|back up)\b", fact, re.I):
            score += 2
        action_rows.append((score, -index, item))
    action_rows.sort(reverse=True, key=lambda row: (row[0], row[1]))

    actions: list[dict[str, Any]] = []
    used_paths: set[str] = set()
    for _score, _index, item in action_rows:
        path_key = str(item.get("path") or "").casefold()
        if path_key in used_paths:
            continue
        used_paths.add(path_key)
        actions.append(item)
        if len(actions) >= 5:
            break

    lines = ["Based on the active workspace, here is the read-only planning agenda:"]
    if dated:
        lines.extend(["", "Time-sensitive obligations:"])
        lines.extend(
            f"- {item['fact']} (`{item['path']}`)"
            for item in dated
        )
    if actions:
        lines.extend(["", "Explicit priorities and next actions:"])
        lines.extend(
            f"- {item['fact']} (`{item['path']}`)"
            for item in actions
        )
    lines.extend([
        "",
        "No workspace files were changed.",
    ])
    return "\n".join(lines)


def _extract_negative_constraints(
    results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract explicit prohibitions without applying domain-specific rules."""

    constraints: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    prohibition = re.compile(
        r"\b(?:do not|don't|must not|never)\s+"
        r"(?P<verb>use|include|add|buy|start|send|delete|remove|share|guess)\s+"
        r"(?P<target>.+)$",
        re.IGNORECASE,
    )

    for result in results:
        path = str(result.get("path") or "")
        for raw_line in str(result.get("preview") or "").splitlines():
            source_line = raw_line.strip().lstrip("-* ").strip()
            match = prohibition.search(source_line)
            if not match:
                continue
            target = re.split(
                r"\b(?:in|for|before|until|unless|because|when)\b",
                match.group("target"),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;!?\"'")
            if not target:
                continue
            key = (path.casefold(), target.casefold())
            if key in seen:
                continue
            seen.add(key)
            constraints.append({
                "path": path,
                "verb": match.group("verb").lower(),
                "target": target,
                "source_line": source_line,
            })
    return constraints


def _enforce_negative_constraints(
    answer: str,
    constraints: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Remove positive recommendation lines that violate source prohibitions."""

    lines = str(answer or "").splitlines()
    kept: list[str] = []
    removed: list[dict[str, str]] = []

    for line in lines:
        low = line.lower()
        drop_for: dict[str, str] | None = None
        rewritten_line: str | None = None
        matched_constraint: dict[str, str] | None = None
        for constraint in constraints:
            target = constraint["target"].lower()
            position = low.find(target)
            if position < 0:
                continue
            prefix = low[max(0, position - 48):position]
            semantic_prefix = re.sub(r"[*_`~]", "", prefix)
            semantic_prefix = re.sub(r"\s+", " ", semantic_prefix)
            negated = bool(re.search(
                r"(?:do not|don't|must not|never|not|without)"
                r"(?!\s+only\b)(?:\s+\w+){0,3}\s+$|"
                r"(?:avoid(?:s|ed|ing)?|exclud(?:e|es|ed|ing))"
                r"(?:\s+\w+){0,2}\s+$",
                semantic_prefix,
            ))
            bullet = bool(re.match(r"^\s*[-*•]\s+", line))
            recommends = bool(re.search(
                r"\b(?:need|should|must|recommend|buy|include|add|use|start)\b",
                low[:position + len(target)],
            ))
            if not negated and (bullet or recommends):
                if bullet:
                    payload = re.sub(r"^\s*[-*•]\s+", "", line)
                    target_position = payload.lower().find(target)
                    meaningful_prefix = payload[:target_position].strip(
                        " ,;:&+*_`~"
                    )
                    if meaningful_prefix:
                        rewritten_line = re.sub(
                            rf"(?:,\s*)?(?:and\s+)?{re.escape(constraint['target'])}\b",
                            "",
                            line,
                            flags=re.IGNORECASE,
                        )
                        rewritten_line = re.sub(r"\s+,", ",", rewritten_line)
                        rewritten_line = re.sub(r",\s*$", "", rewritten_line)
                    else:
                        drop_for = constraint
                else:
                    rewritten_line = re.sub(
                        rf"(?:,\s*)?(?:and\s+)?{re.escape(constraint['target'])}\b",
                        "",
                        line,
                        flags=re.IGNORECASE,
                    )
                matched_constraint = constraint
                break
        if rewritten_line is not None:
            kept.append(rewritten_line)
            removed.append({
                "line": line.strip(),
                "target": matched_constraint["target"] if matched_constraint else "",
                "path": matched_constraint["path"] if matched_constraint else "",
            })
        elif drop_for is None:
            kept.append(line)
        else:
            removed.append({
                "line": line.strip(),
                "target": drop_for["target"],
                "path": drop_for["path"],
            })

    if removed:
        kept.extend(["", "Source constraints applied:"])
        used: set[tuple[str, str]] = set()
        for constraint in constraints:
            key = (constraint["path"], constraint["source_line"])
            if key in used:
                continue
            used.add(key)
            kept.append(
                f"- {constraint['source_line']} (`{constraint['path']}`)"
            )

    return "\n".join(kept).strip(), {
        "checked_constraints": len(constraints),
        "removed_conflicts": removed,
        "changed": bool(removed),
    }



def _query_terms(query: str) -> list[str]:
    text = str(query or "").lower()
    terms = re.findall(r"[a-zA-Z0-9_]+", text)
    stop = {
        "what", "did", "have", "written", "write", "about", "the", "and", "for",
        "with", "from", "notes", "note", "my", "your", "in", "on", "of", "to",
    }
    out: list[str] = []
    for t in terms:
        if len(t) >= 3 and t not in stop and t not in out:
            out.append(t)
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _semantic_rerank_score(result: dict[str, Any], query: str) -> float:
    base = _safe_float(result.get("similarity"), 0.0)
    terms = _query_terms(query)

    title = str(result.get("title") or "").lower()
    path = str(result.get("path") or "").lower()
    heading = str(result.get("heading") or "").lower()
    preview = str(result.get("preview") or "").lower()

    title_path = f"{title} {path}".replace("_", " ").replace("-", " ")
    all_blob = f"{title_path} {heading} {preview}".replace("_", " ").replace("-", " ")

    score = base

    # Strongly prefer exact title/path intent matches.
    if terms and all(t in title_path for t in terms):
        score += 0.22
    if terms and all(t in title for t in terms):
        score += 0.16
    if terms and all(t in path for t in terms):
        score += 0.14

    # Heading/preview are useful, but weaker than note identity.
    if terms and all(t in heading for t in terms):
        score += 0.08
    if terms and all(t in all_blob for t in terms):
        score += 0.05

    # Troubleshooting-type queries should promote troubleshooting docs over generic setup docs.
    if "troubleshooting" in terms or "troubleshoot" in terms:
        if "troubleshooting" in title_path or "troubleshoot" in title_path:
            score += 0.20
        elif "troubleshooting" in heading or "troubleshoot" in heading:
            score += 0.08
        else:
            score -= 0.06

    # Prefer AIOS/WIP agent guidance slightly over old generic Atlas pages when both match.
    if "aios" in path and ("troubleshooting" in title_path or "troubleshoot" in title_path):
        score += 0.06

    return round(score, 4)


def _rerank_and_diversify_results(
    results: list[dict[str, Any]],
    query: str,
    *,
    max_results: int = 6,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for r in results:
        item = dict(r)
        item["rerank_score"] = _semantic_rerank_score(item, query)
        enriched.append(item)

    enriched.sort(
        key=lambda r: (
            _safe_float(r.get("rerank_score")),
            _safe_float(r.get("similarity")),
        ),
        reverse=True,
    )

    # First pass: avoid one generic note taking 4 slots.
    picked: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for r in enriched:
        path = str(r.get("path") or "")
        if path and path in seen_paths:
            continue
        picked.append(r)
        if path:
            seen_paths.add(path)
        if len(picked) >= max_results:
            return picked

    # Second pass: fill remaining slots with additional chunks.
    for r in enriched:
        if r in picked:
            continue
        picked.append(r)
        if len(picked) >= max_results:
            break

    return picked


def _format_search_output(results: list[dict[str, Any]], query: str) -> str:
    lines = []
    lines.append(f"Semantic vault search query: {query}")
    lines.append(f"Results: {len(results)}")

    for i, r in enumerate(results, start=1):
        lines.append("")
        lines.append(f"{i}. {r.get('title') or '(untitled)'}")
        lines.append(f"   path: {r.get('path')}")
        lines.append(f"   heading: {r.get('heading')}")
        lines.append(f"   similarity: {r.get('similarity')}")
        preview = str(r.get("preview") or "").replace("\n", " ")
        lines.append(f"   preview: {preview[:420]}")

    return "\n".join(lines).strip()


def _build_final_answer(
    results: list[dict[str, Any]],
    query: str,
    *,
    source_label: str = "Ideaverse",
) -> str:
    if not results:
        return (
            f"I searched the {source_label} semantic index for **{query}**, but I did not find matching note chunks.\n\n"
            "The search was read-only. No notes were changed."
        )

    lines = []
    lines.append(f"I searched the {source_label} notes for **{query}** and found these strongest matches:")
    lines.append("")

    for i, r in enumerate(results[:6], start=1):
        title = r.get("title") or "(untitled)"
        path = r.get("path") or "(unknown path)"
        heading = r.get("heading") or "(no heading)"
        similarity = r.get("similarity")
        rerank_score = r.get("rerank_score")
        preview = str(r.get("preview") or "").replace("\n", " ").strip()

        lines.append(f"{i}. **{title}**")
        lines.append(f"   Path: `{path}`")
        lines.append(f"   Heading: `{heading}`")
        if similarity is not None:
            lines.append(f"   Similarity: `{similarity}`")
        if rerank_score is not None:
            lines.append(f"   Rerank: `{rerank_score}`")
        if preview:
            lines.append(f"   Preview: {preview[:360]}")
        lines.append("")

    lines.append("This was a read-only semantic search over the local Chroma vault index. No Obsidian notes were edited.")

    return "\n".join(lines).strip()


def run_knowledge_semantic_chat_turn(
    message: str,
    *,
    workspace: str | None = None,
    root_id: str | None = None,
    llm_config: dict[str, Any] | None = None,
) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    frames: list[str] = []
    tool_events: list[dict[str, Any]] = []

    decision = evaluate_knowledge_request(message)
    query = extract_semantic_query(message)
    explicit_workspace = str(workspace or "").strip()
    explicit_root_id = str(root_id or "").strip()
    source_label = "active workspace" if explicit_workspace else "Ideaverse"

    frames.append(sse_frame({
        "type": "agent_prep",
        "title": "Knowledge runtime",
        "content": f"Preparing a read-only semantic search of the {source_label}.",
    }))
    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Routing knowledge request",
        "content": "Checking intent, confidence, and graph safety before semantic search.",
    }))

    runtime_status = {
        "intent": f"workspace_{getattr(decision, 'intent', 'knowledge')}",
        "runtime": "knowledge",
        "runtime_family": "workspace_semantic",
        "safe_readonly": bool(getattr(decision, "is_read_only", True)),
        "safety_profile": (
            "read_only"
            if getattr(decision, "is_read_only", True)
            else "policy_gated"
        ),
        "reason": (
            "The validated active workspace is authoritative for this "
            "knowledge request."
        ),
    }

    runtime_call_id = "runtime_decision_" + uuid.uuid4().hex[:10]
    runtime_output = json.dumps(runtime_status, indent=2, ensure_ascii=False)
    frames.append(sse_frame({
        "type": "tool_start",
        "id": runtime_call_id,
        "call_id": runtime_call_id,
        "tool": "runtime_decision",
        "command": "Runtime decision",
    }))
    frames.append(sse_frame({
        "type": "tool_output",
        "id": runtime_call_id,
        "call_id": runtime_call_id,
        "tool": "runtime_decision",
        "command": "Runtime decision",
        "output": runtime_output,
        "exit_code": 0,
        "runtime_decision": runtime_status,
    }))
    tool_events.append({
        "type": "tool_output",
        "tool": "runtime_decision",
        "command": "Runtime decision",
        "output": runtime_output,
        "exit_code": 0,
        "runtime_decision": runtime_status,
    })

    # DAILY_DRIVER_PREFACE_V016F1C_START
    if getattr(decision, "is_read_only", True):
        try:
            _preface = generate_intent_preface(
                user_request=message,
                intent="vault_search",
                resource=f"{source_label} notes",
                next_action="semantic_vault_search",
                safety="read-only",
                timeout_s=8.0,
            )
            _preface_text = str(_preface.get("text") or "").strip()
            if _preface_text:
                frames.append(sse_frame({"delta": _preface_text + "\n\n"}))
        except Exception:
            # Preface is UX-only. Never block semantic search.
            pass
    # DAILY_DRIVER_PREFACE_V016F1C_END

    decision_call_id = "knowledge_decision_" + uuid.uuid4().hex[:10]
    decision_output = json.dumps(decision.to_dict(), indent=2, ensure_ascii=False)

    frames.append(sse_frame({
        "type": "tool_start",
        "id": decision_call_id,
        "call_id": decision_call_id,
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
    }))
    frames.append(sse_frame({
        "type": "tool_output",
        "id": decision_call_id,
        "call_id": decision_call_id,
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
        "output": decision_output,
    }))
    tool_events.append({
        "type": "tool_output",
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
        "output": decision_output,
    })

    if getattr(decision, "intent", "") in {"unsupported_mutation", "knowledge_propose", "knowledge_research"}:
        if decision.intent == "unsupported_mutation":
            final_answer = (
                "I cannot execute this destructive workspace request directly. "
                "The central policy path blocks delete, move, rename, purge, and "
                "cleanup operations from automatic execution.\n\n"
                "No files were changed. I can prepare a non-executing proposal; "
                "any later mutating action would require explicit confirmation."
            )
        elif decision.intent == "knowledge_propose":
            final_answer = (
                "I cannot apply this workspace write automatically. No files were "
                "changed. I can prepare a reviewable proposal or copy, and execution "
                "would require explicit confirmation through the policy gate."
            )
        else:
            final_answer = (
                "This request requires the isolated research capability rather "
                "than a workspace write. No notes were changed."
            )
        frames.append(sse_frame({"delta": final_answer}))
        metrics = {
            "runtime": "knowledge_semantic_chat",
            "knowledge_runtime": "semantic_vault_search",
            "knowledge_intent": getattr(decision, "intent", ""),
            "knowledge_confidence": getattr(decision, "confidence", ""),
            "needs_human": getattr(decision, "needs_human", False),
            "read_only": True,
            "write_capable": False,
            "elapsed_s": round(time.time() - started, 3),
        }
        frames.append(sse_frame({"type": "metrics", "data": metrics}))
        return final_answer, frames, tool_events, metrics

    # Respect HITL/blocks before any search.
    if getattr(decision, "needs_human", False) and getattr(decision, "intent", "") not in {"knowledge_read"}:
        final_answer = build_human_clarification_prompt(decision)
        frames.append(sse_frame({"delta": final_answer}))
        metrics = {
            "runtime": "knowledge_semantic_chat",
            "knowledge_runtime": "semantic_vault_search",
            "knowledge_intent": getattr(decision, "intent", ""),
            "knowledge_confidence": getattr(decision, "confidence", ""),
            "needs_human": True,
            "read_only": True,
            "write_capable": False,
            "elapsed_s": round(time.time() - started, 3),
        }
        frames.append(sse_frame({"type": "metrics", "data": metrics}))
        return final_answer, frames, tool_events, metrics

    search_call_id = "knowledge_semantic_search_" + uuid.uuid4().hex[:10]

    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Searching semantic vault index",
        "content": f"Searching the {source_label} Chroma index for: {query}",
    }))
    frames.append(sse_frame({
        "type": "tool_start",
        "id": search_call_id,
        "call_id": search_call_id,
        "tool": "knowledge_semantic_search",
        "command": "Search semantic vault index",
    }))

    source_root_id = explicit_root_id or None
    worker_result = None
    retrieval_queries = _planning_retrieval_queries(message)
    if explicit_workspace or explicit_root_id:
        worker_result = run_knowledge_worker(
            WorkerHandoff(
                capability="knowledge_search",
                objective=message,
                context={
                    "query": query,
                    "workspace": explicit_workspace,
                    "root_id": explicit_root_id,
                    "retrieval_queries": retrieval_queries,
                },
            )
        )
        source_root_id = worker_result.data.get("root_id") or source_root_id
        raw_results = list(worker_result.data.get("results") or [])
    else:
        raw_results = search_semantic_vault_index(query, k=18)
    results = _rerank_and_diversify_results(
        raw_results,
        query,
        max_results=10 if retrieval_queries else 6,
    )
    arithmetic_fact = _build_currency_total_fact(query, results)
    planning_agenda = (
        _build_planning_agenda(results)
        if retrieval_queries
        else []
    )
    balanced_priorities = _select_balanced_planning_priorities(
        message,
        planning_agenda,
    )
    negative_constraints = _extract_negative_constraints(results)
    if source_root_id:
        stats = get_semantic_vault_stats(index_namespace=source_root_id)
    elif explicit_workspace or explicit_root_id:
        # An explicit but unresolved source must not even consult metadata for
        # the default/global namespace. Keep the failure fully fail-closed.
        stats = {}
    else:
        stats = get_semantic_vault_stats()
    output = _format_search_output(results, query)
    if worker_result is not None:
        output = (
            f"Source authority: active workspace\n"
            f"Workspace: {explicit_workspace}\n"
            f"Root ID: {source_root_id or '(not indexed)'}\n"
            f"Worker status: {worker_result.status}\n"
            f"Worker summary: {worker_result.summary}\n\n"
            f"{output}"
        )
    if arithmetic_fact:
        arithmetic_output = json.dumps(
            arithmetic_fact,
            ensure_ascii=False,
            indent=2,
        )
        arithmetic_call_id = "knowledge_arithmetic_" + uuid.uuid4().hex[:10]
        frames.append(sse_frame({
            "type": "tool_start",
            "id": arithmetic_call_id,
            "call_id": arithmetic_call_id,
            "tool": "knowledge_arithmetic",
            "command": "Compute total from retrieved currency evidence",
        }))
        frames.append(sse_frame({
            "type": "tool_output",
            "id": arithmetic_call_id,
            "call_id": arithmetic_call_id,
            "tool": "knowledge_arithmetic",
            "command": "Compute total from retrieved currency evidence",
            "output": arithmetic_output,
        }))
        tool_events.append({
            "type": "tool_output",
            "tool": "knowledge_arithmetic",
            "command": "Compute total from retrieved currency evidence",
            "output": arithmetic_output,
        })
        output += "\n\nDeterministic arithmetic evidence:\n" + arithmetic_output

    frames.append(sse_frame({
        "type": "tool_output",
        "id": search_call_id,
        "call_id": search_call_id,
        "tool": "knowledge_semantic_search",
        "command": "Search semantic vault index",
        "output": output,
    }))
    tool_events.append({
        "type": "tool_output",
        "tool": "knowledge_semantic_search",
        "command": "Search semantic vault index",
        "output": output,
    })

    if worker_result is not None and worker_result.status == "failed":
        final_answer = (
            "Workspace knowledge retrieval failed, so I stopped without "
            "searching Ideaverse or another global root."
        )
    elif worker_result is not None and worker_result.data.get("index_required"):
        final_answer = (
            "The active workspace is not indexed yet, so I stopped without "
            "searching Ideaverse or another global root."
        )
    else:
        final_answer = _build_final_answer(
            results,
            query,
            source_label=source_label,
        )
        if arithmetic_fact:
            final_answer = (
                "The retrieved currency entries produce a verified total of "
                f"{arithmetic_fact['currency']}"
                f"{arithmetic_fact['total_formatted']}.\n\n"
                + final_answer
            )
        if balanced_priorities:
            final_answer = _format_balanced_priority_answer(
                balanced_priorities
            )
            llm_response = {
                "used_llm": False,
                "status": "skipped_deterministic_priority_contract",
            }
        elif retrieval_queries and planning_agenda:
            final_answer = _format_comprehensive_planning_answer(
                planning_agenda
            )
            llm_response = {
                "used_llm": False,
                "status": "skipped_deterministic_planning_contract",
            }
        else:
            llm_response = try_knowledge_llm_response(
                llm_config=llm_config,
                user_message=message,
                deterministic_answer=final_answer,
                runtime_decision=decision.to_dict(),
                evidence_summary={
                    "source_authority": (
                        "workspace" if explicit_workspace else "explicit_vault_request"
                    ),
                    "workspace": explicit_workspace or None,
                    "root_id": source_root_id,
                    "negative_constraints": negative_constraints,
                    "balanced_priorities": balanced_priorities,
                    "planning_agenda": planning_agenda,
                    "computed_facts": [arithmetic_fact] if arithmetic_fact else [],
                    "results": results,
                    "retrieval_strategy": (
                        "multi_query_planning_coverage"
                        if retrieval_queries
                        else "single_query_semantic"
                    ),
                    "retrieval_queries": [query, *retrieval_queries],
                },
                hitl_policy={
                    "read_only": True,
                    "write_allowed": False,
                },
                legacy_fallback_contract={
                    "global_fallback_allowed": not bool(explicit_workspace),
                    "fail_closed": bool(explicit_workspace),
                },
            )
            if llm_response.get("used_llm") and str(
                llm_response.get("final_answer") or ""
            ).strip():
                final_answer = str(llm_response["final_answer"]).strip()

    constraint_report = {
        "checked_constraints": 0,
        "removed_conflicts": [],
        "changed": False,
    }
    if negative_constraints:
        final_answer, constraint_report = _enforce_negative_constraints(
            final_answer,
            negative_constraints,
        )
        constraint_call_id = "knowledge_constraint_check_" + uuid.uuid4().hex[:10]
        constraint_output = json.dumps(
            constraint_report,
            ensure_ascii=False,
            indent=2,
        )
        frames.append(sse_frame({
            "type": "tool_start",
            "id": constraint_call_id,
            "call_id": constraint_call_id,
            "tool": "knowledge_constraint_check",
            "command": "Validate answer against explicit source prohibitions",
        }))
        frames.append(sse_frame({
            "type": "tool_output",
            "id": constraint_call_id,
            "call_id": constraint_call_id,
            "tool": "knowledge_constraint_check",
            "command": "Validate answer against explicit source prohibitions",
            "output": constraint_output,
        }))
        tool_events.append({
            "type": "tool_output",
            "tool": "knowledge_constraint_check",
            "command": "Validate answer against explicit source prohibitions",
            "output": constraint_output,
        })

    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Finalizing grounded note answer",
        "content": "Preparing a read-only answer with source paths and headings.",
    }))

    frames.append(sse_frame({"delta": final_answer}))

    metrics = {
        "runtime": "knowledge_semantic_chat",
        "knowledge_runtime": "semantic_vault_search",
        "knowledge_intent": getattr(decision, "intent", ""),
        "knowledge_confidence": getattr(decision, "confidence", ""),
        "needs_human": False,
        "read_only": True,
        "write_capable": False,
        "semantic_query": query,
        "semantic_results": len(results),
        "semantic_collection": stats.get("collection_name"),
        "semantic_namespace_count": stats.get("namespace_count"),
        "workspace": explicit_workspace or None,
        "root_id": source_root_id,
        "source_authority": "workspace" if explicit_workspace else "explicit_vault_request",
        "retrieval_strategy": (
            "multi_query_planning_coverage"
            if retrieval_queries
            else "single_query_semantic"
        ),
        "retrieval_query_count": 1 + len(retrieval_queries),
        "deterministic_arithmetic": bool(arithmetic_fact),
        "planning_agenda_items": len(planning_agenda),
        "balanced_priority_items": len(balanced_priorities),
        "negative_constraints_checked": constraint_report["checked_constraints"],
        "negative_conflicts_removed": len(constraint_report["removed_conflicts"]),
        "elapsed_s": round(time.time() - started, 3),
        "llm_synthesis": bool(
            'llm_response' in locals() and llm_response.get("used_llm")
        ),
        "llm_synthesis_status": (
            llm_response.get("status")
            if 'llm_response' in locals()
            else "not_applicable"
        ),
    }
    frames.append(sse_frame({"type": "metrics", "data": metrics}))

    return final_answer, frames, tool_events, metrics
