from __future__ import annotations

import re
from typing import Any

from src.agent_runtime.evidence import extract_code_lines
from src.agent_runtime.source_authority import (
    authority_score,
    classify_source,
    normalize_path,
    should_skip_source,
)


NO_MATCH_RE = re.compile(r'^\s*No matches\.?\s*$', re.IGNORECASE)
FILE_HEADER_RE = re.compile(r'^# File:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
WINDOW_RE = re.compile(
    r'^# Showing lines\s+(\d+)-(\d+)\s+around target line\s+(\d+)',
    re.IGNORECASE | re.MULTILINE,
)

WEAK_LINE_RE = re.compile(
    r'^\s*('
    r'"""|\'\'\'|#|'
    r'import\s+|from\s+|'
    r'logger\s*=|'
    r'["\'][-A-Za-z0-9 ,.;:!?`_(){}\[\]/\\]+["\']\s*,?\s*$'
    r')',
    re.IGNORECASE,
)


def _query_terms(query: str | None) -> set[str]:
    raw = str(query or "").lower()
    raw = raw.replace("_", " ").replace("-", " ").replace("/", " ")
    terms = set(re.findall(r'[a-zA-Z][a-zA-Z0-9]{2,}', raw))
    stop = {
        "the", "and", "for", "where", "find", "show", "main", "implemented",
        "implementation", "project", "file", "code", "use", "tools", "needed",
        "guess", "does", "this", "that", "with", "from", "into", "current",
        "after", "before", "around",
    }
    return {t for t in terms if t not in stop}


def relevance_score(text: str, query: str | None, path: str | None = None) -> int:
    terms = _query_terms(query)
    if not terms:
        return 0

    haystack = (str(text or "") + " " + str(path or "")).lower()
    haystack = haystack.replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")

    hits = sum(1 for term in terms if term in haystack)
    return min(hits, 6) * 3


def path_intent_score(path: str | None, query: str | None) -> int:
    p = str(path or "").lower().replace("/", "\\")
    q = str(query or "").lower().replace("_", " ").replace("-", " ")

    score = 0

    frontend_words = {"frontend", "front end", "client", "ui", "browser", "javascript", "js"}
    backend_words = {"backend", "server", "api", "route", "routes", "fastapi"}
    agent_words = {"agent", "agent loop", "tool call", "tool calls"}
    chat_route_words = {"chat_routes", "chat routes", "chat route"}

    if any(word in q for word in frontend_words):
        if p.startswith("static\\") or "\\static\\" in p or p.endswith((".js", ".jsx", ".ts", ".tsx", ".css", ".html")):
            score += 14
        if p.startswith("routes\\") or p.startswith("src\\") or p.startswith("core\\"):
            score -= 6

    if any(word in q for word in backend_words):
        if p.startswith("routes\\") or p.startswith("src\\") or p.startswith("services\\") or p.startswith("core\\"):
            score += 8

    if any(word in q for word in agent_words):
        if "src\\agent_loop.py" in p:
            score += 12
        elif p.startswith("src\\"):
            score += 4

    if any(word in q for word in chat_route_words):
        if "routes\\chat_routes.py" in p:
            score += 14

    return score


def observation_key(obs: dict[str, Any]) -> str:
    return "|".join(
        [
            str(obs.get("kind", "")),
            str(obs.get("source_path", "")),
            str(obs.get("line_start", "")),
            str(obs.get("line_end", "")),
            str(obs.get("content", ""))[:120],
        ]
    )


def _is_weak_line(text: str | None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    return bool(WEAK_LINE_RE.match(raw))


def _has_exact_or_action_signal(text: str | None, query: str | None) -> bool:
    hay = str(text or "").lower().replace("_", " ")
    q = str(query or "").lower().replace("_", " ")

    exact_identifiers = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', str(query or ""))
    exact_identifiers = [x.lower() for x in exact_identifiers if "_" in x and len(x) >= 4]

    if any(x in str(text or "").lower() for x in exact_identifiers):
        return True

    if "save" in q and re.search(r'\bsave|saved|saving|commit|insert|create\b', hay):
        return True

    if any(word in q for word in ["execute", "executes", "executing", "run", "runs", "running"]):
        if re.search(r'\bexecute|invoke|called|call\(|await |for .* in |\.invoke\(', hay):
            return True

    if any(word in q for word in ["handle", "handles", "handling"]):
        if re.search(r'\bif\b|\belif\b|\bcase\b|\bswitch\b|\bhandler\b|\bevent\b|\btype\b', hay):
            return True

    return False



def _looks_like_definition_or_import(text: str | None) -> bool:
    raw = str(text or "").strip()
    return bool(
        re.search(r'^\s*(?:async\s+)?def\s+', raw)
        or re.search(r'^\s*class\s+', raw)
        or re.search(r'^\s*(?:from\s+|import\s+)', raw)
    )


def _looks_like_concrete_call_site(text: str | None, query: str | None) -> bool:
    """Prefer actual invocation sites over imports/definitions/parsers.

    Examples we want to boost:
      return await execute_tool_block(...)
      result = tool_obj.invoke(args)
      await some_tool(...)
      fn(...)

    Examples we do not want to boost:
      def execute_tool_block(...)
      from x import execute_tool_block
      parse_tool_blocks(...)
    """

    raw = str(text or "")
    compact = raw.strip()
    lowered = compact.lower()

    if _looks_like_definition_or_import(compact):
        return False

    q = str(query or "").lower()

    # Strongest signal: explicit awaited/returned function call.
    if re.search(r'\b(return\s+)?await\s+[A-Za-z_][A-Za-z0-9_]*\s*\(', compact):
        return True

    # Method/function invocation.
    if re.search(r'[A-Za-z_][A-Za-z0-9_\.]*\s*\(', compact):
        if any(word in q for word in ["execute", "executes", "executing", "run", "runs", "running", "call", "calls"]):
            call_terms = [
                "execute_tool_block",
                "invoke(",
                ".invoke(",
                "execute",
                "tool_obj",
                "tool_block",
                "tool_call",
            ]
            return any(term in lowered for term in call_terms)

        if any(word in q for word in ["save", "saves", "saving", "saved"]):
            return any(term in lowered for term in ["save_assistant_response", "message_saved", "commit("])

    return False


def _call_site_score(text: str | None, query: str | None) -> int:
    if _looks_like_concrete_call_site(text, query):
        return 18

    raw = str(text or "").strip().lower()

    # Penalize definitions/imports for action questions. They are useful context,
    # but not the implementation site.
    q = str(query or "").lower()
    if any(word in q for word in ["execute", "executes", "executing", "run", "runs", "running", "save", "saves", "saving", "saved"]):
        if _looks_like_definition_or_import(raw):
            return -10
        if "parse_tool_blocks" in raw or "_resolve_tool_blocks" in raw:
            return -4

    return 0


def make_observation(
    *,
    observation_id: str,
    kind: str,
    source_path: str = "",
    line_start: int | None = None,
    line_end: int | None = None,
    content: str = "",
    source_tool: str = "",
    derived_from_span: str = "",
    query: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = normalize_path(source_path)
    authority = classify_source(path)
    rel = relevance_score(content, query, path)
    auth = authority_score(authority)
    path_intent = path_intent_score(path, query)

    specificity = 0
    if line_start:
        specificity += 4
    if line_end and line_end != line_start:
        specificity += 3
    if kind == "context_window":
        specificity += 8
    if kind == "code_line":
        specificity += 3
    if kind == "negative_observation":
        specificity += 1

    weak_line = _is_weak_line(content)
    if kind == "code_line" and weak_line:
        specificity -= 8

    if _has_exact_or_action_signal(content, query):
        specificity += 8

    call_site = _call_site_score(content, query)
    specificity += call_site

    confidence = auth + rel + specificity + path_intent

    return {
        "observation_id": observation_id,
        "kind": kind,
        "source_path": path,
        "source_authority": authority,
        "line_start": line_start,
        "line_end": line_end or line_start,
        "content": str(content or ""),
        "source_tool": source_tool,
        "derived_from_span": derived_from_span,
        "query_relevance": rel,
        "authority_score": auth,
        "specificity_score": specificity,
        "path_intent_score": path_intent,
        "weak_line": weak_line,
        "call_site_score": call_site,
        "confidence": confidence,
        "metadata": metadata or {},
    }


def _read_file_around_window_observation(
    *,
    args: dict[str, Any],
    content: str,
    span_id: str,
    query: str | None,
    observation_id: str,
) -> dict[str, Any] | None:
    file_match = FILE_HEADER_RE.search(content)
    path = normalize_path(file_match.group(1)) if file_match else normalize_path(str(args.get("path") or ""))

    if not path or should_skip_source(path):
        return None

    win = WINDOW_RE.search(content)
    if win:
        start = int(win.group(1))
        end = int(win.group(2))
        target = int(win.group(3))
    else:
        start = int(args.get("line") or 1)
        end = start
        target = start

    return make_observation(
        observation_id=observation_id,
        kind="context_window",
        source_path=path,
        line_start=start,
        line_end=end,
        content=content,
        source_tool="read_file_around",
        derived_from_span=span_id,
        query=query,
        metadata={"target_line": target},
    )


def observations_from_tool_result(
    *,
    tool_name: str,
    args: dict[str, Any],
    content: str,
    span_id: str,
    query: str | None,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    text = str(content or "")
    observations: list[dict[str, Any]] = []

    if tool_name == "read_file_around":
        window_obs = _read_file_around_window_observation(
            args=args,
            content=text,
            span_id=span_id,
            query=query,
            observation_id=f"obs_{start_index + len(observations) + 1}",
        )
        if window_obs:
            observations.append(window_obs)

        # Also keep individual code-line observations for anchors/ranking, but
        # the full context-window is the primary claim-support unit.
        code_lines = extract_code_lines(tool_name, args, text)
        for line in code_lines:
            path = normalize_path(line.get("file", ""))
            if not path or should_skip_source(path):
                continue
            observations.append(
                make_observation(
                    observation_id=f"obs_{start_index + len(observations) + 1}",
                    kind="code_line",
                    source_path=path,
                    line_start=int(line["line"]),
                    line_end=int(line["line"]),
                    content=line.get("text", ""),
                    source_tool=tool_name,
                    derived_from_span=span_id,
                    query=query,
                    metadata={"raw": line.get("raw", "")},
                )
            )

    elif tool_name in {"grep", "read_file"}:
        code_lines = extract_code_lines(tool_name, args, text)

        for line in code_lines:
            path = normalize_path(line.get("file", ""))
            if not path or should_skip_source(path):
                continue

            observations.append(
                make_observation(
                    observation_id=f"obs_{start_index + len(observations) + 1}",
                    kind="code_line",
                    source_path=path,
                    line_start=int(line["line"]),
                    line_end=int(line["line"]),
                    content=line.get("text", ""),
                    source_tool=tool_name,
                    derived_from_span=span_id,
                    query=query,
                    metadata={"raw": line.get("raw", "")},
                )
            )

    elif tool_name == "glob_files":
        for raw in text.splitlines():
            path = normalize_path(raw.strip())
            if not path or should_skip_source(path):
                continue
            observations.append(
                make_observation(
                    observation_id=f"obs_{start_index + len(observations) + 1}",
                    kind="file_inventory",
                    source_path=path,
                    content=path,
                    source_tool=tool_name,
                    derived_from_span=span_id,
                    query=query,
                )
            )

    elif tool_name in {"ls", "get_workspace"}:
        pass

    if NO_MATCH_RE.match(text):
        path = normalize_path(str(args.get("path") or ""))
        observations.append(
            make_observation(
                observation_id=f"obs_{start_index + len(observations) + 1}",
                kind="negative_observation",
                source_path=path,
                content=f"{tool_name} found no matches for {args}",
                source_tool=tool_name,
                derived_from_span=span_id,
                query=query,
            )
        )

    return observations


def rank_observations(observations: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return sorted(
        observations or [],
        key=lambda o: (
            -int(o.get("confidence", 0)),
            -int(o.get("query_relevance", 0)),
            str(o.get("source_path", "")),
            int(o.get("line_start") or 0),
        ),
    )


def pick_context_candidate(
    observations: list[dict[str, Any]] | None,
    expanded: list[str] | None,
    *,
    query: str | None = None,
    min_confidence: int = 18,
) -> dict[str, Any] | None:
    expanded_set = set(expanded or [])

    for obs in rank_observations(observations):
        if obs.get("kind") != "code_line":
            continue
        if obs.get("source_authority") != "source_code":
            continue
        if obs.get("source_path") in expanded_set:
            continue
        if obs.get("observation_id") in expanded_set:
            continue
        if int(obs.get("confidence", 0)) < min_confidence:
            continue
        if not obs.get("line_start"):
            continue

        # Avoid expanding imports, docstrings, comments, module headers, or
        # prompt-rule string literals unless they contain exact/action evidence.
        if obs.get("weak_line") and not _has_exact_or_action_signal(obs.get("content", ""), query):
            continue

        return obs

    return None


def format_observations(observations: list[dict[str, Any]] | None, limit: int = 12) -> str:
    rows = []
    for obs in rank_observations(observations)[:limit]:
        loc = obs.get("source_path") or "(unknown)"
        if obs.get("line_start"):
            if obs.get("line_end") and obs.get("line_end") != obs.get("line_start"):
                loc += f":{obs.get('line_start')}-{obs.get('line_end')}"
            else:
                loc += f":{obs.get('line_start')}"
        rows.append(
            f"- {obs.get('observation_id')} [{obs.get('kind')}, {obs.get('source_authority')}, "
            f"confidence={obs.get('confidence')}, relevance={obs.get('query_relevance')}, "
            f"path_intent={obs.get('path_intent_score')}, call_site={obs.get('call_site_score')}] "
            f"{loc} :: {str(obs.get('content', '')).strip().replace(chr(10), ' ')[:300]}"
        )
    return "\n".join(rows) or "No observations recorded."
