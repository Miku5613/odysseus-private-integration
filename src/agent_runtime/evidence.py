from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict


class EvidenceItem(TypedDict, total=False):
    kind: str
    strength: str
    source_tool: str
    summary: str
    file: str
    line: int
    route: str
    symbol: str
    details: dict[str, Any]


class CodeLine(TypedDict):
    file: str
    line: int
    text: str
    raw: str


class AnchorItem(TypedDict, total=False):
    kind: str
    file: str
    line: int
    text: str
    source_tool: str
    score: int
    reason: str


FILE_HEADER_RE = re.compile(r'^# File:\s*(.+?)\s*$', re.IGNORECASE)

# Matches read_file/read_file_around style:
# 457: @router.post("/api/chat_stream")
# >> 458: async def chat_stream(...)
NUMBERED_LINE_RE = re.compile(
    r'^\s*(?:>>)?\s*(\d+):\s*(.*)$'
)

# Matches grep style:
# routes\chat_routes.py:458: async def chat_stream(...)
# src/foo.py:12: def bar(...)
GREP_LINE_RE = re.compile(
    r'^\s*(.+):(\d+):\s*(.*)$'
)

ROUTE_DECORATOR_RE = re.compile(
    r'^\s*@(?:\w+\.)?router\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)

FUNCTION_RE = re.compile(
    r'^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(',
    re.IGNORECASE,
)


def _normalize_file_path(value: str | None) -> str:
    if not value:
        return ""

    raw = str(value).strip().strip('"').strip("'")

    try:
        p = Path(raw)
        _ = p
        return raw
    except Exception:
        return raw


def _file_from_args(args: dict[str, Any]) -> str:
    for key in ("path", "file", "filepath", "file_path"):
        if key in args and args.get(key):
            return _normalize_file_path(str(args.get(key)))
    return ""


def _extract_file_header(content: str) -> str:
    for line in content.splitlines():
        m = FILE_HEADER_RE.match(line.strip())
        if m:
            return _normalize_file_path(m.group(1))
    return ""


def _route_symbol_similarity(route: str, symbol: str) -> bool:
    route_tail = route.rstrip("/").split("/")[-1].lower()
    symbol_norm = symbol.lower()

    if not route_tail:
        return False

    return (
        route_tail == symbol_norm
        or route_tail.replace("-", "_") == symbol_norm
        or route_tail.replace("-", "_").replace(".", "_") == symbol_norm
    )


def _looks_like_file_ref(value: str) -> bool:
    value = value.strip()
    if not value:
        return False

    lowered = value.lower()

    return (
        "\\" in value
        or "/" in value
        or lowered.endswith(".py")
        or lowered.endswith(".js")
        or lowered.endswith(".ts")
        or lowered.endswith(".tsx")
        or lowered.endswith(".jsx")
        or lowered.endswith(".md")
        or lowered.endswith(".json")
        or lowered.endswith(".yaml")
        or lowered.endswith(".yml")
        or lowered.endswith(".toml")
    )


def extract_code_lines(tool_name: str, args: dict[str, Any], content: str) -> list[CodeLine]:
    """Extract file/line/code observations from any tool output.

    This is intentionally not bound to one tool. It handles:
    - read_file output: "458: async def ..."
    - read_file_around output: "# File: ...", then line-numbered code
    - grep output: "routes/file.py:458: async def ..."
    - future tools that return similar file:line or numbered text
    """

    default_file = _extract_file_header(content) or _file_from_args(args)
    lines: list[CodeLine] = []

    for raw in content.splitlines():
        stripped = raw.rstrip()

        # Prefer grep-style parsing when the prefix looks like a file ref.
        grep_match = GREP_LINE_RE.match(stripped)
        if grep_match:
            maybe_file = grep_match.group(1).strip()
            if _looks_like_file_ref(maybe_file):
                lines.append(
                    {
                        "file": _normalize_file_path(maybe_file),
                        "line": int(grep_match.group(2)),
                        "text": grep_match.group(3).strip(),
                        "raw": stripped,
                    }
                )
                continue

        numbered_match = NUMBERED_LINE_RE.match(stripped)
        if numbered_match and default_file:
            lines.append(
                {
                    "file": default_file,
                    "line": int(numbered_match.group(1)),
                    "text": numbered_match.group(2).strip(),
                    "raw": stripped,
                }
            )

    return lines


def _find_route_handler_hits(
    tool_name: str,
    code_lines: list[CodeLine],
) -> list[EvidenceItem]:
    by_file: dict[str, list[CodeLine]] = {}

    for item in code_lines:
        by_file.setdefault(item["file"], []).append(item)

    hits: list[EvidenceItem] = []

    for file_path, file_lines in by_file.items():
        decorators = []
        functions = []

        for item in file_lines:
            text = item["text"]

            route_match = ROUTE_DECORATOR_RE.match(text)
            if route_match:
                decorators.append(
                    {
                        "line": item["line"],
                        "method": route_match.group(1).upper(),
                        "route": route_match.group(2),
                        "raw": item["raw"],
                    }
                )

            function_match = FUNCTION_RE.match(text)
            if function_match:
                functions.append(
                    {
                        "line": item["line"],
                        "symbol": function_match.group(1),
                        "raw": item["raw"],
                    }
                )

        for dec in decorators:
            for fn in functions:
                nearby = 0 <= fn["line"] - dec["line"] <= 5
                similar = _route_symbol_similarity(dec["route"], fn["symbol"])

                if nearby and similar:
                    hits.append(
                        {
                            "kind": "route_handler",
                            "strength": "strong",
                            "source_tool": tool_name,
                            "summary": (
                                f"Confirmed {dec['method']} {dec['route']} handled by "
                                f"{fn['symbol']} near lines {dec['line']}-{fn['line']}."
                            ),
                            "file": file_path,
                            "line": fn["line"],
                            "route": dec["route"],
                            "symbol": fn["symbol"],
                            "details": {
                                "decorator_line": dec["line"],
                                "function_line": fn["line"],
                                "method": dec["method"],
                                "decorator_raw": dec["raw"],
                                "function_raw": fn["raw"],
                            },
                        }
                    )

    return hits


def analyze_tool_result(tool_name: str, args: dict[str, Any], content: str) -> list[EvidenceItem]:
    """Analyze raw tool output and extract deterministic evidence.

    This does not use LLM reasoning and is not tied to a specific tool.
    It extracts general evidence shapes from any tool output.
    """

    content_text = str(content or "")
    code_lines = extract_code_lines(tool_name, args, content_text)

    hits: list[EvidenceItem] = []
    hits.extend(_find_route_handler_hits(tool_name, code_lines))

    return hits


def has_strong_evidence(evidence_hits: list[dict[str, Any]] | None) -> bool:
    return any(hit.get("strength") == "strong" for hit in evidence_hits or [])


def format_evidence_hits(evidence_hits: list[dict[str, Any]] | None) -> str:
    hits = evidence_hits or []
    if not hits:
        return "No runtime-verified evidence hits."

    rows = ["Runtime-verified evidence hits:"]
    for i, hit in enumerate(hits, start=1):
        rows.append(
            f"{i}. [{hit.get('strength', 'unknown')}] {hit.get('kind', 'evidence')}: "
            f"{hit.get('summary', '')}"
        )

        file_path = hit.get("file")
        line = hit.get("line")
        route = hit.get("route")
        symbol = hit.get("symbol")

        details = []
        if file_path:
            details.append(f"file={file_path}")
        if line:
            details.append(f"line={line}")
        if route:
            details.append(f"route={route}")
        if symbol:
            details.append(f"symbol={symbol}")

        if details:
            rows.append("   " + ", ".join(details))

    return "\n".join(rows)



ANCHOR_HINT_RE = re.compile(
    r'(def\s+|async\s+def\s+|class\s+|@\w+\.|router\.|app\.|'
    r'import\s+|from\s+|TODO|FIXME|ERROR|Exception|'
    r'config|setting|route|handler|endpoint|function|tool|policy|save|stream)',
    re.IGNORECASE,
)

BOILERPLATE_RE = re.compile(
    r'^\s*("""|\'\'\'|#|import\s+|from\s+|logger\s*=|$)',
    re.IGNORECASE,
)

SOURCE_TOOL_WEIGHT = {
    # Search result anchors are usually intentional and high-signal.
    "grep": 10,

    # Direct context reads are useful, but they produce lots of incidental
    # numbered lines. Treat them as weak unless the line itself strongly
    # overlaps the user query.
    "read_file": -6,

    # read_file_around is already context expansion. Do not let every nearby
    # numbered line become another expansion cascade.
    "read_file_around": -8,

    # Other tools can still produce file:line anchors later.
    "glob_files": 2,
}


def _query_terms(query: str | None) -> set[str]:
    raw = str(query or "").lower()

    # Split snake/camel-ish phrases enough for code search relevance.
    raw = raw.replace("_", " ").replace("-", " ").replace("/", " ")
    terms = set(re.findall(r'[a-zA-Z][a-zA-Z0-9]{2,}', raw))

    stop = {
        "the", "and", "for", "where", "find", "show", "main", "implemented",
        "implementation", "project", "file", "code", "use", "tools", "needed",
        "guess", "does", "this", "that", "with", "from", "into", "current",
    }
    return {t for t in terms if t not in stop}


def _query_overlap_score(text: str, query: str | None) -> int:
    terms = _query_terms(query)
    if not terms:
        return 0

    lowered = str(text or "").lower().replace("_", " ").replace("-", " ")
    hits = sum(1 for term in terms if term in lowered)

    # Reward multiple overlaps, but avoid letting generic docstrings dominate.
    return min(hits, 5) * 3


def _is_definition_or_decorator(text: str) -> bool:
    return bool(
        re.search(r'^\s*(?:async\s+)?def\s+', text)
        or re.search(r'^\s*class\s+', text)
        or re.search(r'^\s*@', text)
    )


def _anchor_score(line: CodeLine, tool_name: str = "", query: str | None = None) -> int:
    text = line.get("text", "")
    score = 1

    score += SOURCE_TOOL_WEIGHT.get(tool_name, 0)

    if ANCHOR_HINT_RE.search(text):
        score += 4
    if re.search(r'^\s*(?:async\s+)?def\s+', text):
        score += 5
    if re.search(r'^\s*class\s+', text):
        score += 4
    if re.search(r'^\s*@', text):
        score += 4
    if re.search(r'\b(ERROR|Exception|Traceback)\b', text, re.IGNORECASE):
        score += 3

    score += _query_overlap_score(text, query)

    # Penalize incidental boilerplate from broad file reads.
    if BOILERPLATE_RE.search(text):
        score -= 4

    if len(text.strip()) > 180:
        score -= 1

    return max(score, 1)


def extract_anchors(
    tool_name: str,
    args: dict[str, Any],
    content: str,
    query: str | None = None,
) -> list[AnchorItem]:
    """Extract generic file:line anchors from any tool output.

    An anchor is not final evidence. It is a promising location that may be
    expanded with nearby context before the runtime trusts it.

    v0.8.2 makes ranking query-aware and source-aware:
    - grep anchors rank highly
    - broad read_file anchors are weak unless very relevant
    - read_file_around does not trigger expansion cascades
    """

    code_lines = extract_code_lines(tool_name, args, str(content or ""))
    anchors: list[AnchorItem] = []

    for line in code_lines:
        if not line.get("file") or not line.get("line"):
            continue

        score = _anchor_score(line, tool_name=tool_name, query=query)

        # Generic anti-noise gate:
        # Broad read_file output has many numbered lines. Only keep read_file
        # anchors if they are genuine definitions/decorators or have strong
        # query overlap. grep/search anchors do not need this restriction.
        if tool_name == "read_file":
            overlap = _query_overlap_score(line.get("text", ""), query)
            if not _is_definition_or_decorator(line.get("text", "")) and overlap < 6:
                continue

        # Context expansion outputs are already expanded context. Keep only very
        # high-signal anchors from them, otherwise they create expansion chains.
        if tool_name == "read_file_around" and score < 14:
            continue

        anchors.append(
            {
                "kind": "code_anchor",
                "file": line["file"],
                "line": int(line["line"]),
                "text": line.get("text", ""),
                "source_tool": tool_name,
                "score": score,
                "reason": "query_ranked_file_line_observation",
            }
        )

    anchors.sort(key=lambda a: (-int(a.get("score", 0)), str(a.get("file", "")), int(a.get("line", 0))))
    return anchors


def anchor_key(anchor: dict[str, Any]) -> str:
    return f"{anchor.get('file')}:{anchor.get('line')}"


def anchor_key(anchor: dict[str, Any]) -> str:
    return f"{anchor.get('file')}:{anchor.get('line')}"


def pick_unexpanded_anchor(
    anchors: list[dict[str, Any]] | None,
    expanded: list[str] | None,
    *,
    min_score: int = 12,
) -> dict[str, Any] | None:
    expanded_set = set(expanded or [])

    ranked = sorted(
        anchors or [],
        key=lambda a: (-int(a.get("score", 0)), str(a.get("file", "")), int(a.get("line", 0))),
    )

    for anchor in ranked:
        key = anchor_key(anchor)
        score = int(anchor.get("score", 0))
        if (
            key not in expanded_set
            and score >= min_score
            and anchor.get("file")
            and anchor.get("line")
        ):
            return anchor

    return None
