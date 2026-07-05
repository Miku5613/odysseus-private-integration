from __future__ import annotations

import re


STOP_EXACT = {
    "find",
    "where",
    "use",
    "tools",
    "needed",
    "guess",
    "project",
    "frontend",
    "backend",
    "agent",
    "loop",
    "chat_routes",
}


def normalize_text(text: str | None) -> str:
    return str(text or "").lower().replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")


def extract_exact_terms(query: str | None) -> list[str]:
    """Extract user-specified exact/code-like terms that should be respected.

    These are not broad semantic hints. They are concrete targets:
    - /api/chat_stream
    - data: [DONE]
    - [DONE]
    - message_saved
    - tool_output
    - definitely_not_a_real_function_name
    """

    raw = str(query or "")
    terms: list[str] = []

    # API paths.
    for m in re.findall(r'/[A-Za-z0-9_./:-]+', raw):
        terms.append(m)

    # Bracketed protocol markers like [DONE].
    for m in re.findall(r'\[[A-Za-z0-9_:-]+\]', raw):
        terms.append(m)

    # "data: [DONE]" style SSE exact marker.
    for m in re.findall(r'data:\s*\[[A-Za-z0-9_:-]+\]', raw, flags=re.IGNORECASE):
        terms.append(m)

    # Quoted literals.
    for m in re.findall(r'["\']([^"\']{3,80})["\']', raw):
        if any(ch in m for ch in "_:/[]-.") or m.lower().endswith(("event", "events")):
            terms.append(m)

    # Identifier-like names containing underscores.
    for m in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', raw):
        lower = m.lower()
        if "_" in m and len(m) >= 4 and lower not in STOP_EXACT:
            terms.append(m)

    # De-duplicate while preserving order.
    seen = set()
    out = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            out.append(t)
            seen.add(key)

    return out


def exact_term_present(text: str | None, term: str) -> bool:
    hay_raw = str(text or "")
    hay = hay_raw.lower()
    term_raw = str(term or "")
    t = term_raw.lower()

    if not t:
        return True

    if t in hay:
        return True

    # Allow underscore/space equivalence for identifiers.
    if "_" in t:
        return t.replace("_", " ") in normalize_text(hay_raw)

    # data: [DONE] can appear inside escaped SSE string fragments.
    if t.startswith("data:"):
        compact_hay = re.sub(r'\s+', ' ', hay)
        compact_t = re.sub(r'\s+', ' ', t)
        return compact_t in compact_hay or "[done]" in hay

    return False


def missing_exact_terms(text: str | None, query: str | None) -> list[str]:
    return [term for term in extract_exact_terms(query) if not exact_term_present(text, term)]
