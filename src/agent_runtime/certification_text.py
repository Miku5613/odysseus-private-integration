from __future__ import annotations

import re
import unicodedata


_CONTRACTIONS = {
    "can't": "cannot",
    "cannot": "cannot",
    "couldn't": "cannot",
    "could not": "cannot",
    "won't": "will not",
    "wouldn't": "would not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
}
_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)


def _normalize_semantic_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", text)
    for month in _MONTHS:
        text = re.sub(
            rf"\b({month})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
            r"\2 \1",
            text,
        )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def semantic_contains(answer: str, expected: str) -> bool:
    return _normalize_semantic_text(expected) in _normalize_semantic_text(answer)


def positive_forbidden_hit(answer: str, forbidden: str) -> bool:
    normalized = _normalize_semantic_text(answer)
    needle = _normalize_semantic_text(forbidden)
    for match in re.finditer(rf"\b{re.escape(needle)}\b", normalized):
        prefix = normalized[max(0, match.start() - 32):match.start()]
        if re.search(r"\b(?:do not|does not|did not|never|avoid)\s+(?:use\s+)?$", prefix):
            continue
        return True
    return False


__all__ = ["positive_forbidden_hit", "semantic_contains"]
