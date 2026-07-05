from __future__ import annotations

import fnmatch
from pathlib import Path


SOURCE_CODE_DIRS = (
    "src",
    "routes",
    "static",
    "core",
    "services",
    "integrations",
    "companion",
)

GENERATED_PATTERNS = (
    "langgraph_*_inspect.txt",
    "Pasted text*.txt",
    "patch_runtime_*.py",
    "inspect_*.py",
    "fix_runtime_*.py",
)

LOW_AUTHORITY_DIRS = (
    "logs",
    "backups",
    "backup",
    "docs",
    "tests",
)


def normalize_path(path: str | None) -> str:
    raw = str(path or "").strip().strip('"').strip("'")
    if not raw:
        return ""

    raw = raw.replace("/", "\\")
    parts = [p for p in raw.split("\\") if p]

    if "odysseus-upstream-test" in parts:
        idx = parts.index("odysseus-upstream-test")
        return "\\".join(parts[idx + 1 :])

    return raw


def classify_source(path: str | None) -> str:
    p = normalize_path(path)
    if not p:
        return "unknown"

    name = Path(p).name
    lower = p.lower().replace("/", "\\")

    for pattern in GENERATED_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return "generated_artifact"

    first = lower.split("\\", 1)[0]

    if first in LOW_AUTHORITY_DIRS:
        return first

    if first in SOURCE_CODE_DIRS:
        return "source_code"

    if lower.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".json", ".yaml", ".yml", ".toml")):
        return "source_code"

    if lower.endswith((".md", ".txt")):
        return "document"

    return "unknown"


def authority_score(authority: str) -> int:
    return {
        "source_code": 10,
        "tests": 6,
        "document": 4,
        "docs": 4,
        "logs": 3,
        "generated_artifact": -8,
        "unknown": 0,
    }.get(authority, 0)


def should_skip_source(path: str | None) -> bool:
    authority = classify_source(path)
    return authority == "generated_artifact"
