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
    return raw


def classify_source(path: str | None) -> str:
    p = normalize_path(path)
    if not p:
        return "unknown"

    name = Path(p).name
    lower = p.lower().replace("/", "\\")
    parts = {part for part in lower.split("\\") if part}

    # Local scaffolding/history can contain real-looking source references,
    # but it is not active implementation evidence.
    if parts & {
        "exclusions",
        "backups",
        "patches",
        "reports",
        "inspection_reports",
        "looserootartifacts",
        "patchscripts",
        "inspectscripts",
    }:
        return "generated_artifact"


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
    """Compatibility hook: source-like artifacts are no longer hard-filtered.

    ACTIVE COMPATIBILITY SEAM: evidence-ledger callers still invoke this
    function. Returning False is intentional; ``authority_score`` supplies the
    negative provenance weight and lets unusual/backup-only evidence remain
    available when no stronger source exists.
    """
    return False
