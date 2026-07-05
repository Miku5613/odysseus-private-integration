
from __future__ import annotations

from pathlib import Path
import fnmatch
import os
import re
from typing import Iterable


SKIP_DIRS = {
    ".git",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "data",
    "logs",
    "Exclusions",
    ".mypy_cache",
    ".ruff_cache",

    # v0.13 general runtime should not cite its own scaffolding
    # as evidence for app/frontend/backend behavior.
    "agent_runtime",
}

SKIP_FILE_PARTS = {
    "patch_",
    "install_",
    "inspect_",
    "backup_",
    "langgraph_integration_inspect",
    "integration_inspect",
    "diagnostic",
    "scratch",
    ".pyc",
}


def _safe_root(workspace: str) -> Path:
    root = Path(workspace).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Workspace does not exist: {root}")
    return root


def _safe_path(workspace: str, path: str | None = None) -> Path:
    root = _safe_root(workspace)
    p = root if not path else (root / path).resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"Path escapes workspace: {path}")
    return p


def _iter_files(workspace: str) -> Iterable[Path]:
    root = _safe_root(workspace)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            low = name.lower()
            if any(part in low for part in SKIP_FILE_PARTS):
                continue
            yield base / name


def get_workspace(workspace: str) -> str:
    return str(_safe_root(workspace))


def ls(workspace: str, path: str = "") -> str:
    p = _safe_path(workspace, path)
    if not p.exists():
        return f"NOT_FOUND: {path}"
    if p.is_file():
        return str(p.relative_to(_safe_root(workspace)))
    rows = []
    for child in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        suffix = "/" if child.is_dir() else ""
        rows.append(str(child.relative_to(_safe_root(workspace))) + suffix)
    return "\n".join(rows[:300])


def glob_files(workspace: str, pattern: str, max_results: int = 80) -> str:
    root = _safe_root(workspace)
    out = []
    for f in _iter_files(workspace):
        rel = str(f.relative_to(root)).replace("\\", "/")
        if fnmatch.fnmatch(rel, pattern):
            out.append(rel)
            if len(out) >= max_results:
                break
    return "\n".join(out)


def grep(workspace: str, pattern: str, path: str = "", max_results: int = 80) -> str:
    root = _safe_root(workspace)
    start = _safe_path(workspace, path)
    files = [start] if start.is_file() else list(_iter_files(str(start)))
    rx = re.compile(pattern, re.IGNORECASE)
    rows = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                rel = str(file.resolve().relative_to(root)).replace("\\", "/")
                rows.append(f"{rel}:{i}: {line[:240]}")
                if len(rows) >= max_results:
                    return "\n".join(rows)
    return "\n".join(rows)


def read_file(workspace: str, path: str, offset: int = 0, limit: int = 120) -> str:
    root = _safe_root(workspace)
    p = _safe_path(workspace, path)
    if not p.exists() or not p.is_file():
        return f"NOT_FOUND: {path}"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, offset)
    end = min(len(lines), start + max(1, limit))
    rel = str(p.relative_to(root)).replace("\\", "/")
    out = []
    for idx in range(start, end):
        out.append(f"{rel}:{idx+1}: {lines[idx]}")
    return "\n".join(out)


def read_file_around(workspace: str, path: str, line: int, radius: int = 20) -> str:
    start = max(0, int(line) - int(radius) - 1)
    return read_file(workspace, path, offset=start, limit=(int(radius) * 2 + 1))

