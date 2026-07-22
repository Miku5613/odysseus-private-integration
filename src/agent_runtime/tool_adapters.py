from __future__ import annotations

from langchain_core.tools import StructuredTool

import fnmatch
import re
from contextvars import ContextVar
from pathlib import Path

from langchain_core.tools import tool


_ACTIVE_WORKSPACE: ContextVar[Path | None] = ContextVar(
    "odysseus_active_workspace",
    default=None,
)


def bind_workspace(workspace: str | Path):
    root = Path(workspace).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Workspace does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Workspace is not a directory: {root}"
        )

    return _ACTIVE_WORKSPACE.set(root)


def reset_workspace(token) -> None:
    _ACTIVE_WORKSPACE.reset(token)


def active_workspace() -> Path:
    root = _ACTIVE_WORKSPACE.get()

    if root is None:
        raise RuntimeError(
            "No active workspace is bound to this runtime request."
        )

    return root



SKIP_DIRS = {
    ".git",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "logs",
    "data",
    # Contract/source lookup must inspect the live implementation, not local
    # repair artifacts, generated reports, or historical copies.
    "exclusions",
    "backups",
    "backup",
    "archives",
    "archive",
    ".agents",
    ".codex",
}

GENERATED_SKIP_PATTERNS = {
    "langgraph_*_inspect.txt",
    "Pasted text*.txt",
    "patch_runtime_*.py",
    "inspect_*.py",
    "fix_runtime_*.py",
    "*.before_*",
    "*.bak",
    "*.backup",
    "*_backup.*",
}


def _safe_path(path: str | None = None, workspace: str | None = None) -> Path:
    root = (
        Path(workspace).resolve()
        if workspace
        else active_workspace()
    )
    raw = path or "."
    p = Path(raw)

    if not p.is_absolute():
        p = root / p

    resolved = p.resolve()

    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path outside workspace is not allowed: {resolved}")

    return resolved


def _is_skipped(path: Path) -> bool:
    if any(part.lower() in SKIP_DIRS for part in path.parts):
        return True
    name = path.name.lower()
    return any(
        fnmatch.fnmatch(name, pattern.lower())
        for pattern in GENERATED_SKIP_PATTERNS
    )


def _expand_brace_glob(pattern: str) -> list[str]:
    # Supports common shell-ish patterns like **/*.{js,jsx,ts,tsx}.
    if "{" not in pattern or "}" not in pattern:
        return [pattern]

    start = pattern.find("{")
    end = pattern.find("}", start)
    if start == -1 or end == -1 or end < start:
        return [pattern]

    before = pattern[:start]
    after = pattern[end + 1 :]
    options = [p.strip() for p in pattern[start + 1 : end].split(",") if p.strip()]
    if not options:
        return [pattern]

    return [before + option + after for option in options]


@tool
def get_workspace() -> str:
    """Return the active request-scoped workspace folder."""
    return str(active_workspace())


@tool
def ls(path: str = ".") -> str:
    """List files and folders under a workspace path."""
    try:
        target = _safe_path(path)

        if not target.exists():
            return f"Path does not exist: {target}"
        if not target.is_dir():
            return f"Not a directory: {target}"

        root = active_workspace()
        rows = []

        for item in sorted(
            target.iterdir(),
            key=lambda x: (
                not x.is_dir(),
                x.name.lower(),
            ),
        ):
            kind = "DIR " if item.is_dir() else "FILE"
            rows.append(
                f"{kind} {item.relative_to(root)}"
            )

        return "\n".join(rows[:200]) or "(empty)"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def glob_files(pattern: str) -> str:
    """Find files matching a glob pattern inside the workspace, for example '**/*.py' or 'routes/*.py'."""
    try:
        root = active_workspace()
        matches = []

        for expanded_pattern in _expand_brace_glob(pattern):
            for item in root.glob(expanded_pattern):
                if _is_skipped(item):
                    continue
                if item.is_file():
                    matches.append(str(item.relative_to(root)))

        matches = sorted(matches)[:200]
        return "\n".join(matches) or "No matching files."
    except Exception as e:
        return f"ERROR: {e}"


@tool
def grep(pattern: str, path: str = ".", max_matches: int = 80) -> str:
    """Search text files for a regex/text pattern under a workspace path. Returns file:line evidence."""
    try:
        target = _safe_path(path)

        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)

        files = [target] if target.is_file() else list(target.rglob("*"))
        rows = []

        for file in files:
            if len(rows) >= max_matches:
                break
            if not file.is_file() or _is_skipped(file):
                continue

            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    rel = file.relative_to(active_workspace())
                    rows.append(f"{rel}:{i}: {line.strip()[:260]}")
                    if len(rows) >= max_matches:
                        break

        return "\n".join(rows) or "No matches."
    except Exception as e:
        return f"ERROR: {e}"


@tool
def read_file(path: str, start_line: int = 1, max_lines: int = 120) -> str:
    """Read a file from the workspace with line numbers."""
    try:
        target = _safe_path(path)

        if not target.exists():
            return f"File does not exist: {target}"
        if not target.is_file():
            return f"Not a file: {target}"
        if _is_skipped(target):
            return f"Refusing to read skipped path: {target}"

        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(1, int(start_line))
        end = min(len(lines), start + max(1, int(max_lines)) - 1)

        out = []
        for line_no in range(start, end + 1):
            out.append(f"{line_no}: {lines[line_no - 1]}")

        return "\n".join(out) or "(empty file/range)"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def read_file_around(path: str, line: int, context: int = 30) -> str:
    """Read lines around a specific line number inside a workspace file."""
    try:
        target = _safe_path(path)

        if not target.exists():
            return f"File does not exist: {target}"
        if not target.is_file():
            return f"Not a file: {target}"
        if _is_skipped(target):
            return f"Refusing to read skipped path: {target}"

        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        center = max(1, int(line))
        radius = max(1, min(int(context), 80))

        start = max(1, center - radius)
        end = min(len(lines), center + radius)

        out = [
            f"# File: {target.relative_to(active_workspace())}",
            f"# Showing lines {start}-{end} around target line {center}",
            "",
        ]

        for line_no in range(start, end + 1):
            marker = ">>" if line_no == center else "  "
            out.append(f"{marker} {line_no}: {lines[line_no - 1]}")

        return "\n".join(out)
    except Exception as e:
        return f"ERROR: {e}"


def _ask_user_marker(
    question: str,
    options: list[dict],
    multi: bool = False,
) -> str:
    """LangGraph interaction marker handled by graph.py."""
    return "ASK_USER"


ask_user = StructuredTool.from_function(
    func=_ask_user_marker,
    name="ask_user",
    description=(
        "Ask the user one focused clarification question when missing or "
        "ambiguous information materially changes the correct source, "
        "target, tool, or required arguments. Provide 2-6 useful options. "
        "The user may also answer with free text. Calling this ends the "
        "current turn. Do not use this for risk confirmation."
    ),
)

ALL_TOOLS = [
    ask_user,
    get_workspace,
    ls,
    glob_files,
    grep,
    read_file,
    read_file_around,
]


TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def get_tools_for_scope(scope: str):
    from src.agent_runtime.tool_policy import allowed_tools_for_scope

    allowed = allowed_tools_for_scope(scope)
    return [
        tool_obj
        for tool_obj in ALL_TOOLS
        if tool_obj.name == "ask_user" or tool_obj.name in allowed
    ]
