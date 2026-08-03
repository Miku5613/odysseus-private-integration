"""Installation-time availability checks for optional agent runtimes."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
from typing import Any


_LANGGRAPH_MODULES = (
    ("langgraph", "LangGraph"),
    ("langchain_core", "LangChain Core"),
    ("langgraph.checkpoint.sqlite", "LangGraph SQLite checkpoints"),
)


def _langgraph_status() -> dict[str, Any]:
    missing: list[str] = []
    error = ""
    for module, label in _LANGGRAPH_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(label)
            if not error:
                error = f"{type(exc).__name__}: {exc}"
    installed = not missing
    return {
        "available": installed,
        "installed": installed,
        "status": "available" if installed else "not_installed",
        "label": "LangGraph",
        "mandatory": False,
        "missing": missing,
        "detail": "" if installed else error,
    }


def _opencode_executable() -> str:
    configured = os.getenv("ODYSSEUS_OPENCODE_EXECUTABLE", "opencode").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(configured)
        if resolved:
            return resolved
    for name in ("opencode.cmd", "opencode.exe", "opencode"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return ""


def _opencode_status() -> dict[str, Any]:
    executable = _opencode_executable()
    installed = bool(executable)
    return {
        "available": installed,
        "installed": installed,
        "status": "available" if installed else "not_installed",
        "label": "OpenCode",
        "mandatory": False,
        "missing": [] if installed else ["OpenCode CLI"],
        "detail": executable if installed else "OpenCode executable was not found.",
    }


def runtime_availability() -> dict[str, dict[str, Any]]:
    """Return runtime choices without contacting model/provider services."""

    return {
        "legacy": {
            "available": True,
            "installed": True,
            "status": "available",
            "label": "Legacy",
            "mandatory": True,
            "missing": [],
            "detail": "Built into Odysseus.",
        },
        "langgraph": _langgraph_status(),
        "opencode": _opencode_status(),
    }


def runtime_is_available(runtime: str) -> bool:
    key = str(runtime or "").strip().lower()
    if key == "legacy":
        return True
    value = runtime_availability().get(key)
    return bool(value and value.get("available"))


__all__ = ["runtime_availability", "runtime_is_available"]
