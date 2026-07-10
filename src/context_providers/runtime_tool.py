"""Runtime-facing read-only context tool wrapper.

This module is intentionally tool-shaped, but not yet wired into LangGraph or
the live agent runtime.

It lets the runtime compose authorized-root operations without requiring the
user to manually know exact file paths:

- roots: list available authorized roots
- list: list direct children inside a root
- search: recursively search one root or all roots
- read: read one exact file selector
- search_read: search first, then read top matching files safely

The boundary remains:
authorized roots + allowed capabilities + read-only adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import build_default_registry
from .roots import AuthorizedRoot, load_authorized_roots, verify_authorized_root


DEFAULT_CONFIG_PATH = Path("config/local_roots.json")


class RuntimeContextToolError(RuntimeError):
    """Raised when a runtime context tool request is invalid."""


@dataclass(frozen=True)
class RuntimeContextToolLimits:
    max_search_results: int = 8
    max_read_files: int = 3
    max_chars_per_file: int = 6000
    max_total_chars: int = 18000


def context_tool_call(
    *,
    operation: str,
    root_id: str | None = None,
    selector: str | None = None,
    query: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    limits: RuntimeContextToolLimits | None = None,
) -> dict[str, Any]:
    """Run one read-only context operation.

    This function is the future adapter between the agent runtime/tool layer and
    the context provider stack. It returns JSON-safe dictionaries only.
    """

    limits = limits or RuntimeContextToolLimits()
    operation = str(operation or "").strip().lower()

    roots = load_authorized_roots(Path(config_path))
    registry = build_default_registry()

    if operation == "roots":
        return {
            "ok": True,
            "operation": "roots",
            "roots": [
                {
                    **root.public_summary(),
                    "verification": verify_authorized_root(root),
                }
                for root in roots
            ],
        }

    enabled_roots = [root for root in roots if root.enabled]

    if operation == "list":
        root = _require_root(enabled_roots, root_id)
        items = registry.list(root, selector=selector or ".")
        return {
            "ok": True,
            "operation": "list",
            "root": root.public_summary(),
            "selector": selector or ".",
            "items": items,
            "count": len(items),
        }

    if operation == "read":
        root = _require_root(enabled_roots, root_id)
        if not selector:
            raise RuntimeContextToolError("read requires selector.")

        text = registry.read(root, selector)
        clipped = _clip_text(text, limits.max_chars_per_file)
        return {
            "ok": True,
            "operation": "read",
            "root": root.public_summary(),
            "selector": selector,
            "text": clipped["text"],
            "characters": len(text),
            "truncated": clipped["truncated"],
        }

    if operation == "search":
        if not query or not query.strip():
            raise RuntimeContextToolError("search requires query.")

        target_roots = _target_roots(enabled_roots, root_id)
        results = []

        for root in target_roots:
            for item in registry.search(root, query):
                results.append(
                    {
                        "root_id": root.id,
                        "root_display_name": root.display_name,
                        "provider_type": root.provider_type,
                        **item,
                    }
                )
                if len(results) >= limits.max_search_results:
                    break
            if len(results) >= limits.max_search_results:
                break

        return {
            "ok": True,
            "operation": "search",
            "query": query,
            "root_id": root_id,
            "results": results,
            "count": len(results),
        }

    if operation == "search_read":
        if not query or not query.strip():
            raise RuntimeContextToolError("search_read requires query.")

        target_roots = _target_roots(enabled_roots, root_id)
        matches = []

        for root in target_roots:
            for item in registry.search(root, query):
                matches.append(
                    {
                        "root": root,
                        "path": item["path"],
                        "snippet": item.get("snippet", ""),
                    }
                )
                if len(matches) >= limits.max_search_results:
                    break
            if len(matches) >= limits.max_search_results:
                break

        documents = []
        total_chars = 0

        for match in matches[: limits.max_read_files]:
            if total_chars >= limits.max_total_chars:
                break

            root = match["root"]
            path = match["path"]

            try:
                text = registry.read(root, path)
            except Exception as exc:
                documents.append(
                    {
                        "root_id": root.id,
                        "path": path,
                        "ok": False,
                        "error": exc.__class__.__name__,
                        "detail": str(exc),
                        "snippet": match["snippet"],
                    }
                )
                continue

            remaining = max(0, limits.max_total_chars - total_chars)
            per_file_limit = min(limits.max_chars_per_file, remaining)
            clipped = _clip_text(text, per_file_limit)
            total_chars += len(clipped["text"])

            documents.append(
                {
                    "root_id": root.id,
                    "root_display_name": root.display_name,
                    "provider_type": root.provider_type,
                    "path": path,
                    "ok": True,
                    "text": clipped["text"],
                    "characters": len(text),
                    "truncated": clipped["truncated"],
                    "snippet": match["snippet"],
                }
            )

        return {
            "ok": True,
            "operation": "search_read",
            "query": query,
            "root_id": root_id,
            "match_count": len(matches),
            "document_count": len(documents),
            "documents": documents,
        }

    raise RuntimeContextToolError(f"Unsupported context operation: {operation}")


def _require_root(roots: list[AuthorizedRoot], root_id: str | None) -> AuthorizedRoot:
    if not root_id:
        raise RuntimeContextToolError("root_id is required for this operation.")

    for root in roots:
        if root.id == root_id:
            return root

    raise RuntimeContextToolError(f"Enabled root not found: {root_id}")


def _target_roots(roots: list[AuthorizedRoot], root_id: str | None) -> list[AuthorizedRoot]:
    if root_id:
        return [_require_root(roots, root_id)]
    return roots


def _clip_text(text: str, limit: int) -> dict[str, Any]:
    if limit < 0:
        limit = 0

    if len(text) <= limit:
        return {
            "text": text,
            "truncated": False,
        }

    return {
        "text": text[:limit],
        "truncated": True,
    }


__all__ = [
    "RuntimeContextToolError",
    "RuntimeContextToolLimits",
    "context_tool_call",
]
