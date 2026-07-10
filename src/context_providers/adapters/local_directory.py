"""Read-only local directory context adapter.

This adapter is intentionally isolated from live runtime wiring.

It provides practical list/read/search operations for AuthorizedRoot entries
whose provider_type is local_dir or obsidian_vault.

Safety goals:
- resolve all targets inside the authorized root
- reject traversal and absolute target selectors
- do not follow reads outside root through symlinks
- only read text-like files with bounded size
- no write/delete/edit operations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base import BaseContextAdapter, CapabilityNotSupported
from ..roots import AuthorizedRoot


_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".scss",
        ".xml",
        ".csv",
        ".ini",
        ".cfg",
        ".env.example",
        ".gitignore",
    }
)

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "venv",
        ".venv",
        "dist",
        "build",
    }
)


@dataclass(frozen=True)
class LocalDirectoryAdapter(BaseContextAdapter):
    """Read-only adapter for local filesystem roots."""

    max_file_bytes: int = 512_000
    max_list_items: int = 500
    max_search_results: int = 50

    provider_type: str = "local_dir"
    supported_capabilities: frozenset[str] = frozenset(
        {
            "context.list",
            "context.read",
            "context.search",
            "context.status",
        }
    )

    def status(self, root: AuthorizedRoot) -> dict[str, Any]:
        self.require("context.status")
        base = self._resolve_root(root)
        return {
            "root_id": root.id,
            "provider_type": root.provider_type,
            "exists": base.exists(),
            "is_dir": base.is_dir(),
            "readable": base.exists() and base.is_dir(),
        }

    def list(self, root: AuthorizedRoot, selector: str | None = None) -> list[dict[str, Any]]:
        self.require("context.list")
        base = self._resolve_root(root)
        start = self._resolve_target(base, selector or "")

        if not start.exists():
            raise FileNotFoundError(f"Path not found: {selector or '.'}")
        if not start.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {selector or '.'}")

        items: list[dict[str, Any]] = []
        for child in sorted(start.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if len(items) >= self.max_list_items:
                break
            if self._should_skip(child):
                continue

            rel = self._relative(base, child)
            items.append(
                {
                    "path": rel,
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )

        return items

    def read(self, root: AuthorizedRoot, selector: str) -> str:
        self.require("context.read")
        base = self._resolve_root(root)
        target = self._resolve_target(base, selector)

        if not target.exists():
            raise FileNotFoundError(f"Path not found: {selector}")
        if not target.is_file():
            raise IsADirectoryError(f"Path is not a file: {selector}")
        if target.stat().st_size > self.max_file_bytes:
            raise ValueError(f"File is too large to read through context adapter: {selector}")
        if not self._is_text_like(target):
            raise ValueError(f"File extension is not allowed for text read: {selector}")

        return target.read_text(encoding="utf-8", errors="replace")

    def search(self, root: AuthorizedRoot, query: str) -> list[dict[str, Any]]:
        self.require("context.search")
        needle = str(query or "").strip().lower()
        if not needle:
            return []

        base = self._resolve_root(root)
        results: list[dict[str, Any]] = []

        for path in self._walk_text_files(base):
            if len(results) >= self.max_search_results:
                break

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            lowered = text.lower()
            idx = lowered.find(needle)
            if idx < 0:
                continue

            start = max(0, idx - 80)
            end = min(len(text), idx + len(needle) + 160)
            snippet = text[start:end].replace("\r\n", "\n").replace("\r", "\n")

            results.append(
                {
                    "path": self._relative(base, path),
                    "snippet": snippet,
                }
            )

        return results

    def _resolve_root(self, root: AuthorizedRoot) -> Path:
        if root.provider_type not in {"local_dir", "obsidian_vault"}:
            raise ValueError(
                f"LocalDirectoryAdapter cannot handle provider_type={root.provider_type!r}"
            )

        base = root.resolve_path()
        if base is None:
            raise ValueError(f"Authorized root {root.id!r} has no resolved local path.")
        if not base.exists():
            raise FileNotFoundError(f"Authorized root path does not exist: {root.id}")
        if not base.is_dir():
            raise NotADirectoryError(f"Authorized root path is not a directory: {root.id}")

        return base.resolve()

    def _resolve_target(self, base: Path, selector: str) -> Path:
        raw = str(selector or "").strip().replace("\\", "/")

        if raw in {"", "."}:
            return base

        candidate = Path(raw)
        if candidate.is_absolute():
            raise ValueError("Absolute selectors are not allowed.")

        if any(part in {"..", ""} for part in candidate.parts):
            raise ValueError("Traversal selectors are not allowed.")

        resolved = (base / candidate).resolve()

        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError("Resolved path escaped authorized root.") from exc

        return resolved

    def _walk_text_files(self, base: Path):
        for path in base.rglob("*"):
            if self._should_skip(path):
                if path.is_dir():
                    continue
                continue
            if path.is_file() and self._is_text_like(path):
                try:
                    if path.stat().st_size <= self.max_file_bytes:
                        yield path
                except OSError:
                    continue

    def _should_skip(self, path: Path) -> bool:
        return any(part in _SKIP_DIRS for part in path.parts)

    def _is_text_like(self, path: Path) -> bool:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in {".gitignore", ".env.example"}:
            return True
        return suffix in _TEXT_EXTENSIONS

    def _relative(self, base: Path, path: Path) -> str:
        return str(path.resolve().relative_to(base)).replace("\\", "/")
