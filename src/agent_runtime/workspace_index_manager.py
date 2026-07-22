from __future__ import annotations

import json
import hashlib
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.agent_runtime.context_vault_read_adapter import ContextVaultReadAdapter
from src.agent_runtime.knowledge_vault_index import build_structural_vault_index
from src.agent_runtime.knowledge_vault_semantic import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_URL,
    build_semantic_vault_index,
    get_semantic_vault_stats,
    search_semantic_vault_index,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOTS_FILE = PROJECT_ROOT / "config" / "local_roots.json"
WORKSPACE_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
WORKSPACE_EMBEDDING_URL = os.getenv(
    "KNOWLEDGE_EMBEDDING_URL",
    DEFAULT_EMBEDDING_URL,
)
WORKSPACE_TEXT_EXTENSIONS = frozenset(
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

_VALID_ROOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,127}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_root_id(value: str) -> str:
    root_id = str(value or "").strip()

    if not _VALID_ROOT_ID.fullmatch(root_id):
        raise ValueError(
            "root_id must be 2-128 characters and contain only "
            "letters, numbers, _, -, or ."
        )

    return root_id


def suggest_root_id(path: str) -> str:
    resolved = _canonical_workspace(path)
    name = resolved.name.strip().lower()
    slug = re.sub(r"[^a-z0-9_.-]+", "_", name)
    slug = slug.strip("_.-") or "workspace"
    identity = os.path.normcase(str(resolved))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    prefix = "workspace_"
    max_slug = 128 - len(prefix) - len(digest) - 1
    return normalize_root_id(f"{prefix}{slug[:max_slug]}_{digest}")


def _canonical_workspace(path: str) -> Path:
    from src.tool_execution import vet_workspace

    resolved = vet_workspace(str(path or ""))
    if not resolved:
        raise ValueError("Workspace path is not an allowed workspace directory.")
    return Path(resolved).resolve()


def _load_registry() -> dict[str, Any]:
    if not ROOTS_FILE.exists():
        return {"roots": []}

    raw = json.loads(
        ROOTS_FILE.read_text(encoding="utf-8")
    )

    if isinstance(raw, list):
        return {"roots": raw}

    if not isinstance(raw, dict):
        return {"roots": []}

    if isinstance(raw.get("roots"), list):
        return raw

    if isinstance(raw.get("local_roots"), list):
        return {
            **raw,
            "roots": list(raw["local_roots"]),
        }

    return {
        **raw,
        "roots": [],
    }


def _save_registry(data: dict[str, Any]) -> None:
    ROOTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = ROOTS_FILE.with_suffix(ROOTS_FILE.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(ROOTS_FILE)


def list_roots() -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in _load_registry().get("roots", [])
        if isinstance(item, dict)
    ]


def get_root(root_id: str) -> dict[str, Any] | None:
    root_id = normalize_root_id(root_id)

    for root in list_roots():
        if str(root.get("root_id") or root.get("id") or "") == root_id:
            return root

    return None


def find_root_by_path(path: str) -> dict[str, Any] | None:
    target = str(
        Path(path).expanduser().resolve()
    ).casefold()

    for root in list_roots():
        saved = str(root.get("path") or "").strip()

        if not saved:
            continue

        try:
            saved_resolved = str(
                Path(saved).expanduser().resolve()
            ).casefold()
        except Exception:
            continue

        if saved_resolved == target:
            return root

    return None


def register_root(
    root_id: str,
    path: str,
    *,
    display_name: str | None = None,
) -> dict[str, Any]:
    root_id = normalize_root_id(root_id)
    resolved = _canonical_workspace(path)
    definitive_root_id = suggest_root_id(str(resolved))
    if root_id != definitive_root_id:
        raise ValueError(
            f"Root ID must be the definitive ID for this workspace: {definitive_root_id}"
        )

    data = _load_registry()

    roots = [
        dict(item)
        for item in data.get("roots", [])
        if isinstance(item, dict)
    ]

    previous = next(
        (
            item
            for item in roots
            if str(item.get("root_id") or item.get("id") or "") == root_id
        ),
        None,
    )

    resolved_str = str(resolved)

    record = {
        "id": root_id,
        "provider_type": "local_dir",
        "display_name": (
            display_name
            or (
                previous.get("display_name")
                if previous
                else None
            )
            or resolved.name
            or root_id
        ),
        "allowed_capabilities": [
            "context.list",
            "context.read",
        ],
        "safety_scope": "read_only_sandbox",
        "enabled": True,
        "local_path": resolved_str,

        # Index-manager metadata.
        "root_id": root_id,
        "path": resolved_str,
        "index_namespace": root_id,
        "associated_index_id": root_id,
        "embedding_model": WORKSPACE_EMBEDDING_MODEL,
        "last_indexed": (
            previous.get("last_indexed")
            if previous
            else None
        ),
        "last_queried": (
            previous.get("last_queried")
            if previous
            else None
        ),
    }

    roots = [
        item
        for item in roots
        if (
            str(item.get("root_id") or item.get("id") or "") != root_id
            and str(item.get("path") or item.get("local_path") or "").casefold()
            != resolved_str.casefold()
        )
    ]

    roots.append(record)
    data["roots"] = roots

    if "local_roots" in data:
        data["local_roots"] = roots

    _save_registry(data)

    return record


def index_root(
    root_id: str,
    *,
    rebuild: bool = True,
) -> dict[str, Any]:
    root_id = normalize_root_id(root_id)

    root = get_root(root_id)

    if not root:
        raise ValueError(
            f"Unknown root_id: {root_id}"
        )

    path = _canonical_workspace(
        str(root.get("path") or root.get("local_path") or "")
    )

    adapter = ContextVaultReadAdapter(
        root_id=root_id,
        config_path=ROOTS_FILE,
    )
    # Probe the provider boundary explicitly. The generic structural builder is
    # intentionally tolerant of unreadable subfolders, but an invalid root or
    # registry mismatch must never be reported as a successful empty index.
    adapter.list_dir("")
    structural_index = build_structural_vault_index(
        adapter=adapter,
        max_notes=1000,
        cache_path=None,
        file_extensions=WORKSPACE_TEXT_EXTENSIONS,
        strict=False,
    )

    if structural_index.note_count == 0:
        raise ValueError("The workspace contains no supported text files to index.")

    result = build_semantic_vault_index(
        adapter=adapter,
        structural_index=structural_index,
        index_namespace=root_id,
        rebuild=bool(rebuild),
        embedding_url=WORKSPACE_EMBEDDING_URL,
        embedding_model=WORKSPACE_EMBEDDING_MODEL,
        strict_reads=True,
    )

    if result.note_count == 0 or result.chunk_count == 0:
        raise RuntimeError("Workspace indexing completed without producing searchable chunks.")

    data = _load_registry()

    for item in data.get("roots", []):
        if (
            isinstance(item, dict)
            and str(item.get("root_id") or "") == root_id
        ):
            item["last_indexed"] = _now_iso()
            item["index_namespace"] = root_id
            item["embedding_model"] = result.embedding_model

    if "local_roots" in data:
        data["local_roots"] = data["roots"]

    _save_registry(data)

    return {
        "root_id": root_id,
        "path": str(path),
        "stats": (
            result.to_dict()
            if hasattr(result, "to_dict")
            else result
        ),
    }


def search_root(
    root_id: str,
    query: str,
    *,
    k: int = 8,
) -> list[dict[str, Any]]:
    root_id = normalize_root_id(root_id)

    if not get_root(root_id):
        raise ValueError(
            f"Unknown root_id: {root_id}"
        )

    clean_query = str(query or "").strip()
    if not clean_query:
        raise ValueError("A non-empty search query is required.")

    results = search_semantic_vault_index(
        clean_query,
        k=k,
        index_namespace=root_id,
        embedding_url=WORKSPACE_EMBEDDING_URL,
        embedding_model=WORKSPACE_EMBEDDING_MODEL,
    )

    data = _load_registry()

    for item in data.get("roots", []):
        if (
            isinstance(item, dict)
            and str(item.get("root_id") or "") == root_id
        ):
            item["last_queried"] = _now_iso()

    if "local_roots" in data:
        data["local_roots"] = data["roots"]

    _save_registry(data)

    return results


def flush_root_index(root_id: str) -> dict[str, Any]:
    root_id = normalize_root_id(root_id)
    if not get_root(root_id):
        raise ValueError(f"Unknown root_id: {root_id}")

    from src.agent_runtime import knowledge_vault_semantic as semantic

    collection = semantic._collection()

    existing = collection.get(
        where={
            "index_namespace": root_id,
        }
    )

    ids = list(existing.get("ids") or [])

    if ids:
        collection.delete(ids=ids)

    return {
        "root_id": root_id,
        "deleted_chunks": len(ids),
    }


def remove_root(
    root_id: str,
    *,
    flush_index: bool = False,
) -> dict[str, Any]:
    root_id = normalize_root_id(root_id)
    if not get_root(root_id):
        raise ValueError(f"Unknown root_id: {root_id}")

    flush_result = None

    if flush_index:
        flush_result = flush_root_index(root_id)

    data = _load_registry()

    roots = [
        dict(item)
        for item in data.get("roots", [])
        if (
            isinstance(item, dict)
            and str(item.get("root_id") or "") != root_id
        )
    ]

    data["roots"] = roots

    if "local_roots" in data:
        data["local_roots"] = roots

    _save_registry(data)

    return {
        "root_id": root_id,
        "removed": True,
        "flush": flush_result,
    }


def root_stats(root_id: str) -> dict[str, Any]:
    root_id = normalize_root_id(root_id)
    root = get_root(root_id)
    if not root:
        raise ValueError(f"Unknown root_id: {root_id}")

    return {
        "root_id": root_id,
        "root": root,
        "index": get_semantic_vault_stats(
            index_namespace=root_id,
            embedding_url=WORKSPACE_EMBEDDING_URL,
            embedding_model=WORKSPACE_EMBEDDING_MODEL,
        ),
    }


def stale_roots(
    *,
    older_than_days: int = 30,
) -> list[dict[str, Any]]:
    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(
            days=max(
                1,
                int(older_than_days),
            )
        )
    )

    stale = []

    for root in list_roots():
        stamp = (
            root.get("last_queried")
            or root.get("last_indexed")
        )

        if not stamp:
            stale.append(root)
            continue

        try:
            parsed = datetime.fromisoformat(
                str(stamp).replace(
                    "Z",
                    "+00:00",
                )
            )

            if parsed < cutoff:
                stale.append(root)

        except ValueError:
            stale.append(root)

    return stale


__all__ = [
    "PROJECT_ROOT",
    "ROOTS_FILE",
    "WORKSPACE_EMBEDDING_MODEL",
    "WORKSPACE_EMBEDDING_URL",
    "WORKSPACE_TEXT_EXTENSIONS",
    "normalize_root_id",
    "suggest_root_id",
    "list_roots",
    "get_root",
    "find_root_by_path",
    "register_root",
    "index_root",
    "search_root",
    "flush_root_index",
    "remove_root",
    "root_stats",
    "stale_roots",
]
