"""Read-only context provider API routes.

These routes expose configured context roots through explicit read-only calls.

They do not:
- inject context into chat
- start LangGraph/tool routing
- write files
- index content
- mutate Workspace/Atlas/Obsidian state
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.context_providers import (
    CapabilityNotSupported,
    ProviderNotRegistered,
    build_default_registry,
    load_authorized_roots,
    verify_authorized_root,
)

router = APIRouter(prefix="/api/context", tags=["context"])


class ContextListRequest(BaseModel):
    root_id: str = Field(..., min_length=1)
    selector: str | None = None


class ContextReadRequest(BaseModel):
    root_id: str = Field(..., min_length=1)
    selector: str = Field(..., min_length=1)


class ContextSearchRequest(BaseModel):
    root_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


@router.get("/roots")
def list_context_roots() -> dict[str, Any]:
    roots = _load_roots()
    registry = build_default_registry()

    payload_roots = []
    for root in roots:
        adapter_status = _adapter_status(registry, root)
        payload_roots.append(
            {
                **root.public_summary(),
                "verification": verify_authorized_root(root),
                "adapter_status": adapter_status,
            }
        )

    return {
        "roots": payload_roots,
        "root_count": len(payload_roots),
        "registry": registry.public_status(),
    }


@router.post("/list")
def list_context_root(request: ContextListRequest) -> dict[str, Any]:
    root = _get_enabled_root(request.root_id)
    registry = build_default_registry()

    try:
        items = registry.list(root, selector=request.selector)
    except Exception as exc:
        _raise_context_error(exc)

    return {
        "root": root.public_summary(),
        "selector": request.selector or ".",
        "items": items,
        "count": len(items),
    }


@router.post("/read")
def read_context_root_file(request: ContextReadRequest) -> dict[str, Any]:
    root = _get_enabled_root(request.root_id)
    registry = build_default_registry()

    try:
        text = registry.read(root, request.selector)
    except Exception as exc:
        _raise_context_error(exc)

    return {
        "root": root.public_summary(),
        "selector": request.selector,
        "text": text,
        "characters": len(text),
    }


@router.post("/search")
def search_context_root(request: ContextSearchRequest) -> dict[str, Any]:
    root = _get_enabled_root(request.root_id)
    registry = build_default_registry()

    try:
        results = registry.search(root, request.query)
    except Exception as exc:
        _raise_context_error(exc)

    return {
        "root": root.public_summary(),
        "query": request.query,
        "results": results,
        "count": len(results),
    }


def _config_path() -> Path:
    return Path(os.environ.get("ODYSSEUS_CONTEXT_ROOTS_CONFIG", "config/local_roots.json"))


def _load_roots():
    return load_authorized_roots(_config_path())


def _get_enabled_root(root_id: str):
    for root in _load_roots():
        if root.id == root_id:
            if not root.enabled:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "root_disabled",
                        "root_id": root_id,
                    },
                )
            return root

    raise HTTPException(
        status_code=404,
        detail={
            "error": "root_not_found",
            "root_id": root_id,
        },
    )


def _adapter_status(registry, root) -> dict[str, Any]:
    try:
        adapter = registry.adapter_for(root)
    except ProviderNotRegistered as exc:
        return {
            "status": "not_registered",
            "detail": str(exc),
        }

    unsupported = [
        capability
        for capability in root.allowed_capabilities
        if not adapter.supports(capability)
    ]

    if unsupported:
        return {
            "status": "unsupported_capabilities",
            "adapter": adapter.__class__.__name__,
            "unsupported": unsupported,
        }

    return {
        "status": "ok",
        "adapter": adapter.__class__.__name__,
    }


def _raise_context_error(exc: Exception) -> None:
    if isinstance(exc, CapabilityNotSupported):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "capability_not_supported",
                "detail": str(exc),
            },
        ) from exc

    if isinstance(exc, ProviderNotRegistered):
        raise HTTPException(
            status_code=501,
            detail={
                "error": "provider_not_registered",
                "detail": str(exc),
            },
        ) from exc

    if isinstance(exc, FileNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "detail": str(exc),
            },
        ) from exc

    if isinstance(exc, (ValueError, IsADirectoryError, NotADirectoryError)):
        raise HTTPException(
            status_code=400,
            detail={
                "error": exc.__class__.__name__,
                "detail": str(exc),
            },
        ) from exc

    raise HTTPException(
        status_code=500,
        detail={
            "error": exc.__class__.__name__,
            "detail": str(exc),
        },
    ) from exc
