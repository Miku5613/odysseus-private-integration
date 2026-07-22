from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.middleware import require_admin
from src.agent_runtime.workspace_index_manager import (
    flush_root_index,
    index_root,
    list_roots,
    register_root,
    remove_root,
    root_stats,
    stale_roots,
    suggest_root_id,
)


class RegisterRootRequest(BaseModel):
    root_id: str
    path: str
    display_name: str | None = None


class IndexRootRequest(BaseModel):
    root_id: str


logger = logging.getLogger(__name__)


def _service_failure(action: str, exc: Exception) -> HTTPException:
    logger.warning("Workspace index %s failed: %s", action, exc)
    return HTTPException(
        status_code=503,
        detail=f"Workspace index {action} failed: {exc}",
    )


def setup_index_routes() -> APIRouter:
    router = APIRouter(
        prefix="/api/indexes",
        tags=["indexes"],
    )

    @router.get("")
    async def get_indexes(request: Request):
        require_admin(request)

        return {
            "roots": list_roots(),
        }

    @router.get("/suggest-root-id")
    async def get_suggested_root_id(
        path: str,
        request: Request,
    ):
        require_admin(request)

        try:
            return {
                "root_id": suggest_root_id(path),
            }
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @router.post("/register")
    async def register_index_root(
        payload: RegisterRootRequest,
        request: Request,
    ):
        require_admin(request)

        try:
            return register_root(
                payload.root_id,
                payload.path,
                display_name=payload.display_name,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    @router.post("/index")
    async def build_index(
        payload: IndexRootRequest,
        request: Request,
    ):
        require_admin(request)

        try:
            return await asyncio.to_thread(
                index_root,
                payload.root_id,
                rebuild=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise _service_failure("build", exc) from exc

    @router.post("/reindex")
    async def rebuild_index(
        payload: IndexRootRequest,
        request: Request,
    ):
        require_admin(request)

        try:
            return await asyncio.to_thread(
                index_root,
                payload.root_id,
                rebuild=True,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise _service_failure("rebuild", exc) from exc

    @router.get("/stale")
    async def get_stale_indexes(
        request: Request,
        older_than_days: int = 30,
    ):
        require_admin(request)

        return {
            "roots": stale_roots(
                older_than_days=older_than_days,
            ),
        }

    @router.get("/{root_id}/stats")
    async def get_index_stats(
        root_id: str,
        request: Request,
    ):
        require_admin(request)

        try:
            return await asyncio.to_thread(root_stats, root_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise _service_failure("stats", exc) from exc

    @router.delete("/{root_id}/index")
    async def flush_index(
        root_id: str,
        request: Request,
    ):
        require_admin(request)

        try:
            return await asyncio.to_thread(flush_root_index, root_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise _service_failure("flush", exc) from exc

    @router.delete("/{root_id}")
    async def delete_root(
        root_id: str,
        request: Request,
        flush_index: bool = False,
    ):
        require_admin(request)

        try:
            return await asyncio.to_thread(
                remove_root,
                root_id,
                flush_index=flush_index,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise _service_failure("remove", exc) from exc

    return router
