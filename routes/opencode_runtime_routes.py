from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.opencode_runtime import OpenCodeRuntimeService


router = APIRouter(
    prefix="/api/opencode/runtime",
    tags=["opencode-runtime"],
)
_service = OpenCodeRuntimeService()


class OpenCodeRuntimeRequest(BaseModel):
    message: str = Field(min_length=1)
    odysseus_session_id: str = Field(min_length=1)
    title: str = ""
    permission_mode: str = "ask"


class OpenCodeRestartRequest(BaseModel):
    model_id: str = Field(min_length=1)
    workspace: str = Field(min_length=1)


class OpenCodeLogTerminalRequest(BaseModel):
    odysseus_session_id: str = ""


class OpenCodePermissionModeRequest(BaseModel):
    mode: str = Field(min_length=1)


class OpenCodePermissionReplyRequest(BaseModel):
    odysseus_session_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    decision: str = Field(min_length=1)


@router.get("/health")
async def opencode_runtime_health():
    try:
        return await _service.health()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@router.post("/stream")
async def stream_opencode_runtime(request: OpenCodeRuntimeRequest):
    return StreamingResponse(
        _service.stream(
            message=request.message,
            odysseus_session_id=request.odysseus_session_id,
            title=request.title,
            permission_mode=request.permission_mode,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/server/restart")
async def restart_opencode_server(request: OpenCodeRestartRequest):
    try:
        value = await _service.restart_server(
            model_id=request.model_id,
            workspace=request.workspace,
        )
        return {"ok": True, **value}
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@router.post("/server/stop")
async def stop_opencode_server():
    try:
        return await _service.stop_server()
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@router.post("/server/log-terminal")
async def open_opencode_log_terminal(
    request: OpenCodeLogTerminalRequest | None = None,
):
    try:
        return await _service.open_log_terminal(
            request.odysseus_session_id if request else ""
        )
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@router.get("/permissions/{odysseus_session_id}")
async def get_opencode_permission_state(
    odysseus_session_id: str,
):
    try:
        return _service.permission_state(odysseus_session_id)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.put("/permissions/{odysseus_session_id}/mode")
async def set_opencode_permission_mode(
    odysseus_session_id: str,
    request: OpenCodePermissionModeRequest,
):
    try:
        return _service.set_permission_mode(
            odysseus_session_id,
            request.mode,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post("/permissions/reply")
async def reply_to_opencode_permission(
    request: OpenCodePermissionReplyRequest,
):
    try:
        return await _service.resolve_permission_request(
            odysseus_session_id=request.odysseus_session_id,
            request_id=request.request_id,
            decision=request.decision,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.delete("/permissions/{odysseus_session_id}/approvals")
async def clear_opencode_permission_approvals(
    odysseus_session_id: str,
):
    try:
        return _service.clear_permission_approvals(
            odysseus_session_id
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post("/cancel/{run_id}")
async def cancel_opencode_runtime(run_id: str):
    cancelled = await _service.cancel(run_id)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail="No active OpenCode run with that id.",
        )
    return {"ok": True, "run_id": run_id}


@router.delete("/conversation/{odysseus_session_id}")
async def forget_opencode_conversation(
    odysseus_session_id: str,
):
    deleted = await _service.forget_conversation(
        odysseus_session_id
    )
    return {
        "ok": True,
        "deleted": deleted,
        "odysseus_session_id": odysseus_session_id,
    }
