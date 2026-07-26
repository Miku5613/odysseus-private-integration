from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.models import (
    ChatMessage,
    get_session_manager_instance,
)
from src.auth_helpers import effective_user
from src.opencode_runtime import OpenCodeRuntimeService
from src.opencode_runtime.model_registry import (
    resolve_model_selection,
)


router = APIRouter(
    prefix="/api/opencode/runtime",
    tags=["opencode-runtime"],
)
def _session_manager():
    manager = get_session_manager_instance()
    if manager is None:
        raise RuntimeError("Odysseus session manager is unavailable.")
    return manager


def _verify_opencode_session_owner(
    request: Request,
    session_id: str,
) -> None:
    manager = _session_manager()
    try:
        session = manager.get_session(str(session_id or "").strip())
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        ) from exc

    user = effective_user(request)
    owner = str(getattr(session, "owner", "") or "").strip()
    if owner and owner != str(user or "").strip():
        raise HTTPException(
            status_code=403,
            detail="Session does not belong to the current user.",
        )


def _persist_transcript_message(
    session_id: str,
    role: str,
    content: str,
    metadata: dict[str, object],
) -> None:
    manager = _session_manager()
    session = manager.get_session(str(session_id or "").strip())
    normalized_role = str(role or "").strip().lower()
    normalized_content = str(content or "")

    if normalized_role not in {"user", "assistant"}:
        raise ValueError("Unsupported transcript role.")
    if not normalized_content.strip():
        raise ValueError("Transcript content must not be empty.")

    message = ChatMessage(
        role=normalized_role,
        content=normalized_content,
        metadata=dict(metadata or {}),
    )
    before_count = len(session.history)
    manager.add_message(session.id, message)

    if not (message.metadata or {}).get("_db_id"):
        if (
            len(session.history) == before_count + 1
            and session.history[-1] is message
        ):
            session.history.pop()
            session._history = session.history
            session.message_count = len(session.history)
        raise RuntimeError("Odysseus transcript persistence did not commit.")


def _resolve_persisted_native_session(
    session_id: str,
) -> dict[str, str] | None:
    manager = _session_manager()
    session = manager.get_session(str(session_id or "").strip())

    for message in reversed(session.history):
        metadata = (
            message.metadata
            if isinstance(message.metadata, dict)
            else {}
        )
        if str(metadata.get("runtime") or "") != "opencode":
            continue

        native_session_id = str(
            metadata.get("opencode_session_id") or ""
        ).strip()
        if not native_session_id:
            continue

        return {
            "opencode_session_id": native_session_id,
            "provider_id": str(
                metadata.get("provider_id") or ""
            ).strip(),
            "model_id": str(
                metadata.get("model_id") or ""
            ).strip(),
            "workspace": str(
                metadata.get("workspace") or ""
            ).strip(),
        }

    return None


_service = OpenCodeRuntimeService(
    transcript_persister=_persist_transcript_message,
    session_resolver=_resolve_persisted_native_session,
)


class OpenCodeRuntimeRequest(BaseModel):
    message: str = Field(min_length=1)
    odysseus_session_id: str = Field(min_length=1)
    title: str = ""
    permission_mode: str = "ask"


class OpenCodeRestartRequest(BaseModel):
    endpoint_id: str = Field(min_length=1)
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
async def stream_opencode_runtime(
    http_request: Request,
    request: OpenCodeRuntimeRequest,
):
    _verify_opencode_session_owner(
        http_request,
        request.odysseus_session_id,
    )
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
        selection = resolve_model_selection(
            endpoint_id=request.endpoint_id,
            model_id=request.model_id,
        )
        value = await _service.restart_server(
            endpoint_id=selection.endpoint_id,
            provider_id=selection.provider_id,
            provider_name=selection.provider_name,
            provider_base_url=selection.base_url,
            provider_models=selection.models,
            provider_api_key=selection.api_key,
            model_id=selection.model_id,
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
