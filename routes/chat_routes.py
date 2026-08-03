"""Chat routes - /api/chat, /api/chat_stream, /api/inject_context, /api/search."""

import asyncio
import json
import os
from pathlib import Path
from functools import partial
import re
import time
import logging
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, List, Optional

from fastapi import APIRouter, Request, HTTPException, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from core.models import ChatMessage
from src.request_models import ChatRequest
from src.llm_core import llm_call_async, stream_llm, stream_llm_with_fallback
from src.agent_loop import stream_agent_loop
from src.agent_runtime.live_supervisor import stream_general_supervisor_sse
from src.agent_runtime.multi_graph_coordinator import (
    stream_multi_graph_coordinator_sse,
)
from src.agent_runtime.run_lifecycle import LangGraphRunLifecycle
from src.agent_runtime.durable_checkpoint import (
    CheckpointUnavailableError,
    get_default_checkpoint_store,
)
from src.agent_runtime.research_worker import (
    cancel_owned_research_jobs,
    reconcile_owned_research_jobs,
)
from src.agent_runtime.step_resume import resume_interrupted_step
from src.agent_runtime.general_bridge import (
    general_runtime_enabled,
    general_runtime_supports,
    stream_general_runtime_legacy_sse,
)
from src.agent_runtime.knowledge_chat_bridge import run_knowledge_chat_turn, should_intercept_knowledge_chat
from src.agent_runtime.knowledge_llm_response_bridge import knowledge_llm_config_from_runtime_locals
from src.agent_runtime.knowledge_semantic_chat_bridge import (
    is_semantic_vault_search_request,
    run_knowledge_semantic_chat_turn,
    should_intercept_workspace_knowledge,
)
from src.agent_runtime.runtime_router import (
    route_agent_runtime,
    explain_runtime_route,
    is_casual_conversation_request,
)
from src.agent_runtime.agent_admission import assess_explicit_agent_admission
from src.agent_runtime.mutation_request_policy import evaluate_mutation_request

def _runtime_trace_sse_frame(message: str, runtime_decision):
    try:
        trace_text = explain_runtime_route(message)
        status = runtime_decision.to_status()
        call_id = f"runtime-decision-{status.get('intent', 'unknown')}"

        start_payload = {
            "type": "tool_start",
            "id": call_id,
            "call_id": call_id,
            "tool": "runtime_decision",
            "command": "Runtime decision",
        }

        output_payload = {
            "type": "tool_output",
            "id": call_id,
            "call_id": call_id,
            "tool": "runtime_decision",
            "command": "Runtime decision",
            "output": trace_text,
            "exit_code": 0,
            "runtime_trace": trace_text,
            "runtime_decision": status,
        }

        return (
            f"data: {json.dumps(start_payload)}\n\n"
            f"data: {json.dumps(output_payload)}\n\n"
        )
    except Exception as exc:
        payload = {
            "type": "tool_output",
            "id": "runtime-decision-unavailable",
            "call_id": "runtime-decision-unavailable",
            "tool": "runtime_decision",
            "command": "Runtime decision unavailable",
            "output": str(exc),
            "exit_code": 1,
        }
        return f"data: {json.dumps(payload)}\n\n"



def _runtime_trace_tool_event(message: str, runtime_decision):
    try:
        trace_text = explain_runtime_route(message)
        status = runtime_decision.to_status()
        return {
            "round": 1,
            "tool": "runtime_decision",
            "command": "Runtime decision",
            "output": trace_text,
            "exit_code": 0,
            "runtime_trace": trace_text,
            "runtime_decision": status,
        }
    except Exception as exc:
        return {
            "round": 1,
            "tool": "runtime_decision",
            "command": "Runtime decision unavailable",
            "output": str(exc),
            "exit_code": 1,
        }





def _agent_langgraph_architecture_enabled_for_request(chat_mode: str | None) -> bool:
    """Return whether Agent mode may enter the LangGraph architecture.

    This is the request-level kill switch. When the Agent Tools setting is OFF,
    Agent mode must bypass knowledge-runtime, contract runtime, general runtime,
    and runtime-router probes, then fall through to the legacy agent loop.
    """
    try:
        from src.runtime_availability import runtime_is_available
        if not runtime_is_available("langgraph"):
            return False
        from src.settings import get_setting
        raw = get_setting("agent_langgraph_enabled", True)
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}
    except Exception:
        return False


def _langgraph_multi_graph_allowed_by_settings() -> bool:
    try:
        from src.agent_runtime.runtime_settings import (
            get_langgraph_runtime_settings,
            runtime_policy,
        )
        return bool(
            runtime_policy(get_langgraph_runtime_settings())["allow_multi_graph"]
        )
    except Exception:
        return False


def _route_agent_runtime_for_request(message: str, enabled: bool):
    """Return no LangGraph decision when the request-level kill switch is off.

    This narrow seam makes Legacy isolation executable and testable: the
    router itself is never invoked for a disabled request.
    """
    if not enabled:
        return None
    return route_agent_runtime(message)


async def _stream_langgraph_first_with_legacy_fallback(*args, **kwargs):
    """Run native LangGraph without widening a failed request into Legacy.

    The historical name remains as a compatibility seam for callers and tests.
    Legacy fallback is permitted only inside an eligible failed PlanStep, never
    for the whole user query.
    """
    import inspect

    try:
        allowed = set(
            inspect.signature(
                stream_general_supervisor_sse
            ).parameters
        )

        langgraph_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in allowed
        }

        async for chunk in stream_general_supervisor_sse(
            *args,
            **langgraph_kwargs,
        ):
            yield chunk

        return

    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as exc:
        logger.exception(
            "[langgraph-first] native LangGraph failed; "
            "failing closed without whole-query Legacy fallback"
        )
        failure = (
            "The native LangGraph runtime could not complete this request "
            "safely. No Legacy fallback was started. "
            f"{type(exc).__name__}: {exc}"
        )
        yield f"data: {json.dumps({'delta': failure})}\n\n"
        yield "data: " + json.dumps({
            "type": "metrics",
            "data": {
                "runtime": "general-supervisor-v1",
                "selected_model": (
                    kwargs.get("model")
                    or (args[1] if len(args) > 1 else None)
                ),
                "presenting_model": None,
                "model": None,
                "worker": None,
                "fallback_used": False,
                "fallback_scope": "none",
                "response_origin": "controlled_runtime_failure",
                "langgraph_runtime": True,
                "whole_query_legacy_fallback": False,
            },
        }) + "\n\n"
        yield "data: [DONE]\n\n"


async def _stream_agent_admission_failure(*args, **kwargs):
    """Return a typed, fail-closed control-plane failure over the SSE seam."""

    detail = str(
        kwargs.get("admission_failure")
        or "Structured Agent admission could not be validated."
    )
    yield "data: " + json.dumps({
        "delta": (
            "I could not validate the Agent execution plan safely after one "
            "bounded structured retry. No workspace, web, or Legacy work was "
            "started."
        ),
    }) + "\n\n"
    yield "data: " + json.dumps({
        "type": "metrics",
        "data": {
            "runtime": "agent-admission-v1",
            "response_origin": "controlled_runtime_failure",
            "failure_type": "structured_output_failed",
            "failure_detail": detail[:800],
            "fallback_used": False,
            "fallback_scope": "none",
            "langgraph_runtime": True,
        },
    }) + "\n\n"
    yield "data: [DONE]\n\n"


async def _stream_multi_graph_fail_closed(*args, **kwargs):
    """Run the admitted coordinator without widening into whole-query Legacy."""
    import inspect

    coordinator_kwargs = dict(kwargs)
    lifecycle = LangGraphRunLifecycle.create(
        session_id=coordinator_kwargs.get("session_id"),
        run_id=coordinator_kwargs.get("run_id"),
        thread_id=coordinator_kwargs.get("thread_id"),
        checkpoint_ns=coordinator_kwargs.get("checkpoint_ns"),
        owner=coordinator_kwargs.get("owner"),
    )
    coordinator_kwargs.setdefault(
        "run_id",
        lifecycle.identity.run_id,
    )
    coordinator_kwargs.setdefault(
        "thread_id",
        lifecycle.identity.thread_id,
    )
    coordinator_kwargs.setdefault(
        "checkpoint_ns",
        lifecycle.identity.checkpoint_ns,
    )
    coordinator_kwargs.setdefault(
        "cancellation_scope",
        lifecycle.cancellation,
    )

    try:
        coordinator_kwargs.setdefault(
            "checkpoint_store",
            get_default_checkpoint_store(),
        )
        allowed = set(
            inspect.signature(
                stream_multi_graph_coordinator_sse
            ).parameters
        )
        coordinator_kwargs = {
            key: value
            for key, value in coordinator_kwargs.items()
            if key in allowed
        }

        async for chunk in stream_multi_graph_coordinator_sse(
            *args,
            **coordinator_kwargs,
        ):
            yield chunk

        return

    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as exc:
        logger.exception(
            "[multi-graph] coordinator failed before output; "
            "failing closed without whole-query Legacy fallback"
        )

        error_text = (
            "The native multi-step runtime could not start safely. "
            "No Legacy fallback was started. "
            f"{type(exc).__name__}: {exc}"
        )
        yield f"data: {json.dumps({'delta': error_text})}\n\n"
        terminal = lifecycle.finish(
            "FAILED",
            stage="coordinator_start",
            reason=f"{type(exc).__name__}: {exc}",
            retryable=True,
            completed_steps=[],
            whole_query_legacy_fallback=False,
        )
        if terminal is not None:
            yield "data: " + json.dumps(terminal) + "\n\n"
        yield (
            "data: "
            + json.dumps({
                "type": "metrics",
                "data": {
                    "runtime": "p3c5b2-multi-graph-v1",
                    "multi_graph": True,
                    "plan_status": "coordinator_start_failed",
                    "terminal_status": "FAILED",
                    "whole_query_legacy_fallback": False,
                    "langgraph_runtime": True,
                },
            })
            + "\n\n"
        )
        yield "data: [DONE]\n\n"


def _append_runtime_tool_event(events, data):
    if not isinstance(events, list) or not isinstance(data, dict):
        return
    if data.get("type") != "tool_output":
        return
    tool = data.get("tool")
    if not tool:
        return

    ev = {
        "round": data.get("round") or 1,
        "tool": tool,
        "command": data.get("command") or "",
        "output": data.get("output") or "",
        "exit_code": data.get("exit_code", 0),
    }

    for key in (
        "diff",
        "screenshot",
        "image_url",
        "image_id",
        "image_prompt",
        "image_model",
        "image_size",
        "image_quality",
        "runtime_trace",
        "runtime_decision",
    ):
        if key in data and data.get(key) is not None:
            ev[key] = data.get(key)

    events.append(ev)


async def _replay_durable_langgraph_run(
    state: dict[str, Any],
    *,
    after_cursor: int = 0,
    events: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[str, None]:
    for event in events if events is not None else state.get("event_log") or []:
        if not isinstance(event, dict):
            continue
        cursor = int((event.get("data") or {}).get("event_cursor") or 0)
        if cursor <= after_cursor:
            continue
        yield f"data: {json.dumps(event, default=str)}\n\n"
    replay_complete = {
        "type": "runtime_replay_complete",
        "data": {
            "run_id": state.get("run_id"),
            "status": state.get("status"),
            "event_cursor": state.get("event_cursor", 0),
            "run_completed": str(
                state.get("status") or ""
            ).upper() in {"COMPLETED", "FAILED", "CANCELLED"},
            "transport_complete": True,
        },
    }
    yield f"data: {json.dumps(replay_complete, default=str)}\n\n"
    yield "data: [DONE]\n\n"


def _durable_checkpoint_store_or_503():
    try:
        return get_default_checkpoint_store()
    except CheckpointUnavailableError as exc:
        raise HTTPException(
            503,
            f"Durable LangGraph recovery is unavailable: {exc}",
        ) from exc


from src.agent_runtime.contract_bridge import (
    contract_runtime_enabled,
    looks_like_readonly_contract_task,
    stream_contract_runtime_legacy_sse,
)
from src import agent_runs
from src.model_context import estimate_tokens
from src.chat_helpers import coerce_message_and_session
from src.endpoint_resolver import normalize_base as _normalize_base, build_chat_url
from src.session_search import search_session_messages
from src.prompt_security import untrusted_context_message
from core.exceptions import SessionNotFoundError
from src.auth_helpers import effective_user, get_current_user
from routes.session_routes import _verify_session_owner
from routes.document_helpers import _owner_session_filter
from core.database import SessionLocal, get_session_mode, set_session_mode
from core.database import Session as DBSession, ChatMessage as DBChatMessage
from core.database import Document as DBDocument, ModelEndpoint
from core.log_safety import redact_url
from routes.research_routes import _resolve_research_endpoint
from routes.model_routes import _visible_models
from routes.chat_helpers import (
    resolve_session_auth,
    build_chat_context,
    save_assistant_response,
    run_post_response_tasks,
    clean_thinking_for_save,
    _enforce_chat_privileges,
)
from src.action_intents import ToolIntent, classify_tool_intent as _classify_tool_intent
from src.image_model_ids import looks_like_image_generation_model
from src.tool_policy import (
    WEB_TOOL_NAMES,
    build_effective_tool_policy,
    is_web_search_explicitly_denied,
    web_search_enabled_for_turn,
)

logger = logging.getLogger(__name__)

# Track active streams for partial-save safety net
_active_streams: Dict[str, dict] = {}
_IMAGE_MODEL_PREFIXES = ("gpt-image", "dall-e", "chatgpt-image")


def _stream_set(session_id: str, **fields) -> None:
    """Update fields on the active-stream entry for `session_id`, or
    no-op if the entry has already been popped. Using .get() avoids a
    KeyError race between `if x in d` and `d[x]["k"] = v` if a sibling
    finally pops the key in between (which becomes possible the moment
    a coroutine cancellation reaches an inner cleanup before the
    outermost cleanup runs)."""
    rec = _active_streams.get(session_id)
    if rec is None:
        return
    rec.update(fields)


def _message_plain_text(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content or "")


def _last_user_plain_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user":
            return _message_plain_text(msg.get("content"))
    return ""


def _ensure_current_request_is_latest_user(messages: List[Dict[str, Any]], current_message: str) -> List[Dict[str, Any]]:
    """Defensively keep detached streams grounded on the request that created them."""
    current = str(current_message or "").strip()
    if not current:
        return messages
    latest = _last_user_plain_text(messages).strip()
    if latest == current or current in latest or latest in current:
        return messages
    logger.warning(
        "[chat_stream] latest user context mismatch; appending current request for model call. latest=%r current=%r",
        latest[:120],
        current[:120],
    )
    repaired = list(messages or [])
    repaired.append({"role": "user", "content": current})
    return repaired


_WEB_FOLLOWUP_RE = re.compile(
    r"^\s*(?:(?:can|could|would|will)\s+you\s+)?"
    r"(?:check|try\s+again|look(?:\s+now|\s+it\s+up)?|search(?:\s+now|\s+online|\s+it)?|"
    r"do\s+it|again|approved|approve(?:d)?|yes|ok(?:ay)?|proceed|go\s+ahead|"
    r"send(?:\s+it)?|submit(?:\s+it)?|email(?:\s+them|\s+it)?)\??\s*$",
    re.I,
)
_RECENT_WEB_CONTEXT_RE = re.compile(
    r"\b(?:weather|forecast|rain|raining|hourly|news|headlines|rate|exchange|currency|"
    r"price|current|latest|search|look\s+up|online)\b",
    re.I,
)
_RECENT_BROWSER_CONTEXT_RE = re.compile(
    r"\b(?:browser|browse|open\s+(?:the\s+)?(?:site|page|url|link)|click|"
    r"fill(?:\s+out)?|submit|send\s+(?:the\s+)?form|contact\s+form|web\s*form|"
    r"form\s+submission|playwright|automation)\b",
    re.I,
)
_BROWSER_MCP_TOOLS = {
    "mcp__builtin_browser__browser_navigate",
    "mcp__builtin_browser__browser_snapshot",
    "mcp__builtin_browser__browser_click",
    "mcp__builtin_browser__browser_type",
    "mcp__builtin_browser__browser_fill_form",
    "mcp__builtin_browser__browser_select_option",
    "mcp__builtin_browser__browser_press_key",
    "mcp__builtin_browser__browser_wait_for",
    "mcp__builtin_browser__browser_take_screenshot",
    "mcp__builtin_browser__browser_drag",
    "mcp__builtin_browser__browser_navigate_back",
    "mcp__builtin_browser__browser_close",
}


def _recent_session_text(sess, limit: int = 8, max_chars: int = 2000) -> str:
    history = getattr(sess, "history", None) or getattr(sess, "_history", None) or []
    chunks: List[str] = []
    for msg in history[-limit:]:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        text = _message_plain_text(content).strip()
        if text:
            chunks.append(text)
    return " ".join(chunks)[-max_chars:]


def _is_contextual_web_followup(message: str, sess) -> bool:
    """Treat short retry/check replies as web lookups when recent context was web."""
    if not message or not _WEB_FOLLOWUP_RE.search(message):
        return False
    return bool(_RECENT_WEB_CONTEXT_RE.search(_recent_session_text(sess)))


def _is_contextual_browser_followup(message: str, sess) -> bool:
    """Treat short retry replies as browser tasks when recent context was forms/browser automation."""
    if not message or not _WEB_FOLLOWUP_RE.search(message):
        return False
    return bool(_RECENT_BROWSER_CONTEXT_RE.search(_recent_session_text(sess, limit=12, max_chars=4000)))


def _resolve_request_workspace(request, raw_value) -> tuple:
    """Resolve the posted workspace for this request: (workspace, rejected).

    Privilege is checked BEFORE the path ever touches the filesystem. Only
    admin/single-user callers can use the workspace-backed file/shell tools,
    so only they get vet_workspace() and the workspace_rejected signal. For
    any other caller the submitted value is dropped uniformly, with no vetting
    and no event: otherwise the presence/absence of workspace_rejected would
    let a non-admin chat caller probe which host paths exist.

    vet_workspace rejects non-directories, sensitive roots (.ssh, .gnupg,
    ...), and filesystem roots; on rejection there is no confinement and the
    default tool-path allowlist applies. The rejected value is surfaced so the
    stream can tell an admin client (which believes a workspace is active)
    that it was dropped.
    """
    requested = (raw_value or "").strip()
    if not requested:
        return "", ""
    from src.tool_security import owner_is_admin_or_single_user
    if not owner_is_admin_or_single_user(get_current_user(request)):
        return "", ""
    from src.tool_execution import vet_workspace
    workspace = vet_workspace(requested) or ""
    return workspace, (requested if not workspace else "")


_ABS_PATH_RE = re.compile(r"(?<!\S)(~?/[^\"'\s`<>]+)")
_LOCAL_FILE_TASK_RE = re.compile(
    r"\b(?:file|folder|directory|path|workspace|repo|project|movie|video|"
    r"subtitle|subtitles|srt|vtt|ass|download|save|rename|move|copy|extract|"
    r"convert|ffmpeg|run|execute|open|read|inspect|fix|debug|test|build)\b",
    re.IGNORECASE,
)


def _resolve_workspace_from_message_path(request, message: str) -> tuple[str, str]:
    """Auto-bind a workspace only when the user names an explicit safe path.

    This is intentionally deterministic rather than LLM/RAG-driven: RAG can
    choose the tool family, but filesystem binding must not let a prompt infer
    or probe arbitrary host paths. For a file path, bind its parent directory.
    For a directory path, bind that directory.
    """
    text = str(message or "")
    if not text or not _LOCAL_FILE_TASK_RE.search(text):
        return "", ""

    from src.tool_security import owner_is_admin_or_single_user
    if not owner_is_admin_or_single_user(get_current_user(request)):
        return "", ""

    from src.tool_execution import vet_workspace

    for match in _ABS_PATH_RE.finditer(text):
        raw = match.group(1).rstrip(".,;:)]}")
        expanded = os.path.realpath(os.path.expanduser(raw))
        candidates = [expanded]
        if os.path.isfile(expanded):
            candidates.insert(0, os.path.dirname(expanded))
        for candidate in candidates:
            workspace = vet_workspace(candidate) or ""
            if workspace:
                return workspace, ""
    return "", ""


def _session_url_matches_endpoint(session_url: str, endpoint_base: str) -> bool:
    if not session_url or not endpoint_base:
        return False
    sess = session_url.rstrip("/")
    base = _normalize_base(endpoint_base).rstrip("/")
    variants = {
        base,
        base + "/chat/completions",
        build_chat_url(base).rstrip("/"),
    }
    return sess in variants or sess.startswith(base + "/")


def _clear_orphaned_session_endpoint(sess, owner: str | None = None) -> bool:
    """Clear a session model if its endpoint was deleted from ModelEndpoint."""
    if not getattr(sess, "endpoint_url", ""):
        return False
    db = SessionLocal()
    try:
        q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
        if owner:
            from src.auth_helpers import owner_filter
            q = owner_filter(q, ModelEndpoint, owner)
        endpoints = q.all()
        for ep in endpoints:
            if _session_url_matches_endpoint(sess.endpoint_url or "", ep.base_url or ""):
                return False
        db_session = db.query(DBSession).filter(DBSession.id == sess.id).first()
        if db_session:
            db_session.endpoint_url = ""
            db_session.model = ""
            db_session.updated_at = datetime.utcnow()
            db.commit()
        sess.endpoint_url = ""
        sess.model = ""
        sess.headers = {}
        return True
    except Exception as e:
        logger.warning("Failed to clear orphaned session endpoint", exc_info=e)
        db.rollback()
        return False
    finally:
        db.close()


def _endpoint_cache_contains_model(endpoint, model: str) -> bool:
    """Return True when a populated endpoint model cache includes ``model``.

    Empty/malformed caches are treated as unknown rather than a negative match
    so older image endpoints without cached models still work.
    """
    raw = getattr(endpoint, "cached_models", None)
    if not raw:
        return True
    try:
        models = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.warning("Failed to parse cached models list, treating as containing model", exc_info=e)
        return True
    if not isinstance(models, list) or not models:
        return True
    wanted = (model or "").strip()
    return wanted in {str(item).strip() for item in models}


def _is_image_generation_session(sess, owner: str | None = None) -> bool:
    """Whether this chat session should bypass text chat and generate images.

    Model-name prefixes are explicit image models. Endpoint type is only used
    when the current session endpoint actually matches that image endpoint, and
    when a populated endpoint model cache includes the selected model. This
    prevents an image endpoint on the same host from misrouting ordinary text
    models into the image-generation path.
    """
    model = (getattr(sess, "model", "") or "").strip()
    if looks_like_image_generation_model(model):
        return True

    endpoint_url = (getattr(sess, "endpoint_url", "") or "").strip()
    if not endpoint_url:
        return False

    db = SessionLocal()
    try:
        q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
        if owner:
            from src.auth_helpers import owner_filter
            q = owner_filter(q, ModelEndpoint, owner)
        endpoints = q.all()
        for endpoint in endpoints:
            if (getattr(endpoint, "model_type", None) or "llm") != "image":
                continue
            if not _session_url_matches_endpoint(endpoint_url, getattr(endpoint, "base_url", "") or ""):
                continue
            if _endpoint_cache_contains_model(endpoint, model):
                return True
    except Exception:
        return False
    finally:
        db.close()
    return False


def _first_image_attachment(chat_handler, att_ids: List[str], owner: str | None = None) -> Optional[Dict[str, Any]]:
    """Return the first attached image file that this owner can read."""
    upload_handler = getattr(chat_handler, "upload_handler", None)
    if not upload_handler:
        return None
    for att_id in att_ids or []:
        try:
            info = upload_handler.resolve_upload(att_id, owner=owner)
        except Exception as e:
            logger.warning("Failed to resolve image edit upload %s", att_id, exc_info=e)
            continue
        if not info:
            continue
        name = info.get("name") or info.get("original_name") or info.get("id") or ""
        mime = info.get("mime", "")
        try:
            if upload_handler.is_image_file(name, mime):
                return info
        except Exception:
            continue
    return None


def _recover_empty_session_model(sess, session_id: str, owner: str | None = None) -> bool:
    """Re-populate sess.model from the matching endpoint's cached models.

    Covers the window between endpoint setup and the first chat send: the
    picker showed a model in the dropdown but the session record never got
    written (Issue #587 — UI uses the cached endpoint list, not s.model).
    For ChatGPT Subscription, also repairs stale OpenAI API model names such as
    ``gpt-5`` that are not accepted by the Codex-backed ChatGPT account route.
    """
    current_model = (getattr(sess, "model", "") or "").strip()
    endpoint_url = (getattr(sess, "endpoint_url", "") or "").strip()
    is_chatgpt_subscription = False
    if current_model:
        try:
            from src.chatgpt_subscription import is_chatgpt_subscription_base
            is_chatgpt_subscription = is_chatgpt_subscription_base(endpoint_url)
            if not is_chatgpt_subscription:
                return False
        except Exception:
            return False
    db = SessionLocal()
    try:
        # Prefer the endpoint whose base URL matches the session — we know the
        # user already pointed this session at that endpoint, so its first
        # cached model is the most defensible default.
        ep = None
        if getattr(sess, "endpoint_url", ""):
            q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
            if owner:
                from src.auth_helpers import owner_filter
                q = owner_filter(q, ModelEndpoint, owner)
            endpoints = q.all()
            for cand in endpoints:
                if _session_url_matches_endpoint(sess.endpoint_url or "", cand.base_url or ""):
                    ep = cand
                    break
        if not ep:
            return False
        if not is_chatgpt_subscription:
            try:
                from src.chatgpt_subscription import is_chatgpt_subscription_base
                is_chatgpt_subscription = is_chatgpt_subscription_base(getattr(ep, "base_url", "") or endpoint_url)
            except Exception:
                is_chatgpt_subscription = False
        try:
            cached = json.loads(ep.cached_models) if isinstance(ep.cached_models, str) else (ep.cached_models or [])
        except Exception as e:
            logger.warning("Failed to parse cached_models for endpoint %r", getattr(ep, "id", "?"), exc_info=e)
            cached = []
        if not cached:
            visible = []
        else:
            try:
                visible = _visible_models(cached, getattr(ep, "hidden_models", None))
            except Exception:
                visible = cached
        if current_model and current_model in {str(item).strip() for item in visible}:
            return False
        if is_chatgpt_subscription:
            live_models = []
            if getattr(ep, "provider_auth_id", None):
                try:
                    from src.chatgpt_subscription import fetch_available_models
                    from src.endpoint_resolver import resolve_endpoint_runtime
                    _base, api_key = resolve_endpoint_runtime(ep, owner=owner)
                    if api_key:
                        live_models = fetch_available_models(api_key)
                        if live_models:
                            ep.cached_models = json.dumps(live_models)
                            db.commit()
                except Exception:
                    live_models = []
            # ChatGPT Subscription recovery must use the live Codex catalog.
            # Cached rows are only trusted above to avoid revalidating a model
            # that is already present in the visible picker list.
            cached = live_models
            if not cached:
                return False
            try:
                visible = _visible_models(cached, getattr(ep, "hidden_models", None))
            except Exception:
                visible = cached
            if current_model and current_model in {str(item).strip() for item in visible}:
                return False
        if not visible:
            return False
        model = visible[0]
        if not isinstance(model, str) or not model.strip():
            return False
        model = model.strip()
        # Persist so the next request, websocket reconnect, or page reload
        # picks up the same model (we'd otherwise re-pick on every send
        # and silently switch on the user if the cached order shifts).
        db_session_q = db.query(DBSession).filter(DBSession.id == session_id)
        if owner:
            db_session_q = db_session_q.filter(DBSession.owner == owner)
        db_session = db_session_q.first()
        if db_session:
            db_session.model = model
            db_session.updated_at = datetime.utcnow()
            db.commit()
        sess.model = model
        logger.info(
            "Recovered session model for %s — picked %r from endpoint %s",
            session_id, model, ep.id,
        )
        return True
    except Exception as e:
        db.rollback()
        logger.warning("Failed to recover empty session model for %s: %s", session_id, e)
    return False


def _reconcile_selected_route_from_request(
    request: Request,
    sess,
    session_id: str,
    form_data,
    owner: str | None = None,
) -> bool:
    """Apply the model route the browser selected before streaming.

    The frontend creates a pending chat first and only materializes it on first
    send. Startup/default-model refreshes can race with that UI state, so the
    stream request includes the route that was selected at click/send time.
    Trust only registered endpoint ids, or the session's existing endpoint URL.
    """
    selected_model = str(form_data.get("selected_model") or "").strip()
    selected_endpoint_id = str(form_data.get("selected_endpoint_id") or "").strip()
    selected_endpoint_url = str(form_data.get("selected_endpoint_url") or "").strip()
    if not selected_model:
        return False

    endpoint_url = ""
    headers = None
    if selected_endpoint_id or selected_endpoint_url:
        try:
            from src.auth_helpers import owner_filter
            from src.endpoint_resolver import build_headers, normalize_base
            db = SessionLocal()
            try:
                q = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True)
                if selected_endpoint_id:
                    q = q.filter(ModelEndpoint.id == selected_endpoint_id)
                if owner:
                    q = owner_filter(q, ModelEndpoint, owner)
                candidates = q.all() if selected_endpoint_url and not selected_endpoint_id else [q.first()]
                ep = None
                for cand in candidates:
                    if not cand:
                        continue
                    if selected_endpoint_id or _session_url_matches_endpoint(selected_endpoint_url, cand.base_url or ""):
                        ep = cand
                        break
                if not ep:
                    return False
                endpoint_url = build_chat_url(normalize_base(ep.base_url or ""))
                headers = build_headers(ep.api_key or "", ep.base_url or "") if ep.api_key else {}
            finally:
                db.close()
        except Exception as e:
            logger.warning("Failed to resolve selected endpoint %s/%s for %s: %s", selected_endpoint_id, selected_endpoint_url, session_id, e)
            return False

    if not endpoint_url:
        return False

    if (
        selected_model == (getattr(sess, "model", "") or "")
        and endpoint_url == (getattr(sess, "endpoint_url", "") or "")
    ):
        return False

    sess.model = selected_model
    sess.endpoint_url = endpoint_url
    sess.headers = headers or {}
    db = SessionLocal()
    try:
        db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
        if db_session:
            db_session.model = selected_model
            db_session.endpoint_url = endpoint_url
            db_session.headers = sess.headers or {}
            db_session.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()
    logger.info("Reconciled selected route for %s: model=%r endpoint=%s", session_id, selected_model, redact_url(endpoint_url))
    return True


def _set_user_time_from_request(request: Request) -> None:
    """Copy browser timezone headers into the per-request context.

    This is intentionally ephemeral: it is used only while building prompts
    and running tools for this request. It is not persisted or logged.
    """
    try:
        tz_offset = request.headers.get("x-tz-offset")
        tz_name = request.headers.get("x-tz-name")
        from src.user_time import clear_user_time_context, set_user_tz_name, set_user_tz_offset

        clear_user_time_context()
        if tz_offset is not None:
            set_user_tz_offset(tz_offset)
        if tz_name:
            set_user_tz_name(tz_name)
    except Exception:
        pass


def setup_chat_routes(
    session_manager,
    chat_handler,
    chat_processor,
    memory_manager,
    research_handler,
    upload_handler,
    memory_vector=None,
    webhook_manager=None,
    skills_manager=None,
) -> APIRouter:
    router = APIRouter(tags=["chat"])

    # ------------------------------------------------------------------ #
    # POST /api/chat (non-streaming)
    # ------------------------------------------------------------------ #
    @router.post("/api/chat", response_model=Dict[str, str])
    async def chat_endpoint(request: Request, chat_request: ChatRequest) -> Dict[str, str]:
        _set_user_time_from_request(request)

        message = chat_request.message
        session = chat_request.session
        att_ids = chat_request.attachments or []
        use_web = chat_request.use_web
        use_research = chat_request.use_research
        time_filter = chat_request.time_filter
        preset_id = chat_request.preset_id

        # Verify the caller owns this session before loading it.
        # Without this, any authenticated user can post into another user's chat.
        _verify_session_owner(request, session)

        try:
            sess = session_manager.get_session(session)
        except KeyError:
            raise HTTPException(404, f"Session '{session}' not found")
        owner = effective_user(request)
        if _clear_orphaned_session_endpoint(sess, owner=owner):
            raise HTTPException(400, "Selected model endpoint was removed. Pick another model in Settings.")

        # Empty model + live endpoint = setup race (Issue #587). Repair from
        # the endpoint's cached model list before privilege checks, which
        # otherwise see "" and behave inconsistently with the allowlist.
        _recover_empty_session_model(sess, session, owner=owner)
        if not getattr(sess, "model", "").strip():
            raise HTTPException(
                400,
                "No model selected for this chat. Open the model picker and choose one before sending.",
            )

        # Same allowed_models + daily-cap gate as chat_stream (mirror so the
        # non-streaming path can't be used to bypass).
        _enforce_chat_privileges(request, sess)

        tool_policy = build_effective_tool_policy(last_user_message=message)
        allow_tool_preprocessing = not tool_policy.block_all_tool_calls

        # Inline memory command
        memory_response = None
        if not tool_policy.blocks("manage_memory"):
            memory_response = await chat_handler.handle_memory_command(sess, message)
        if memory_response:
            return {"response": memory_response}

        # Build shared context (preset, preprocess, preface, compact)
        ctx = await build_chat_context(
            sess, request, chat_handler, chat_processor,
            message=message,
            session_id=session,
            preset_id=preset_id,
            att_ids=att_ids,
            use_web=use_web,
            time_filter=time_filter,
            webhook_manager=webhook_manager,
            allow_tool_preprocessing=allow_tool_preprocessing,
        )

        # Research injection
        research_blocked_by_policy = (
            tool_policy.blocks("trigger_research")
            or tool_policy.blocks("manage_research")
        )
        if use_research and not research_blocked_by_policy:
            try:
                _r_ep, _r_model, _r_headers = _resolve_research_endpoint(sess)
                research_ctx = await research_handler.call_research_service(
                    message, _r_ep, _r_model, llm_headers=_r_headers
                )
                ctx.messages.insert(
                    len(ctx.preface),
                    untrusted_context_message("research context", research_ctx),
                )
            except Exception as e:
                logger.error(f"Research failed: {e}")

        reply = await llm_call_async(
            sess.endpoint_url,
            sess.model,
            ctx.messages,
            headers=sess.headers,
            temperature=ctx.preset.temperature,
            max_tokens=ctx.preset.max_tokens,
            prompt_type=preset_id,
            session_id=session,
        )
        _clean_reply, _clean_md = clean_thinking_for_save(reply, {"model": sess.model})
        sess.add_message(ChatMessage("assistant", _clean_reply, metadata=_clean_md))

        from core.database import update_session_last_accessed
        update_session_last_accessed(session)
        session_manager.save_sessions()

        # Background tasks (memory, webhook, auto-name)
        run_post_response_tasks(
            sess, session_manager, session, message, reply, None,
            ctx.uprefs, memory_manager, memory_vector, webhook_manager,
            character_name=ctx.preset.character_name,
            owner=ctx.user,
            allow_background_extraction=not tool_policy.block_all_tool_calls,
        )

        return {"response": reply}

    # ------------------------------------------------------------------ #
    # POST /api/chat_stream
    # ------------------------------------------------------------------ #
    @router.post("/api/chat_stream")
    async def chat_stream(request: Request) -> StreamingResponse:
        body = None
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                try:
                    body = await request.json()
                except json.JSONDecodeError as e:
                    raise HTTPException(400, f"Invalid JSON: {e}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Request parsing error: {e}")

        _set_user_time_from_request(request)

        form_data = await request.form()
        message = form_data.get("message")
        session = form_data.get("session")
        attachments = form_data.get("attachments")
        use_web = form_data.get("use_web")
        use_research = form_data.get("use_research")
        time_filter = form_data.get("time_filter")
        preset_id = form_data.get("preset_id")
        # Issue #3229: API callers send JSON, not FormData.  Read from the
        # JSON body as fallback so callers who send {"allow_bash": true}
        # actually get bash enabled.
        allow_bash = form_data.get("allow_bash") or (body or {}).get("allow_bash")
        allow_web_search = form_data.get("allow_web_search") or (body or {}).get("allow_web_search")
        use_rag = form_data.get("use_rag")
        search_context = form_data.get("search_context")  # pre-fetched web search results (compare mode)
        compare_mode = str(form_data.get("compare_mode", "")).lower() == "true"
        incognito = str(form_data.get("incognito", "")).lower() == "true"
        # Plan mode is not part of the merge-ready UI. Ignore stale clients or
        # manual form posts that still send plan_mode=true.
        plan_mode = False
        chat_mode = str(form_data.get("mode", "")).lower()  # 'chat' or 'agent'
        # Workspace: confine the agent's file/shell tools to this folder.
        workspace, workspace_rejected = _resolve_request_workspace(
            request, form_data.get("workspace")
        )
        # Plan mode is a modifier on agent mode — it only makes sense with tools.
        if plan_mode:
            chat_mode = "agent"
        # An approved plan being EXECUTED: the frontend sends the checklist back
        # on each turn so we can pin it in context. This way a long plan on a
        # weak model survives history truncation — the agent can always re-read
        # the plan. Ignored while still proposing (plan_mode on). Capped so a
        # huge plan can't blow the prompt.
        approved_plan = ""
        if not plan_mode:
            approved_plan = (form_data.get("approved_plan") or "").strip()[:8192]
        # Did the USER explicitly pick agent mode? (vs. us auto-escalating
        # below). Skill extraction should only learn from real agent sessions,
        # not chats we quietly promoted for a notes/calendar intent.
        user_requested_agent = (chat_mode == "agent")
        _search_enabled = web_search_enabled_for_turn(allow_web_search, use_web)


        # P3_C5_EXPLICIT_AGENT_ADMISSION_PREP_START
        # Preserve upstream Chat-to-Agent auto-escalation below. This seam only
        # replaces deterministic demotion of explicitly selected Agent turns.
        # AGENT_LANGGRAPH_REQUEST_BYPASS_V1_START
        # When LangGraph is disabled, explicit Agent remains on Legacy Agent.
        _agent_langgraph_architecture_enabled = _agent_langgraph_architecture_enabled_for_request(chat_mode)
        # AGENT_LANGGRAPH_REQUEST_BYPASS_V1_END
        _agent_admission = None
        _agent_admission_failure = None
        # P3_C5_EXPLICIT_AGENT_ADMISSION_PREP_END


        _explicit_web_intent = False
        _explicit_browser_intent = False
        if isinstance(message, str):
            _msg_l = message.lower()
            _explicit_web_intent = bool(re.search(
                r"\b(search|look\s*up|lookup|google|browse|web|online|latest|current|today|news|weather|forecast|rate|exchange\s+rate)\b",
                _msg_l,
            ))
            _explicit_browser_intent = bool(re.search(
                r"\b(browser|browse|open\s+(?:the\s+)?(?:site|page|url|link)|"
                r"click|fill(?:\s+out)?|submit|send\s+(?:the\s+)?form|"
                r"contact\s+form|web\s*form|form\s+submission)\b",
                _msg_l,
            ))
        _allow_browser_for_web_turn = bool(
            _explicit_browser_intent
            or _explicit_web_intent
            or _search_enabled
        )
        # Intent auto-escalation: if the user is clearly asking the assistant
        # to create a todo, reminder, or calendar event, promote chat → agent
        # for this turn so the LLM has access to manage_notes / manage_calendar.
        # This is a LIGHT promotion — see the disabled_tools block below, which
        # withholds shell/code/file tools so the model doesn't try to `bash`
        # its way through a plain chat request (and fail, especially with the
        # shell disabled).
        auto_escalated = False
        _tool_intent = _classify_tool_intent(message) if isinstance(message, str) else None
        _workspace_agent_intent = False
        if _agent_langgraph_architecture_enabled and chat_mode == "chat" and _tool_intent and _tool_intent.needs_tools:
            chat_mode = "agent"
            auto_escalated = True
            _workspace_agent_intent = _tool_intent.category in {"shell", "workspace"}
            if _workspace_agent_intent:
                allow_bash = "true"
            logger.info(
                "chat→agent auto-escalation: category=%s reason=%s",
                _tool_intent.category,
                _tool_intent.reason,
            )
        elif chat_mode == "chat" and _search_enabled:
            chat_mode = "agent"
            auto_escalated = True
            logger.info("chat→agent auto-escalation: search enabled")
        elif chat_mode == "chat" and _explicit_web_intent:
            chat_mode = "agent"
            auto_escalated = True
            logger.info("chat→agent auto-escalation: explicit web intent")
        active_doc_id = form_data.get("active_doc_id", "").strip()
        logger.info(f"[doc-inject] chat_mode={chat_mode}, active_doc_id={active_doc_id!r}")

        # Active email reader — when the user has an email open in the UI, the
        # frontend passes its uid/folder/account so "reply", "summarize this",
        # etc. resolve to the real email instead of the agent inventing a
        # fake markdown draft.
        active_email_uid = form_data.get("active_email_uid", "").strip()
        active_email_folder = form_data.get("active_email_folder", "INBOX").strip() or "INBOX"
        active_email_account = form_data.get("active_email_account", "").strip()
        active_email_ctx: Optional[Dict[str, str]] = None
        # Always reset between requests so a stale active-email pointer from
        # a previous turn (different reader closed, different account, etc.)
        # can't leak in when the user has no email open this turn.
        try:
            from src.tool_implementations import clear_active_email
            clear_active_email()
        except Exception:
            pass
        if active_email_uid:
            active_email_ctx = {
                "uid": active_email_uid,
                "folder": active_email_folder,
                "account": active_email_account,
            }
            # Try to enrich with subject + from so the agent's system prompt
            # block can quote them. Best-effort: a stale cache is fine, a
            # missing email just means we pass uid/folder/account only.
            try:
                from routes.email_routes import _read_cache_get, _read_cache_key
                _ck = _read_cache_key(active_email_account or None, active_email_folder, active_email_uid, owner=get_current_user(request))
                _cached_email = _read_cache_get(_ck)
                if _cached_email and isinstance(_cached_email, dict):
                    active_email_ctx["subject"] = str(_cached_email.get("subject") or "")
                    active_email_ctx["from"] = str(
                        _cached_email.get("from_address")
                        or _cached_email.get("from")
                        or _cached_email.get("from_name")
                        or ""
                    )
                    _body_preview = (_cached_email.get("body") or "")[:2000]
                    if _body_preview:
                        active_email_ctx["body_preview"] = _body_preview
            except Exception as _e:
                logger.debug(f"[email-inject] cache enrich skipped: {_e}")
            # Stash so email tools can resolve "this email" without UID guessing.
            try:
                from src.tool_implementations import set_active_email
                set_active_email(
                    uid=active_email_uid,
                    folder=active_email_folder,
                    account=active_email_account or None,
                    subject=active_email_ctx.get("subject"),
                    sender=active_email_ctx.get("from"),
                )
            except Exception as _e:
                logger.debug(f"[email-inject] set_active_email failed: {_e}")
            logger.info(
                "[email-inject] active_email uid=%s folder=%s account=%s subject=%r",
                active_email_uid, active_email_folder, active_email_account or "(default)",
                active_email_ctx.get("subject", ""),
            )

        try:
            # Attachment-only sends: skip the message-required check when the
            # user has attached one or more files (the attachment IS the action).
            _has_atts = (
                bool(body and isinstance(body.get("attachments"), list) and body["attachments"])
                or bool(form_data.get("attachments"))
            )
            message, session = coerce_message_and_session(
                body, message, session, session_manager, allow_empty=_has_atts,
            )
            # Verify ownership AFTER coerce (which may resolve a default session)
            # but BEFORE loading. Prevents cross-user session hijack.
            _verify_session_owner(request, session)
            sess = session_manager.get_session(session)
            owner = effective_user(request)
            _reconcile_selected_route_from_request(request, sess, session, form_data, owner=owner)
            if _clear_orphaned_session_endpoint(sess, owner=owner):
                raise HTTPException(400, "Selected model endpoint was removed. Pick another model in Settings.")
            # Issue #587: picker shows a model from the endpoint cache but
            # s.model never made it onto the DB row (first-send race after
            # endpoint setup, or a previous endpoint delete/recreate). Pull
            # the first cached model off the matching endpoint so the
            # upstream isn't called with model="" (which surfaces as a
            # generic 401/503).
            _recover_empty_session_model(sess, session, owner=owner)
            if not getattr(sess, "model", "").strip():
                raise HTTPException(
                    400,
                    "No model selected for this chat. Open the model picker and choose one before sending.",
                )
            if (
                chat_mode == "chat"
                and isinstance(message, str)
                and (not _tool_intent or not _tool_intent.needs_tools)
                and _is_contextual_web_followup(message, sess)
            ):
                _tool_intent = ToolIntent(True, "web", "contextual web lookup follow-up")
                chat_mode = "agent"
                auto_escalated = True
                _workspace_agent_intent = False
                logger.info(
                    "chat→agent auto-escalation: category=%s reason=%s",
                    _tool_intent.category,
                    _tool_intent.reason,
                )
            if isinstance(message, str) and _is_contextual_browser_followup(message, sess):
                _explicit_browser_intent = True
                if chat_mode == "chat":
                    chat_mode = "agent"
                    auto_escalated = True
                    _workspace_agent_intent = False
                    logger.info("chat→agent auto-escalation: contextual browser/form follow-up")
            if not workspace and isinstance(message, str):
                _auto_workspace, _ = _resolve_workspace_from_message_path(request, message)
                if _auto_workspace:
                    workspace = _auto_workspace
                    chat_mode = "agent"
                    auto_escalated = True
                    _workspace_agent_intent = True
                    allow_bash = "true"
                    logger.info("chat→agent auto-escalation: explicit path workspace=%s", workspace)
        except SessionNotFoundError as e:
            raise HTTPException(404, str(e))
        except (ValueError, ValidationError):
            raise HTTPException(400, "Invalid request parameters")

        # ------------------------------------------------------------------ #
        # Privilege gates that must fire BEFORE any LLM work / token spend.
        #   1. allowed_models — reject if session.model isn't in the user's
        #      configured allowlist (empty list = "no restriction").
        #   2. max_messages_per_day — count user-role ChatMessage rows owned
        #      by this user in the last UTC day; 429 if at/over the cap.
        # Admins always have full privileges via get_privileges (returns
        # ADMIN_PRIVILEGES wholesale) so this is a no-op for them.
        _enforce_chat_privileges(request, sess)

        # Ensure session has auth headers
        resolve_session_auth(sess, session, owner=effective_user(request))

        # P3_C5_EXPLICIT_AGENT_ADMISSION_START
        # Only the user's explicit Agent selection is semantically admitted.
        # Upstream Chat-to-Agent promotion remains owned by action_intents.
        _agent_admission_task = None
        if user_requested_agent:
            try:
                from src.endpoint_resolver import resolve_chat_fallback_candidates

                try:
                    _admission_fallback_candidates = (
                        resolve_chat_fallback_candidates(owner=owner)
                    )
                except Exception as _admission_fallback_exc:
                    logger.warning(
                        "[agent-admission] failed to resolve chat fallbacks: %s",
                        _admission_fallback_exc,
                    )
                    _admission_fallback_candidates = []

                _agent_admission_task = asyncio.create_task(
                    assess_explicit_agent_admission(
                        message,
                        langgraph_enabled=_agent_langgraph_architecture_enabled,
                        llm_candidates=[
                            (sess.endpoint_url, sess.model, sess.headers),
                            *_admission_fallback_candidates,
                        ],
                        prompt_type=preset_id,
                        active_workspace=bool(workspace),
                        workspace_label=(
                            os.path.basename(
                                str(workspace).rstrip("\\/")
                            )
                            if workspace
                            else None
                        ),
                    )
                )
            except Exception as _agent_admission_exc:
                logger.exception(
                    "[agent-admission] could not start semantic admission: %s",
                    _agent_admission_exc,
                )
                _agent_admission_failure = (
                    f"{type(_agent_admission_exc).__name__}: "
                    f"{_agent_admission_exc}"
                )
        # P3_C5_EXPLICIT_AGENT_ADMISSION_END

        # Check for research_pending BEFORE mode persist overwrites it
        do_research = str(use_research).lower() == "true"
        if not do_research:
            if get_session_mode(session) == 'research_pending':
                do_research = True
                logger.info(f"Session {session} in research_pending — auto-triggering research")

        att_ids = []
        if body and isinstance(body.get("attachments"), list):
            att_ids = [str(x) for x in body["attachments"]]
        elif attachments:
            try:
                att_ids = [str(x) for x in json.loads(attachments)]
            except Exception as e:
                logger.warning("Failed to parse attachments JSON, ignoring attachments", exc_info=e)

        image_generation_session = _is_image_generation_session(sess, owner=effective_user(request))
        no_memory = str(form_data.get("no_memory", "")).lower() == "true"
        if image_generation_session:
            no_memory = True
            use_rag = "false"
            search_context = None
        pre_context_tool_policy = build_effective_tool_policy(
            last_user_message=message,
        )
        allow_tool_preprocessing = not pre_context_tool_policy.block_all_tool_calls

        # Build shared context (stream path uses enhanced_message for context preface)
        ctx = await build_chat_context(
            sess, request, chat_handler, chat_processor,
            message=message,
            session_id=session,
            preset_id=preset_id,
            att_ids=att_ids,
            use_web=_search_enabled,
            use_rag=use_rag,
            time_filter=time_filter,
            incognito=incognito,
            no_memory=no_memory,
            search_context=search_context,
            compare_mode=compare_mode,
            webhook_manager=webhook_manager,
            use_enhanced_message=True,
            # Skills index only ships when the model can actually call
            # manage_skills (agent mode). In plain chat or incognito the
            # index would be useless / unwanted noise.
            agent_mode=(chat_mode == "agent"),
            allow_tool_preprocessing=allow_tool_preprocessing,
        )

        _research_flags = {"do": do_research}  # Mutable container for generator scope

        # Query active document — prefer explicit ID from frontend, fall back to session lookup
        active_doc = None
        _doc_db = SessionLocal()
        try:
            if active_doc_id:
                logger.info(f"[doc-inject] active_doc_id from frontend: {active_doc_id}")
                # Scope to the caller's documents. The session and in-memory
                # fallbacks below are already owner/session-bound; this
                # explicit-id path looked up by id alone, so a user could
                # inject another user's document by passing its id.
                _doc_q = _doc_db.query(DBDocument).filter(DBDocument.id == active_doc_id)
                active_doc = _owner_session_filter(_doc_q, ctx.user).first()
                if active_doc:
                    doc_session = active_doc.session_id
                    doc_owner = getattr(active_doc, "owner", None)
                    if doc_owner and ctx.user and doc_owner != ctx.user:
                        logger.warning(
                            "[doc-inject] ignoring active_doc_id %s owned by another user",
                            active_doc_id,
                        )
                        active_doc = None
                    else:
                        # NOTE: previously dropped the doc when doc.session_id
                        # != current chat session — but that broke the common
                        # case of "open an email draft from one chat, ask a
                        # different chat to write into it". The frontend only
                        # sends active_doc_id for docs currently visible in
                        # the UI, and we already owner-checked above, so trust
                        # the explicit signal. We just log the mismatch and
                        # re-bind the doc to the current session so future
                        # turns find it via the session-fallback path too.
                        if doc_session and doc_session != session:
                            logger.info(
                                "[doc-inject] cross-session active_doc_id %s (was session %s, now %s) — accepting and rebinding",
                                active_doc_id, doc_session, session,
                            )
                            try:
                                active_doc.session_id = session
                                _doc_db.commit()
                            except Exception as _e:
                                _doc_db.rollback()
                                logger.warning(f"[doc-inject] session rebind failed: {_e}")
                        logger.info(f"[doc-inject] found by ID: title={active_doc.title!r}, lang={active_doc.language!r}, is_active={active_doc.is_active}, content_len={len(active_doc.current_content or '')}")
                else:
                    logger.warning(f"[doc-inject] NOT FOUND by ID {active_doc_id}")
            if not active_doc:
                _email_doc_q = _doc_db.query(DBDocument).filter(
                    DBDocument.session_id == session,
                    DBDocument.is_active == True,
                    DBDocument.language == "email",
                )
                active_doc = _owner_session_filter(_email_doc_q, ctx.user).order_by(DBDocument.updated_at.desc()).first()
                if active_doc:
                    logger.info(f"[doc-inject] found email draft by session fallback: title={active_doc.title!r}")
            if not active_doc:
                _session_doc_q = _doc_db.query(DBDocument).filter(
                    DBDocument.session_id == session,
                    DBDocument.is_active == True
                )
                active_doc = _owner_session_filter(_session_doc_q, ctx.user).order_by(DBDocument.updated_at.desc()).first()
                if active_doc:
                    logger.info(f"[doc-inject] found by session fallback: title={active_doc.title!r}")
            # Last resort: the document the agent itself just created/edited
            # (tracked in-memory by the tool layer). This rescues docs that
            # got orphaned from their session (session_id NULL) — otherwise
            # neither lookup above can associate them with this conversation,
            # so the agent never sees what it just wrote. Guarded so we never
            # leak a doc that belongs to a DIFFERENT session.
            if not active_doc:
                try:
                    from src.agent_tools.document_tools import get_active_document
                    _mem_id = get_active_document()
                    if _mem_id:
                        _mem_q = _doc_db.query(DBDocument).filter(DBDocument.id == _mem_id)
                        cand = _owner_session_filter(_mem_q, ctx.user).first()
                        if cand and (not cand.session_id or cand.session_id == session):
                            active_doc = cand
                            logger.info(f"[doc-inject] found by in-memory active id: title={active_doc.title!r} (session_id={cand.session_id!r})")
                except Exception as _e:
                    logger.debug(f"[doc-inject] in-memory fallback failed: {_e}")
            if not active_doc:
                logger.info(f"[doc-inject] no active doc for session {session}")
            if active_doc:
                _doc_db.expunge(active_doc)
        except Exception as e:
            logger.warning(f"Failed to query active document: {e}")
        finally:
            _doc_db.close()

        # Build disabled-tools set from frontend toggles + user privileges
        disabled_tools = set()
        # Only disable bash when the caller *explicitly* set it to a falsy
        # value. When unset (None), defer to per-user privilege checks below.
        # Web search is per-turn opt-in: either the chat pre-search setting
        # (`use_web=true`) or agent web toggle (`allow_web_search=true`) must
        # explicitly enable it.
        if allow_bash is not None and str(allow_bash).lower() != "true":
            disabled_tools.add("bash")
        _explicit_web_intent = _explicit_web_intent or bool(_tool_intent and _tool_intent.category == "web")
        if is_web_search_explicitly_denied(allow_web_search) or not _search_enabled:
            disabled_tools.update(WEB_TOOL_NAMES)
        if _explicit_web_intent:
            # A direct lookup/search request should not drift into personal
            # tools or shell fallbacks. It can only use web_search/web_fetch
            # when the request's explicit web setting enabled them.
            disabled_tools.update({
                "bash", "python",
                "search_chats", "manage_skills", "manage_memory",
                "read_file", "write_file", "edit_file",
                "create_document", "edit_document", "update_document",
                "send_email", "reply_to_email",
                "manage_notes", "manage_calendar", "manage_tasks",
                "api_call",
            })
            if _search_enabled:
                disabled_tools.difference_update(WEB_TOOL_NAMES)
            else:
                disabled_tools.update(WEB_TOOL_NAMES)
        elif _search_enabled:
            disabled_tools.difference_update(WEB_TOOL_NAMES)

        # Nobody/incognito mode: deny tools that would expose the user's
        # persistent memory, past chats, or other identity-linked data.
        if incognito:
            disabled_tools.update({
                "manage_memory",      # persistent memory store
                "search_chats",       # past chat history
                "manage_skills",      # skill presets tied to user
                "create_session",
                "list_sessions",
                "manage_session",
                "send_to_session",
                "chat_with_model",
            })

        # Active email reader open → strip the tools that let the agent drift
        # away from the visible email or skip review. The only allowed compose
        # path is ui_control open_email_reply, which opens the same draft editor
        # as the Reply button with the generated body pre-filled. This prevents
        # the model from falling back to direct SMTP when it botches a draft
        # call, and prevents fake email-shaped documents.
        if active_email_ctx and active_email_ctx.get("uid"):
            disabled_tools.update({
                "create_document",
                "send_email",
                "reply_to_email",
                "mcp__email__send_email",
                "mcp__email__reply_to_email",
            })

        # Active email reader open -> strip tools that let the agent drift
        # away from the visible email, create fake email-shaped documents, or
        # send/reply directly through SMTP-style paths. The allowed reviewed
        # compose path should stay tied to the currently open email draft UI.
        if active_email_ctx and active_email_ctx.get("uid"):
            disabled_tools.update({
                "create_document",
                "send_email",
                "reply_to_email",
                "mcp__email__send_email",
                "mcp__email__reply_to_email",
            })

        # Enforce per-user privileges
        _privs = {}
        _user = ctx.user
        if _user and hasattr(request.app.state, 'auth_manager') and request.app.state.auth_manager:
            _privs = request.app.state.auth_manager.get_privileges(_user)
        if _privs:
            if not _privs.get("can_use_bash", True):
                disabled_tools.update({"bash", "python", "read_file", "write_file"})
            if not _privs.get("can_use_browser", True):
                disabled_tools.update(_BROWSER_MCP_TOOLS)
            if not _privs.get("can_use_documents", True):
                disabled_tools.update({"create_document", "edit_document", "update_document", "suggest_document"})
            if not _privs.get("can_generate_images", True):
                disabled_tools.add("generate_image")
            if not _privs.get("can_manage_memory", True):
                disabled_tools.update({"manage_memory", "manage_skills"})
            if not _privs.get("can_use_research", True):
                _research_flags["do"] = False
            if not _privs.get("can_use_agent", True):
                _effective_mode = 'chat'
                chat_mode = 'chat'
        # Global admin disabled tools
        from src.settings import get_setting
        _global_disabled = get_setting("disabled_tools", [])
        if _global_disabled and isinstance(_global_disabled, list):
            explicit_web_allowed = allow_web_search is not None and str(allow_web_search).lower() == "true"
            if explicit_web_allowed:
                disabled_tools.update(t for t in _global_disabled if t not in {"web_search", "web_fetch"})
            else:
                disabled_tools.update(_global_disabled)

        # Light auto-escalation: the user is in chat mode and just expressed a
        # notes/calendar/email intent. Grant the relevant managers but withhold
        # the heavy "do things on the computer" tools — otherwise the model
        # tries to shell out for a request that never needed it, then fails
        # (and looks broken when the shell is disabled).
        if auto_escalated:
            if not _workspace_agent_intent:
                disabled_tools.update({
                    "bash", "python", "read_file", "write_file", "builtin_browser",
                })
                if not _allow_browser_for_web_turn:
                    disabled_tools.update(_BROWSER_MCP_TOOLS)

        # Disable document tools in compare sessions — they break the pane UI
        if sess.name and sess.name.startswith("[CMP]"):
            disabled_tools.update({"create_document", "edit_document", "update_document"})

        # Compare mode: disable tools based on compare type
        if compare_mode:
            _compare_strip = {
                "create_document", "edit_document", "update_document",
                "chat_with_model", "create_session", "list_sessions",
                "send_to_session",
                "pipeline", "manage_session", "manage_memory", "list_models",
                "generate_image", "ui_control",
            }
            disabled_tools.update(_compare_strip)
            # In chat mode compare, disable ALL agent tools (no bash, python, file ops)
            if chat_mode == 'chat':
                disabled_tools.update({"bash", "python", "read_file", "write_file", "web_search", "web_fetch", "search_chats", "manage_tasks"})

        # Plan mode: investigate read-only, propose a plan, don't mutate. Block
        # every tool not on the read-only allowlist. (stream_agent_loop enforces
        # this again + drops MCP, so this is belt-and-suspenders.)
        if plan_mode:
            from src.tool_security import plan_mode_disabled_tools
            disabled_tools.update(plan_mode_disabled_tools())

        tool_policy = build_effective_tool_policy(
            disabled_tools=disabled_tools,
            last_user_message=message,
        )
        disabled_tools = tool_policy.all_disabled_names()
        research_blocked_by_policy = bool(
            tool_policy.blocks("trigger_research")
            or tool_policy.blocks("manage_research")
        )
        effective_do_research = bool(
            do_research and _research_flags["do"] and not research_blocked_by_policy
        )

        # Persist session mode after policy/privilege gates so blocked research
        # turns remain ordinary chat/agent streams and saved messages.
        _effective_mode = 'research' if effective_do_research else (chat_mode or 'chat')

        async def stream_with_save() -> AsyncGenerator[str, None]:
            nonlocal chat_mode
            nonlocal _agent_admission
            nonlocal _agent_admission_failure
            nonlocal _effective_mode

            if user_requested_agent:
                yield "data: " + json.dumps({
                    "type": "agent_prep",
                    "data": {
                        "runtime": "agent-admission-v1",
                        "label": (
                            "Understanding the request and selecting permitted sources"
                        ),
                    },
                }) + "\n\n"

            if _agent_admission_task is not None:
                try:
                    _agent_admission = await _agent_admission_task
                    logger.info(
                        "[agent-admission] explicit Agent path=%s execution_class=%s",
                        _agent_admission.path,
                        (
                            _agent_admission.assessment.execution_class
                            if _agent_admission.assessment is not None
                            else None
                        ),
                    )
                    if _agent_admission.use_normal_chat:
                        chat_mode = "chat"
                except Exception as _agent_admission_exc:
                    logger.exception(
                        "[agent-admission] semantic admission failed closed: %s",
                        _agent_admission_exc,
                    )
                    _agent_admission = None
                    _agent_admission_failure = (
                        f"{type(_agent_admission_exc).__name__}: "
                        f"{_agent_admission_exc}"
                    )

            _effective_mode = (
                'research'
                if effective_do_research
                else (chat_mode or 'chat')
            )
            if _effective_mode in ('agent', 'research', 'chat'):
                set_session_mode(session, _effective_mode)

            research_sources = None
            web_sources = ctx.web_sources

            # Register active stream for partial-save safety net
            _active_streams[session] = {"status": "streaming", "partial": "", "query": message, "is_research": effective_do_research, "mode": _effective_mode}

            # The client sent a workspace the server refused to bind (deleted
            # folder, file path, sensitive dir, filesystem root). Tell it up
            # front so the UI can clear the pill instead of displaying a
            # confinement that is not actually in effect.
            if workspace_rejected:
                yield f"data: {json.dumps({'type': 'workspace_rejected', 'data': {'path': workspace_rejected}})}\n\n"
                yield f"data: {json.dumps({'delta': 'The selected workspace could not be opened, so I stopped this workspace-scoped request instead of searching another root.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            if ctx.preprocessed.attachment_meta:
                yield f"data: {json.dumps({'type': 'attachments', 'data': ctx.preprocessed.attachment_meta})}\n\n"

            # Announce any docs auto-created during preprocess (e.g. fillable
            # PDF → editable markdown) so the editor pane switches to them
            # before the model starts streaming.
            for _opened in ctx.auto_opened_docs:
                yield (
                    f'data: {json.dumps({"type": "doc_update", **_opened})}\n\n'
                )

            if ctx.rag_sources:
                yield f"data: {json.dumps({'type': 'rag_sources', 'data': ctx.rag_sources})}\n\n"

            if web_sources:
                yield f"data: {json.dumps({'type': 'web_sources', 'data': web_sources})}\n\n"

            # Emit which memories were injected into context (captured before stream)
            if ctx.used_memories:
                yield f"data: {json.dumps({'type': 'memories_used', 'data': ctx.used_memories})}\n\n"

            # Run research as a background task (survives page refresh)
            if effective_do_research:
                _r_ep, _r_model, _r_headers = _resolve_research_endpoint(sess)
                _auth_keys = list(_r_headers.keys()) if _r_headers else []
                logger.info(f"Research endpoint resolved: model={_r_model}, endpoint={redact_url(_r_ep)}, auth_keys={_auth_keys}, sess_headers_keys={list(sess.headers.keys()) if isinstance(sess.headers, dict) else type(sess.headers)}")

                # Clarification round: only for very short/vague queries on first research message.
                # Skip in compare mode — each pane is a fresh session, so every one would
                # ask clarifying questions and the user would have to answer each pane
                # separately, breaking the parallel comparison.
                _prior_json = research_handler._get_session_json(session)
                _history_len = len(sess.history) if hasattr(sess, 'history') else 0
                _is_first_research = not _prior_json and _history_len <= 2 and not compare_mode

                if _is_first_research:
                    logger.info(f"First research message — asking clarifying questions for: {message[:60]}")
                    yield f'data: {json.dumps({"type": "model_info", "model": sess.model, "suffix": "Research"})}\n\n'
                    # Set DB mode to research_pending so the NEXT message auto-triggers research
                    set_session_mode(session, "research_pending")
                    ctx.messages.insert(0, {"role": "system", "content":
                        "The user wants to start deep web research. Before searching, ask 2-3 brief "
                        "clarifying questions to understand exactly what they want to know. For example: "
                        "what aspects matter most, are they comparing to something, what's their context "
                        "(moving, traveling, curiosity). Be conversational. Keep it short."
                    })
                    _skip_research = True
                else:
                    _skip_research = False

                if not _skip_research:
                    # Phase 2: Start actual research
                    def _on_research_done(_sid, _result, _sources, _findings):
                        """Persist research to DB when background task finishes."""
                        if incognito:
                            return
                        try:
                            _s = session_manager.get_session(_sid)
                            if not _s:
                                logger.warning(f"Session {_sid} expired before research completed")
                                return
                            _md = {"research": True, "model": _s.model}
                            if _sources:
                                _md["research_sources"] = _sources
                            if _findings:
                                _md["research_findings"] = _findings
                            _clean_res, _md = clean_thinking_for_save(_result, _md)
                            _s.add_message(ChatMessage("assistant", _clean_res, metadata=_md))
                            session_manager.save_sessions()
                            logger.info(f"Research result persisted to DB for session {_sid}")
                        except Exception as _e:
                            logger.error(f"Failed to persist research to DB: {_e}")

                    # Check for prior research to continue from
                    _prior_report = ""
                    _prior_findings = None
                    _prior_urls = None
                    _prior_json = research_handler._get_session_json(session)
                    if _prior_json:
                        _prior_report = _prior_json.get("raw_report", "")
                        _prior_findings = _prior_json.get("raw_findings")
                        _src_urls = {s.get("url", "") for s in (_prior_json.get("sources") or []) if s.get("url")}
                        _prior_urls = _src_urls if _src_urls else None
                        if _prior_report:
                            logger.info(f"Continuing research for session {session} with {len(_src_urls)} prior URLs")

                    # Synthesize conversation into a focused research query
                    _research_query = await research_handler.synthesize_query(
                        sess, message, _r_ep, _r_model, _r_headers,
                    )
                    logger.info(f"Research query: {_research_query[:120]}")

                    research_handler.start_research(
                        session, _research_query, _r_ep, _r_model,
                        llm_headers=_r_headers,
                        prior_report=_prior_report,
                        prior_findings=_prior_findings,
                        prior_urls=_prior_urls,
                        on_complete=_on_research_done,
                        owner=_user,
                    )

                    _heartbeat_counter = 0
                    _last_progress = {}
                    _sent_avg = False
                    while True:
                        status = research_handler.get_status(session)
                        if not status or status["status"] != "running":
                            break
                        progress = status.get("progress", {})
                        if progress and progress != _last_progress:
                            _last_progress = progress
                            if not _sent_avg:
                                _sent_avg = True
                                progress = dict(progress)
                                progress["started_at"] = status.get("started_at")
                                avg = status.get("avg_duration")
                                if avg:
                                    progress["avg_duration"] = avg
                            yield f"data: {json.dumps({'type': 'research_progress', 'data': progress})}\n\n"
                            _heartbeat_counter = 0
                        else:
                            _heartbeat_counter += 1
                            yield f": heartbeat {_heartbeat_counter}\n\n"
                        await asyncio.sleep(1.0)

                    research_sources = research_handler.get_sources(session)
                    if research_sources:
                        yield f"data: {json.dumps({'type': 'research_sources', 'data': research_sources})}\n\n"

                    research_findings = research_handler.get_raw_findings(session)
                    if research_findings:
                        yield f"data: {json.dumps({'type': 'research_findings', 'data': research_findings})}\n\n"

                    # Signal frontend to fetch and render the research result
                    yield f"data: {json.dumps({'type': 'research_done', 'data': {'session_id': session}})}\n\n"
                    yield "data: [DONE]\n\n"
                    research_handler.clear_result(session)
                    _stream_set(session, status="done")
                    _active_streams.pop(session, None)
                    return

            messages = _ensure_current_request_is_latest_user(ctx.messages, message)

            # Auto-compact notification
            if ctx.was_compacted:
                yield f"data: {json.dumps({'type': 'compacted', 'context_length': ctx.context_length})}\n\n"
            if ctx.context_trimmed and not ctx.was_compacted:
                yield f"data: {json.dumps({'type': 'context_trimmed', 'data': {'context_length': ctx.context_length, 'messages_before': ctx.context_messages_before_trim, 'messages_after': ctx.context_messages_after_trim, 'tokens_before': ctx.context_tokens_before_trim, 'tokens_after': ctx.context_tokens_after_trim}})}\n\n"

            full_response = ""
            thinking_response = ""
            last_metrics = None

            # Configured fallback chain for the default chat model. Tried in
            # order if the session's primary model fails before producing
            # output. Resolved once per request.
            try:
                from src.endpoint_resolver import resolve_chat_fallback_candidates
                _fallback_candidates = resolve_chat_fallback_candidates(owner=_user)
            except Exception:
                _fallback_candidates = []

            # Send model name early so the frontend can show it during streaming
            _model_suffix = "Research" if effective_do_research else None
            _model_info = {"type": "model_info", "model": sess.model}
            if _model_suffix:
                _model_info["suffix"] = _model_suffix
            if ctx.preset.character_name:
                _model_info["character_name"] = ctx.preset.character_name
            yield f'data: {json.dumps(_model_info)}\n\n'

            if image_generation_session:
                from src.settings import get_setting
                if tool_policy.blocks("generate_image"):
                    _blocked_msg = tool_policy.reason_for("generate_image")
                    yield f'data: {json.dumps({"delta": _blocked_msg})}\n\n'
                    yield "data: [DONE]\n\n"
                    _active_streams.pop(session, None)
                    return
                if not get_setting("image_gen_enabled", True):
                    yield f'data: {json.dumps({"delta": "Image generation is disabled by the administrator."})}\n\n'
                    yield "data: [DONE]\n\n"
                    _active_streams.pop(session, None)
                    return
                from src.ai_interaction import do_edit_image, do_generate_image
                _user_msg = message or ""
                _image_upload = _first_image_attachment(chat_handler, att_ids, owner=_user)
                _image_tool_name = "edit_image" if _image_upload else "generate_image"
                yield f'data: {json.dumps({"type": "tool_start", "tool": _image_tool_name, "command": _user_msg[:100]})}\n\n'
                yield ": heartbeat\n\n"
                _progress_queue: asyncio.Queue = asyncio.Queue()

                async def _image_progress_callback(progress: Dict[str, Any]):
                    try:
                        _progress_queue.put_nowait(progress)
                    except Exception:
                        pass

                if _image_upload:
                    _img_task = asyncio.create_task(do_edit_image(
                        _user_msg,
                        _image_upload.get("path", ""),
                        model_spec=sess.model,
                        session_id=session,
                        owner=_user,
                        size="1024x1024",
                        progress_callback=_image_progress_callback,
                    ))
                else:
                    _img_task = asyncio.create_task(do_generate_image(f"{_user_msg}\n{sess.model}\n512x512", session, owner=_user))
                _img_started = time.time()
                _img_tick = 0
                while not _img_task.done():
                    try:
                        _progress = await asyncio.wait_for(_progress_queue.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        _progress = None
                    _img_tick += 1
                    _elapsed = int(time.time() - _img_started)
                    _label = "Editing image" if _image_upload else "Generating image"
                    yield ": image generation still running\n\n"
                    _progress_data = {"type": "tool_progress", "tool": _image_tool_name, "message": f"{_label}… {_elapsed}s", "elapsed": _elapsed, "tick": _img_tick}
                    if isinstance(_progress, dict) and _progress.get("total"):
                        _step = int(_progress.get("step") or 0)
                        _total = int(_progress.get("total") or 0)
                        _percent = _progress.get("percent")
                        _progress_data.update({
                            "step": _step,
                            "total": _total,
                            "percent": _percent,
                            "message": f"{_label}… {_step}/{_total}",
                        })
                    yield f'data: {json.dumps(_progress_data)}\n\n'
                _img_result = await _img_task
                _img_output = _img_result.get("results", _img_result.get("error", ""))
                _img_tool_data = {"type": "tool_output", "tool": _image_tool_name, "command": _user_msg[:100], "output": _img_output, "exit_code": 0 if "error" not in _img_result else 1}
                for _k in ("image_url", "image_id", "image_prompt", "image_model", "image_size", "image_quality"):
                    if _k in _img_result:
                        _img_tool_data[_k] = _img_result[_k]
                if _image_upload:
                    _img_tool_data["source_image"] = {
                        "id": _image_upload.get("id"),
                        "name": _image_upload.get("name") or _image_upload.get("original_name"),
                    }
                yield f'data: {json.dumps(_img_tool_data)}\n\n'
                if _img_result.get("image_url"):
                    _img_event = {"type": "generated_image", "url": _img_result.get("image_url")}
                    for _k in ("image_url", "image_id", "image_prompt", "image_model", "image_size", "image_quality"):
                        if _img_result.get(_k):
                            _img_event[_k] = _img_result[_k]
                    yield f'data: {json.dumps(_img_event)}\n\n'
                _desc = _img_result.get("results", _img_result.get("error", "Image generation complete"))
                full_response = _desc
                yield f'data: {json.dumps({"delta": _desc})}\n\n'
                # Save to session history
                if not incognito:
                    _ev = {"round": 1, "tool": _image_tool_name, "command": _user_msg[:100], "output": _img_output, "exit_code": 0 if "error" not in _img_result else 1}
                    for _ek in ("image_url", "image_id", "image_prompt", "image_model", "image_size", "image_quality"):
                        if _img_result.get(_ek):
                            _ev[_ek] = _img_result[_ek]
                    if _image_upload:
                        _ev["source_image_id"] = _image_upload.get("id")
                        _ev["source_image_name"] = _image_upload.get("name") or _image_upload.get("original_name")
                    sess.add_message(ChatMessage("assistant", full_response, metadata={"tool_events": [_ev], "model": sess.model}))
                    session_manager.save_sessions()
                yield f'data: {json.dumps({"type": "metrics", "data": {"total_time": 0}})}\n\n'
                yield "data: [DONE]\n\n"
                _active_streams.pop(session, None)
                return
            elif chat_mode == "chat":
                _chat_start = time.time()
                _answered_by = None  # set if the selected model failed and a fallback answered
                _requested_model = sess.model
                _actual_model = None
                # ── Chat mode: call stream_llm directly, NO tools, NO document access ──
                try:
                    _chat_candidates = [(sess.endpoint_url, sess.model, sess.headers)] + _fallback_candidates
                    async for chunk in stream_llm_with_fallback(
                        _chat_candidates,
                        messages,
                        temperature=ctx.preset.temperature,
                        # Respect the preset; 0/unset = let the server decide (no
                        # cap), matching agent mode. The old hard 4096 fallback
                        # truncated reasoning models mid-<think> — they'd burn the
                        # whole budget thinking and never emit the answer (seen in
                        # Compare on heavy generation prompts).
                        max_tokens=ctx.preset.max_tokens,
                        prompt_type=preset_id,
                        tools=None,
                        session_id=session,
                    ):
                        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                            try:
                                data = json.loads(chunk[6:])
                                if "delta" in data:
                                    # Reasoning tokens arrive flagged thinking:true.
                                    # Forward them so the client can show a thinking
                                    # indicator, but don't fold them into the saved
                                    # reply (mirrors the rewrite path below).
                                    if data.get("thinking"):
                                        thinking_response += data["delta"]
                                    else:
                                        full_response += data["delta"]
                                        _stream_set(session, partial=full_response)
                                    yield chunk
                                elif data.get("type") == "fallback":
                                    # Selected model failed; a fallback answered.
                                    # Forward the notice and remember the real model.
                                    _answered_by = data.get("answered_by") or _answered_by
                                    _actual_model = _actual_model or _answered_by
                                    data["selected_model"] = data.get("selected_model") or _requested_model
                                    yield chunk
                                elif data.get("type") == "model_actual":
                                    _actual_model = data.get("model") or _actual_model
                                    data["requested_model"] = _requested_model
                                    yield f'data: {json.dumps(data)}\n\n'
                                elif data.get("type") == "usage":
                                    last_metrics = data.get("data", {})
                                    _reported_model = last_metrics.get("model")
                                    last_metrics["requested_model"] = _requested_model
                                    last_metrics["model"] = _reported_model or _actual_model or _answered_by or _requested_model
                                    if ctx.context_trimmed:
                                        last_metrics["context_trimmed"] = True
                                        last_metrics["context_messages_before_trim"] = ctx.context_messages_before_trim
                                        last_metrics["context_messages_after_trim"] = ctx.context_messages_after_trim
                                        last_metrics["context_tokens_before_trim"] = ctx.context_tokens_before_trim
                                        last_metrics["context_tokens_after_trim"] = ctx.context_tokens_after_trim
                                    request_context_tokens = ctx.context_tokens_after_trim or estimate_tokens(messages)
                                    last_metrics["request_context_tokens"] = request_context_tokens
                                    if ctx.context_length and request_context_tokens:
                                        pct = min(round((request_context_tokens / ctx.context_length) * 100, 1), 100.0)
                                        last_metrics["context_percent"] = pct
                                        last_metrics["context_length"] = ctx.context_length
                                    # The frontend reads `tokens_per_second`; the raw usage event
                                    # carries the backend's true gen speed as `gen_tps` (llama.cpp
                                    # timings). Map it through so this direct-chat path shows real
                                    # t/s instead of "n/a" → falling back to a bare token count.
                                    if last_metrics.get("gen_tps") and not last_metrics.get("tokens_per_second"):
                                        last_metrics["tokens_per_second"] = last_metrics["gen_tps"]
                                        last_metrics["tps_source"] = "backend"
                                    # Wall-clock response time for the stats popup ("Time").
                                    last_metrics.setdefault("response_time", round(time.time() - _chat_start, 2))
                                    yield f'data: {json.dumps({"type": "metrics", "data": last_metrics})}\n\n'
                            except json.JSONDecodeError:
                                yield chunk
                        elif chunk.startswith("event: error"):
                            logger.warning(f"Stream error for {sess.model} on {sess.endpoint_url}: {chunk!r}")
                            yield chunk
                        elif chunk.startswith("event: "):
                            yield chunk
                        elif chunk == "data: [DONE]\n\n":
                            # Generate fallback metrics if LLM didn't send usage
                            if not last_metrics and full_response:
                                _elapsed = time.time() - _chat_start
                                _est_in = estimate_tokens(messages)
                                _est_out = len(full_response) // 4
                                _tps = round(_est_out / _elapsed, 2) if _elapsed > 0 else 0
                                _ctx_pct = min(round((_est_in / ctx.context_length) * 100, 1), 100.0) if ctx.context_length else 0
                                last_metrics = {
                                    "response_time": round(_elapsed, 2),
                                    "input_tokens": _est_in,
                                    "output_tokens": _est_out,
                                    "tokens_per_second": _tps,
                                    "request_context_tokens": _est_in,
                                    "context_percent": _ctx_pct,
                                    "context_length": ctx.context_length,
                                    "model": _actual_model or _answered_by or _requested_model,
                                    "requested_model": _requested_model,
                                    "usage_source": "estimated",
                                }
                                yield f'data: {json.dumps({"type": "metrics", "data": last_metrics})}\n\n'
                            if full_response:
                                _metrics_to_save = dict(last_metrics or {})
                                if thinking_response.strip() and not _metrics_to_save.get("thinking"):
                                    _metrics_to_save["thinking"] = thinking_response.strip()
                                _saved_id = save_assistant_response(
                                    sess, session_manager, session, full_response, _metrics_to_save,
                                    character_name=ctx.preset.character_name,
                                    web_sources=web_sources,
                                    rag_sources=ctx.rag_sources,
                                    research_sources=research_sources,
                                    used_memories=ctx.used_memories,
                                    do_research=effective_do_research,
                                    incognito=incognito,
                                )
                                if _saved_id:
                                    yield f'data: {json.dumps({"type": "message_saved", "id": _saved_id})}\n\n'
                                run_post_response_tasks(
                                    sess, session_manager, session, message, full_response,
                                    _metrics_to_save, ctx.uprefs, memory_manager, memory_vector, webhook_manager,
                                    incognito=incognito, compare_mode=compare_mode,
                                    character_name=ctx.preset.character_name,
                                    owner=_user,
                                    allow_background_extraction=not tool_policy.block_all_tool_calls,
                                )
                            _stream_set(session, status="done")
                            yield chunk
                except (asyncio.CancelledError, GeneratorExit):
                    if full_response and not incognito:
                        logger.info("Client disconnected mid-stream (chat mode) for session %s, saving partial (%d chars)", session, len(full_response))
                        _stopped_content, _stopped_md = clean_thinking_for_save(
                            full_response,
                            {
                                "stopped": True,
                                "model": _actual_model or _answered_by or _requested_model,
                                "requested_model": _requested_model,
                            },
                        )
                        sess.add_message(ChatMessage("assistant", _stopped_content, metadata=_stopped_md))
                        session_manager.save_sessions()
                    raise
                finally:
                    _active_streams.pop(session, None)
            else:
                # ── Agent mode: full agent loop with tools ──
                _agent_rounds = 0
                _agent_tool_calls = 0
                _answered_by = None  # set if the selected model failed and a fallback answered
                _requested_model = sess.model
                _actual_model = None
                try:
                    from src.settings import get_setting
                    from src.agent_tools import MAX_AGENT_ROUNDS as _DEFAULT_ROUNDS
                    # Per-message tool budget from settings; guard defensively in
                    # case settings.json was hand-edited to a non-numeric value
                    # (the HTTP admin endpoint validates, but direct edits bypass
                    # it). 0 = unlimited, matching auth_routes set_settings().
                    try:
                        _tool_budget = int(get_setting("agent_max_tool_calls", 0))
                    except (TypeError, ValueError):
                        _tool_budget = 0
                    # Per-message round cap from settings; clamp defensively in
                    # case settings.json was hand-edited to a bad value.
                    try:
                        _max_rounds = int(get_setting("agent_max_rounds", _DEFAULT_ROUNDS) or _DEFAULT_ROUNDS)
                    except (TypeError, ValueError):
                        _max_rounds = _DEFAULT_ROUNDS
                    _max_rounds = max(1, min(_max_rounds, 200))

                    _forced_tools = None
                    if _search_enabled:
                        _forced_tools = set(WEB_TOOL_NAMES)
                        if _explicit_browser_intent:
                            _forced_tools |= set(_BROWSER_MCP_TOOLS)
                    elif _explicit_browser_intent:
                        _forced_tools = set(_BROWSER_MCP_TOOLS)

                    _forced_tools = None
                    if allow_web_search is not None and str(allow_web_search).lower() == "true":
                        _forced_tools = {"web_search", "web_fetch"}

                    # RUNTIME_ROUTER_V013D_DECISION_START
                    _runtime_decision = _route_agent_runtime_for_request(
                        message,
                        _agent_langgraph_architecture_enabled,
                    )

                    _router_agent_runtime_requested = (
                        _agent_langgraph_architecture_enabled
                        and _runtime_decision is not None
                        and chat_mode == "agent"
                        and os.getenv("ODYSSEUS_AGENT_RUNTIME", "langgraph").strip().lower()
                        in {"contract", "general", "langgraph"}
                    )
                    if _runtime_decision is not None:
                        logger.info(
                            "[odysseus-router] intent=%s runtime=%s safe=%s reason=%s",
                            _runtime_decision.intent,
                            _runtime_decision.target_runtime,
                            _runtime_decision.is_safe_readonly,
                            _runtime_decision.reason,
                        )
                    else:
                        logger.info(
                            "[odysseus-router] skipped because Agent LangGraph architecture is disabled"
                        )
                    # RUNTIME_ROUTER_V013D_DECISION_END

                    # P3_C5_EXPLICIT_GRAPH_ROUTE_START
                    # Explicit Agent semantic admission is authoritative before
                    # backend-specific compatibility routers. single_graph enters
                    # the General Supervisor. multi_graph enters the dependency-
                    # aware coordinator. Legacy remains the failure fallback.
                    _explicit_agent_single_graph = bool(
                        user_requested_agent
                        and _agent_admission is not None
                        and _agent_admission.use_single_graph
                    )
                    _explicit_agent_multi_graph = bool(
                        user_requested_agent
                        and _agent_admission is not None
                        and _agent_admission.use_multi_graph
                        and not compare_mode
                        and not effective_do_research
                        and not plan_mode
                        and not approved_plan
                        and _langgraph_multi_graph_allowed_by_settings()
                    )
                    _langgraph_first_ambiguous_agent = (
                        _agent_langgraph_architecture_enabled
                        and chat_mode == "agent"
                        and (
                            _explicit_agent_single_graph
                            or (
                                _runtime_decision is not None
                                and _runtime_decision.intent == "legacy_exploratory"
                                and _runtime_decision.target_runtime == "legacy"
                                and not is_casual_conversation_request(message)
                            )
                        )
                    )
                    # P3_C5_EXPLICIT_GRAPH_ROUTE_END

                    # CONTRACT_RUNTIME_V012D_AGENT_BRANCH_START
                    _contract_requested_agent = (
                        _router_agent_runtime_requested
                        and _runtime_decision is not None
                        and _runtime_decision.target_runtime == "contract"
                    )
                    _contract_allowed_agent = (
                        _contract_requested_agent
                        and not _explicit_agent_single_graph
                        and not _explicit_agent_multi_graph
                        and contract_runtime_enabled()
                        and _runtime_decision.is_safe_readonly
                        and bool(workspace)
                        and not compare_mode
                        and not effective_do_research
                        and not _is_image_generation_session(sess, owner=_user)
                    )
                    # KNOWLEDGE_RUNTIME_V016D_CHAT_BRANCH_START
                    try:
                        _knowledge_runtime_enabled = os.getenv("ODYSSEUS_ENABLE_KNOWLEDGE_RUNTIME", "1").strip().lower() not in {"0", "false", "no", "off"}
                        _knowledge_intercept = (
                            should_intercept_knowledge_chat(message)
                            or (
                                bool(workspace)
                                and should_intercept_workspace_knowledge(message)
                            )
                        )
                        if (
                            _agent_langgraph_architecture_enabled
                            and chat_mode == "agent"
                            and _knowledge_runtime_enabled
                            and _knowledge_intercept
                            and not (
                                _explicit_agent_single_graph
                                or _explicit_agent_multi_graph
                            )
                        ):
                            logger.info("[knowledge-runtime] routing request to v0.16-D knowledge chat bridge")
                            _knowledge_llm_config = knowledge_llm_config_from_runtime_locals(locals())
                            if _knowledge_llm_config.get("model") and not _knowledge_llm_config.get("url"):
                                _knowledge_llm_config["url"] = "http://127.0.0.1:11434/v1/chat/completions"
                                _knowledge_llm_config["url_source"] = "local_ollama_openai_compat_fallback"
                            if workspace or is_semantic_vault_search_request(message):
                                _knowledge_full_response, _knowledge_frames, _knowledge_tool_events, _knowledge_metrics = run_knowledge_semantic_chat_turn(
                                    message,
                                    workspace=workspace or None,
                                    llm_config=_knowledge_llm_config,
                                )
                            else:
                                _knowledge_full_response, _knowledge_frames, _knowledge_tool_events, _knowledge_metrics = run_knowledge_chat_turn(message, llm_config=_knowledge_llm_config)
                            for _knowledge_chunk in _knowledge_frames:
                                yield _knowledge_chunk
                            try:
                                _knowledge_db_id = save_assistant_response(
                                    sess,
                                    session_manager,
                                    session,
                                    _knowledge_full_response,
                                    _knowledge_metrics,
                                    character_name=ctx.preset.character_name,
                                    tool_events=_knowledge_tool_events,
                                    incognito=incognito,
                                )
                                if _knowledge_db_id:
                                    yield "data: " + json.dumps({"type": "message_saved", "id": _knowledge_db_id}, ensure_ascii=False) + "\n\n"
                            except Exception as _knowledge_save_exc:
                                logger.exception("[knowledge-runtime] failed to save knowledge response: %s", _knowledge_save_exc)
                                yield "data: " + json.dumps({
                                    "type": "tool_output",
                                    "tool": "knowledge_save_error",
                                    "command": "Save knowledge response",
                                    "output": str(_knowledge_save_exc),
                                    "exit_code": 1,
                                }, ensure_ascii=False) + "\n\n"
                            yield "data: [DONE]\n\n"
                            return
                    except Exception as _knowledge_runtime_exc:
                        if workspace:
                            logger.exception(
                                "[knowledge-runtime] workspace bridge failed; stopping without a global fallback: %s",
                                _knowledge_runtime_exc,
                            )
                            yield "data: " + json.dumps({
                                "delta": (
                                    "Workspace knowledge retrieval failed, so I stopped "
                                    "without searching Ideaverse or another global root."
                                )
                            }, ensure_ascii=False) + "\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        logger.exception("[knowledge-runtime] v0.16-D bridge failed; falling back to existing runtime: %s", _knowledge_runtime_exc)
                    # KNOWLEDGE_RUNTIME_V016D_CHAT_BRANCH_END

                    if (
                        _agent_langgraph_architecture_enabled
                        and _runtime_decision is not None
                        and _runtime_decision.intent == "write_mutation"
                        and not (
                            _explicit_agent_single_graph
                            or _explicit_agent_multi_graph
                        )
                    ):
                        _policy_result = evaluate_mutation_request(message)
                        _policy_call_id = "mutation-policy-decision"
                        _policy_output = json.dumps(
                            _policy_result,
                            ensure_ascii=False,
                            indent=2,
                        )
                        yield _runtime_trace_sse_frame(message, _runtime_decision)
                        yield "data: " + json.dumps({
                            "type": "tool_start",
                            "id": _policy_call_id,
                            "call_id": _policy_call_id,
                            "tool": "policy_decision",
                            "command": "Central mutation policy",
                        }, ensure_ascii=False) + "\n\n"
                        yield "data: " + json.dumps({
                            "type": "tool_output",
                            "id": _policy_call_id,
                            "call_id": _policy_call_id,
                            "tool": "policy_decision",
                            "command": "Central mutation policy",
                            "output": _policy_output,
                            "exit_code": 0,
                            "policy_decision": _policy_result,
                        }, ensure_ascii=False) + "\n\n"
                        _policy_answer = str(_policy_result["answer"])
                        yield "data: " + json.dumps(
                            {"delta": _policy_answer},
                            ensure_ascii=False,
                        ) + "\n\n"
                        yield "data: " + json.dumps({
                            "type": "metrics",
                            "data": {
                                "runtime": "central-mutation-policy-v1",
                                "model": "deterministic-policy-engine",
                                "requested_model": sess.model,
                                "langgraph_runtime": True,
                                "policy_action": _policy_result["decision"]["action"],
                                "tool_calls": 0,
                            },
                        }, ensure_ascii=False) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    if _contract_allowed_agent:
                        logger.info(
                            "[contract-bridge] routing read-only agent request to deterministic contract runtime"
                        )
                        yield _runtime_trace_sse_frame(message, _runtime_decision)
                        _runtime_tool_events = [_runtime_trace_tool_event(message, _runtime_decision)]
                    elif _contract_requested_agent:
                        logger.info(
                            "[contract-bridge] declined by env/session gate: enabled=%s safe=%s compare=%s research=%s image=%s",
                            contract_runtime_enabled(),
                            _runtime_decision.is_safe_readonly,
                            compare_mode,
                            effective_do_research,
                            _is_image_generation_session(sess, owner=_user),
                        )

                    if _contract_allowed_agent:
                        _contract_start = time.time()
                        _reported_model = "deterministic-contract-runtime"
                        _actual_model = _reported_model
                        async for chunk in stream_contract_runtime_legacy_sse(
                            message,
                            workspace=workspace,
                        ):
                            if chunk.startswith("data: "):
                                payload = chunk[len("data: "):].strip()
                                if payload == "[DONE]":
                                    if not last_metrics:
                                        last_metrics = {
                                            "model": _reported_model,
                                            "requested_model": sess.model,
                                            "response_time": round(time.time() - _contract_start, 2),
                                            "runtime": "v0.12d-contract-bridge",
                                        }
                                        yield f'data: {json.dumps({"type": "metrics", "data": last_metrics})}\n\n'
                                    if full_response and not incognito:
                                        _saved_id = save_assistant_response(
                                            sess, session_manager, session, full_response, last_metrics,
                                            character_name=ctx.preset.character_name,
                                            tool_events=_runtime_tool_events,
                                        )
                                        if _saved_id:
                                            yield f'data: {json.dumps({"type": "message_saved", "id": _saved_id})}\n\n'
                                        # Contract runtime is deterministic/read-only.
                                        # Do not run post-response LLM extraction here; it can call
                                        # the selected model after the contract answer is already done.
                                        logger.info("[contract-bridge-agent-branch] skipped post-response LLM tasks")
                                    _stream_set(session, status="done")
                                    yield chunk
                                    return

                                try:
                                    data = json.loads(payload)
                                    if "delta" in data:
                                        full_response += data["delta"]
                                        _stream_set(session, partial=full_response)
                                    elif data.get("type") == "tool_output":
                                        _append_runtime_tool_event(_runtime_tool_events, data)
                                    elif data.get("type") == "metrics":
                                        last_metrics = data.get("data", {}) or {}
                                        last_metrics["requested_model"] = sess.model
                                        last_metrics["runtime"] = "v0.12d-contract-bridge"
                                        yield f'data: {json.dumps({"type": "metrics", "data": last_metrics})}\n\n'
                                        continue
                                except json.JSONDecodeError:
                                    pass

                            yield chunk
                        return
                    # CONTRACT_RUNTIME_V012D_AGENT_BRANCH_END

                    # GENERAL_RUNTIME_V013_AGENT_BRANCH_START
                    _general_requested_agent = (
                        _router_agent_runtime_requested
                        and _runtime_decision is not None
                        and _runtime_decision.target_runtime == "general"
                    )
                    _general_allowed_agent = (
                        _general_requested_agent
                        and not _explicit_agent_single_graph
                        and not _explicit_agent_multi_graph
                        and general_runtime_enabled()
                        and _runtime_decision.is_safe_readonly
                        and bool(workspace)
                        and not compare_mode
                        and not effective_do_research
                        and not _is_image_generation_session(sess, owner=_user)
                    )

                    if _general_allowed_agent:
                        logger.info("[general-bridge] routing safe read-only agent request to v0.13 general runtime")
                        yield _runtime_trace_sse_frame(message, _runtime_decision)
                        _runtime_tool_events = [_runtime_trace_tool_event(message, _runtime_decision)]
                        _general_start = time.time()
                        _reported_model = "deterministic-general-scaffold"
                        _actual_model = _reported_model

                        async for chunk in stream_general_runtime_legacy_sse(
                            message,
                            workspace=workspace,
                        ):
                            if chunk.startswith("data: "):
                                payload = chunk[len("data: "):].strip()
                                if payload == "[DONE]":
                                    if not last_metrics:
                                        last_metrics = {
                                            "model": _reported_model,
                                            "requested_model": sess.model,
                                            "response_time": round(time.time() - _general_start, 2),
                                            "runtime": "v0.13-general-bridge",
                                        }
                                        yield f'data: {json.dumps({"type": "metrics", "data": last_metrics})}\n\n'

                                    if full_response and not incognito:
                                        _saved_id = save_assistant_response(
                                            sess, session_manager, session, full_response, last_metrics,
                                            character_name=ctx.preset.character_name,
                                            tool_events=_runtime_tool_events,
                                        )
                                        if _saved_id:
                                            yield f'data: {json.dumps({"type": "message_saved", "id": _saved_id})}\n\n'

                                        logger.info("[general-bridge] skipped post-response LLM tasks")

                                    _stream_set(session, status="done")
                                    yield chunk
                                    return

                                try:
                                    data = json.loads(payload)
                                    if "delta" in data:
                                        full_response += data["delta"]
                                        _stream_set(session, partial=full_response)
                                    elif data.get("type") == "tool_output":
                                        _append_runtime_tool_event(_runtime_tool_events, data)
                                    elif data.get("type") == "metrics":
                                        last_metrics = data.get("data", {}) or {}
                                        last_metrics["requested_model"] = sess.model
                                        last_metrics["runtime"] = "v0.13-general-bridge"
                                        yield f'data: {json.dumps({"type": "metrics", "data": last_metrics})}\n\n'
                                        continue
                                except json.JSONDecodeError:
                                    pass

                            yield chunk
                        return

                    elif _general_requested_agent:
                        logger.info(
                            "[general-bridge] declined by env/session gate: enabled=%s safe=%s compare=%s research=%s image=%s",
                            general_runtime_enabled(),
                            _runtime_decision.is_safe_readonly,
                            compare_mode,
                            effective_do_research,
                            _is_image_generation_session(sess, owner=_user),
                        )
                    # GENERAL_RUNTIME_V013_AGENT_BRANCH_END

                    # RUNTIME_TRACE_V015F_LEGACY_DECLINE_START
                    _legacy_trace_needed = (
                        not _explicit_agent_single_graph
                        and not _explicit_agent_multi_graph
                        and _router_agent_runtime_requested
                        and (
                            _runtime_decision.target_runtime == "legacy"
                            or (_contract_requested_agent and not _contract_allowed_agent)
                            or (_general_requested_agent and not _general_allowed_agent)
                        )
                    )
                    if _legacy_trace_needed:
                        logger.info(
                            "[runtime-trace] emitting visible legacy/decline route trace: intent=%s runtime=%s safe=%s reason=%s",
                            _runtime_decision.intent,
                            _runtime_decision.target_runtime,
                            _runtime_decision.is_safe_readonly,
                            _runtime_decision.reason,
                        )
                        yield _runtime_trace_sse_frame(message, _runtime_decision)
                        _runtime_tool_events = [_runtime_trace_tool_event(message, _runtime_decision)]
                    # RUNTIME_TRACE_V015F_LEGACY_DECLINE_END

                    if _agent_admission_failure:
                        _selected_agent_stream = partial(
                            _stream_agent_admission_failure,
                            admission_failure=_agent_admission_failure,
                        )
                    elif _explicit_agent_multi_graph:
                        _selected_agent_stream = partial(
                            _stream_multi_graph_fail_closed,
                            admission_assessment=(
                                _agent_admission.assessment
                            ),
                        )
                        logger.info(
                            "[multi-graph] explicit Agent request entering "
                            "dependency-aware coordinator"
                        )
                    elif _langgraph_first_ambiguous_agent:
                        _selected_agent_stream = partial(
                            _stream_langgraph_first_with_legacy_fallback,
                            admission_assessment=(
                                _agent_admission.assessment
                                if _agent_admission is not None
                                else None
                            ),
                        )
                        logger.info(
                            "[langgraph-first] ambiguous Agent request entering native LangGraph runtime"
                        )
                    else:
                        _selected_agent_stream = stream_agent_loop

                    async for chunk in _selected_agent_stream(
                        sess.endpoint_url,
                        sess.model,
                        messages,
                        headers=sess.headers,
                        temperature=ctx.preset.temperature,
                        max_tokens=ctx.preset.max_tokens,
                        prompt_type=preset_id,
                        max_tool_calls=_tool_budget,
                        max_rounds=_max_rounds,
                        context_length=ctx.context_length,
                        active_document=active_doc,
                        active_email=active_email_ctx,
                        session_id=session,
                        disabled_tools=disabled_tools if disabled_tools else None,
                        tool_policy=tool_policy,
                        owner=_user,
                        fallbacks=_fallback_candidates,
                        plan_mode=plan_mode,
                        approved_plan=approved_plan or None,
                        workspace=workspace or None,
                        forced_tools=_forced_tools,
                        uploaded_files=ctx.uploaded_files,
                    ):
                        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                            try:
                                data = json.loads(chunk[6:])
                                if "delta" in data:
                                    # Reasoning tokens arrive flagged thinking:true.
                                    # Forward them for the live indicator, but keep
                                    # them out of the saved reply (same as chat mode).
                                    if data.get("thinking"):
                                        thinking_response += data["delta"]
                                    else:
                                        full_response += data["delta"]
                                        _stream_set(session, partial=full_response)
                                    yield chunk
                                elif data.get("type") == "web_sources":
                                    web_sources = data.get("data", [])
                                    yield chunk
                                elif data.get("type") in (
                                    "tool_start", "tool_output", "agent_step",
                                    "doc_stream_open", "doc_stream_delta",
                                    "doc_update", "doc_suggestions", "ui_control",
                                    "rounds_exhausted", "budget_exceeded",
                                    "loop_breaker_triggered",
                                    "intent_nudge_exhausted",
                                    "ask_user",
                                    "plan_update",
                                    "runtime_run",
                                    "runtime_state",
                                    "runtime_snapshot",
                                    "runtime_terminal",
                                ):
                                    if data.get("type") == "agent_step":
                                        _agent_rounds = max(_agent_rounds, data.get("round", 1))
                                    elif data.get("type") == "tool_start":
                                        _agent_tool_calls += 1
                                    yield chunk
                                elif data.get("type") == "fallback":
                                    # Selected model failed; a fallback answered.
                                    # Forward the notice and remember the real
                                    # model so metrics reflect it, not the masked
                                    # selected model.
                                    _answered_by = data.get("answered_by") or _answered_by
                                    _actual_model = _actual_model or _answered_by
                                    data["selected_model"] = data.get("selected_model") or _requested_model
                                    yield chunk
                                elif data.get("type") == "model_actual":
                                    _actual_model = data.get("model") or _actual_model
                                    data["requested_model"] = _requested_model
                                    yield f'data: {json.dumps(data)}\n\n'
                                elif data.get("type") == "metrics":
                                    last_metrics = data.get("data", {})
                                    _reported_model = last_metrics.get("model")
                                    last_metrics["requested_model"] = last_metrics.get("requested_model") or _requested_model
                                    if "presenting_model" in last_metrics:
                                        last_metrics["model"] = last_metrics.get(
                                            "presenting_model"
                                        )
                                    else:
                                        last_metrics["model"] = _reported_model or _actual_model or _answered_by or _requested_model
                                    if ctx.context_trimmed:
                                        last_metrics["context_trimmed"] = True
                                        last_metrics["context_messages_before_trim"] = ctx.context_messages_before_trim
                                        last_metrics["context_messages_after_trim"] = ctx.context_messages_after_trim
                                        last_metrics["context_tokens_before_trim"] = ctx.context_tokens_before_trim
                                        last_metrics["context_tokens_after_trim"] = ctx.context_tokens_after_trim
                                    yield f'data: {json.dumps({"type": "metrics", "data": last_metrics})}\n\n'
                            except json.JSONDecodeError:
                                yield chunk
                        elif chunk.startswith("event: "):
                            yield chunk
                        elif chunk == "data: [DONE]\n\n":
                            _has_tool_events = bool((last_metrics or {}).get("tool_events"))
                            if full_response or _has_tool_events:
                                _response_to_save = full_response or "Done."
                                _metrics_to_save = dict(last_metrics or {})
                                if thinking_response.strip() and not _metrics_to_save.get("thinking"):
                                    _metrics_to_save["thinking"] = thinking_response.strip()
                                _saved_id = save_assistant_response(
                                    sess, session_manager, session, _response_to_save, _metrics_to_save,
                                    character_name=ctx.preset.character_name,
                                    web_sources=web_sources,
                                    rag_sources=ctx.rag_sources,
                                    used_memories=ctx.used_memories,
                                    incognito=incognito,
                                )
                                if _saved_id:
                                    yield f'data: {json.dumps({"type": "message_saved", "id": _saved_id})}\n\n'
                                run_post_response_tasks(
                                    sess, session_manager, session, message, _response_to_save,
                                    _metrics_to_save, ctx.uprefs, memory_manager, memory_vector, webhook_manager,
                                    incognito=incognito, compare_mode=compare_mode,
                                    character_name=ctx.preset.character_name,
                                                            agent_rounds=_agent_rounds,
                                    agent_tool_calls=_agent_tool_calls,
                                    skills_manager=skills_manager,
                                    owner=_user,
                                    extract_skills=user_requested_agent,
                                    allow_background_extraction=not tool_policy.block_all_tool_calls,
                                )
                            _stream_set(session, status="done")
                            yield chunk
                except (asyncio.CancelledError, GeneratorExit):
                    # Client disconnected — save partial response. Wrap
                    # the save in its own try so an exception inside
                    # add_message / save_sessions doesn't mask the
                    # original CancelledError (which prevented the
                    # outer finally from running and left _active_streams
                    # with a stale entry).
                    try:
                        if full_response and not incognito:
                            logger.info("Client disconnected mid-stream for session %s, saving partial response (%d chars)", session, len(full_response))
                            _stopped_content2, _stopped_md2 = clean_thinking_for_save(
                                full_response,
                                {
                                    "stopped": True,
                                    "model": _actual_model or _answered_by or _requested_model,
                                    "requested_model": _requested_model,
                                },
                            )
                            sess.add_message(ChatMessage("assistant", _stopped_content2, metadata=_stopped_md2))
                            session_manager.save_sessions()
                    except Exception:
                        logger.exception("Failed to save partial response on disconnect (session %s)", session)
                    raise
                finally:
                    _active_streams.pop(session, None)

        async def _safe_stream() -> AsyncGenerator[str, None]:
            """Wrapper that guarantees _active_streams cleanup even if stream_with_save
            raises before reaching a mode-specific finally block."""
            try:
                async for chunk in stream_with_save():
                    yield chunk
            finally:
                _active_streams.pop(session, None)

        # Compare panes are short-lived, single-shot generations whose sessions
        # exist only to drive that one pane — there's nothing to "resume" and
        # the user expects the pane's Stop button (which aborts the fetch,
        # closing this SSE) to promptly cancel the upstream LLM call. Detaching
        # them would keep burning upstream tokens/compute after the pane is
        # stopped or the comparison is abandoned, and would surface a stale
        # "still streaming" /resume target for a session nobody will revisit.
        #
        # So: stream them directly (no agent_runs wrapping). Starlette cancels
        # the underlying async generator (raising CancelledError/GeneratorExit
        # inside it) as soon as it notices the client disconnected — which the
        # mode-specific except blocks above already handle by saving the
        # partial response exactly once. This stops the upstream call promptly
        # without waiting on the next streamed chunk.
        #
        # Normal chat/agent streams keep the DETACHED behavior below: they
        # survive the client closing the tab / navigating away. The SSE response just subscribes (replay
        # buffered output + live); dropping the SSE only removes a subscriber —
        # the run keeps going and saves the assistant message on completion
        # regardless. Reconnect via /api/chat/resume.
        if compare_mode:
            return StreamingResponse(_safe_stream(), media_type="text/event-stream")

        agent_runs.start(session, _safe_stream())
        return StreamingResponse(agent_runs.subscribe(session), media_type="text/event-stream")

    # ------------------------------------------------------------------ #
    # GET /api/chat/resume — reconnect to a detached run that's still going
    # (e.g. after reopening a session whose agent kept running in the background)
    # ------------------------------------------------------------------ #
    @router.get("/api/chat/resume/{session_id}")
    async def chat_resume(
        request: Request,
        session_id: str,
        after_cursor: int = Query(0, ge=0),
    ) -> StreamingResponse:
        _verify_session_owner(request, session_id)
        if agent_runs.is_active(session_id):
            return StreamingResponse(
                agent_runs.subscribe(session_id),
                media_type="text/event-stream",
            )
        store = _durable_checkpoint_store_or_503()
        state = store.recover_latest(
            session_id=session_id,
            owner=effective_user(request),
        )
        if state is None:
            raise HTTPException(404, "No recoverable run for this session")
        state = store.reconcile_state(state, executor_alive=False)
        recovered = LangGraphRunLifecycle.recover(
            checkpoint_store=store,
            thread_id=state["thread_id"],
            checkpoint_ns=state["checkpoint_namespace"],
        )
        if recovered is not None:
            await reconcile_owned_research_jobs(recovered)
            state = store.recover_latest(
                run_id=state["run_id"],
                owner=effective_user(request),
            ) or state
        replay_events = store.events_after(
            run_id=str(state["run_id"]),
            after_cursor=after_cursor,
        )
        return StreamingResponse(
            _replay_durable_langgraph_run(
                state,
                after_cursor=after_cursor,
                events=replay_events,
            ),
            media_type="text/event-stream",
        )

    # ------------------------------------------------------------------ #
    # POST /api/chat/stop — cancel a detached run (Stop button). Closing the SSE
    # no longer stops it (it's detached), so the Stop button must call this.
    # ------------------------------------------------------------------ #
    @router.post(
        "/api/chat/tool_confirmation/{session_id}"
    )
    async def resolve_langgraph_tool_confirmation(
        request: Request,
        session_id: str,
    ) -> Dict[str, Any]:
        """Resolve one exact LangGraph tool confirmation."""

        _verify_session_owner(request, session_id)

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON")

        call_id = str(body.get("call_id") or "").strip()
        fingerprint = str(
            body.get("fingerprint") or ""
        ).strip()
        approved = body.get("approved")

        if (
            not call_id
            or not fingerprint
            or not isinstance(approved, bool)
        ):
            raise HTTPException(
                400,
                "call_id, fingerprint, and boolean approved "
                "are required",
            )

        try:
            from src.agent_runtime.tool_confirmation_execution import (
                resolve_and_execute_tool_confirmation,
            )

            return resolve_and_execute_tool_confirmation(
                session_id=session_id,
                call_id=call_id,
                fingerprint=fingerprint,
                approved=approved,
                owner=effective_user(request),
                run_id=(
                    get_default_checkpoint_store()
                    .latest_run_record(
                        session_id=session_id,
                        owner=effective_user(request),
                    )
                    or {}
                ).get("run_id"),
            )

        except KeyError as exc:
            raise HTTPException(404, str(exc))

        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @router.post("/api/chat/stop/{session_id}")
    async def chat_stop(request: Request, session_id: str) -> Dict[str, Any]:
        _verify_session_owner(request, session_id)
        stopped = agent_runs.stop(session_id)
        if not stopped:
            store = _durable_checkpoint_store_or_503()
            state = store.recover_latest(
                session_id=session_id,
                owner=effective_user(request),
            )
            if state:
                state = store.reconcile_state(
                    state,
                    executor_alive=False,
                )
            if state and state.get("status") in {
                "PENDING",
                "RUNNING",
                "AWAITING_APPROVAL",
                "AWAITING_CLARIFICATION",
                "SUSPENDED",
                "INTERRUPTED",
            }:
                recovered = LangGraphRunLifecycle.recover(
                    checkpoint_store=store,
                    thread_id=state["thread_id"],
                    checkpoint_ns=state["checkpoint_namespace"],
                )
                if recovered is not None:
                    owner = effective_user(request)
                    await cancel_owned_research_jobs(
                        state,
                        owner=owner,
                    )
                    cancellation_results = (
                        await recovered.cancellation.cancel_and_wait()
                    )
                    recovered.record_child_cancellation(
                        cancellation_results
                    )
                    recovered.finish(
                        "CANCELLED",
                        stage="recovered_stop",
                        reason="Stopped after durable recovery.",
                        completed_steps=recovered.completed_steps,
                    )
                    stopped = True
        return {"stopped": stopped}

    @router.post("/api/chat/resume_step/{session_id}")
    async def chat_resume_step(
        request: Request,
        session_id: str,
    ) -> Dict[str, Any]:
        """Resume one safe durable PlanStep boundary, never replay or a query."""

        _verify_session_owner(request, session_id)
        try:
            body = await request.json()
        except Exception:
            body = {}
        step_id = str(body.get("step_id") or "").strip() or None
        store = _durable_checkpoint_store_or_503()
        state = store.recover_latest(
            session_id=session_id,
            owner=effective_user(request),
        )
        if state is None:
            raise HTTPException(404, "No recoverable run for this session")
        state = store.reconcile_state(
            state,
            executor_alive=agent_runs.is_active(session_id),
        )
        try:
            result = await resume_interrupted_step(
                checkpoint_store=store,
                checkpoint_state=state,
                owner=effective_user(request),
                step_id=step_id,
            )
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        if result.get("outcome") in {
            "REQUIRES_VERIFICATION",
            "REQUIRES_APPROVAL",
            "NOT_RESUMABLE",
        }:
            return JSONResponse(result, status_code=409)
        return result

    # ------------------------------------------------------------------ #
    # GET /api/chat/stream_status — check if a stream is active for a session
    # ------------------------------------------------------------------ #
    @router.get("/api/chat/stream_status/{session_id}")
    async def chat_stream_status(request: Request, session_id: str) -> Dict[str, Any]:
        _verify_session_owner(request, session_id)
        # A detached run can still be going even if _active_streams was popped;
        # report it as active so the client knows to reconnect via /resume.
        # Read once via .get() to avoid a KeyError race between the membership
        # check and the indexed read if a sibling stream's finally pops the
        # entry in between (same pattern _stream_set already uses).
        rec = _active_streams.get(session_id)
        if rec is None:
            if agent_runs.is_active(session_id):
                return {"status": "streaming", "detached": True}
            store = _durable_checkpoint_store_or_503()
            state = store.recover_latest(
                session_id=session_id,
                owner=effective_user(request),
            )
            if state is not None:
                state = store.reconcile_state(
                    state,
                    executor_alive=False,
                )
                recovered = LangGraphRunLifecycle.recover(
                    checkpoint_store=store,
                    thread_id=state["thread_id"],
                    checkpoint_ns=state["checkpoint_namespace"],
                )
                if recovered is not None:
                    await reconcile_owned_research_jobs(recovered)
                    state = (
                        store.recover_latest(
                            run_id=state["run_id"],
                            owner=effective_user(request),
                        )
                        or state
                    )
                return {
                    "status": str(state.get("status") or "unknown").lower(),
                    "detached": False,
                    "durable": True,
                    "run_id": state.get("run_id"),
                    "thread_id": state.get("thread_id"),
                    "checkpoint_ns": state.get("checkpoint_namespace"),
                    "event_cursor": state.get("event_cursor", 0),
                    "plan": state.get("current_plan"),
                    "completed_steps": state.get("completed_steps") or [],
                    "terminal": state.get("terminal"),
                }
            raise HTTPException(404, "No active stream for this session")
        return rec

    # ------------------------------------------------------------------ #
    # POST /api/inject_context
    # ------------------------------------------------------------------ #
    @router.post("/api/inject_context/{session_id}")
    async def inject_context(request: Request, session_id: str, context: str = Form(...)) -> Dict[str, str]:
        _verify_session_owner(request, session_id)
        try:
            sess = session_manager.get_session(session_id)
            msg = untrusted_context_message("injected research context", f"Research Context: {context}")
            sess.add_message(ChatMessage(msg["role"], msg["content"], metadata=msg.get("metadata")))
            session_manager.save_sessions()
            return {"status": "context_injected"}
        except KeyError:
            raise HTTPException(404, "Session not found")

    # ------------------------------------------------------------------ #
    # GET /api/search — search across chat messages
    # ------------------------------------------------------------------ #
    @router.get("/api/search")
    async def search_messages(
        request: Request,
        q: str = Query("", min_length=0),
        limit: int = Query(20, ge=1, le=100),
    ) -> List[Dict[str, Any]]:
        if not q or not q.strip():
            return []

        _user = effective_user(request)
        return [
            result.to_dict()
            for result in search_session_messages(
                q,
                limit=limit,
                owner=_user,
                restrict_owner=_user is not None,
                include_legacy_owner=False,
            )
        ]

    # ------------------------------------------------------------------ #
    # POST /api/rewrite — lightweight rewrite of last AI message (no tools)
    # ------------------------------------------------------------------ #
    @router.post("/api/rewrite")
    async def rewrite_message(request: Request) -> StreamingResponse:
        """Rewrite the last AI message with an instruction (shorter/simpler/etc).

        Unlike the full chat pipeline, this does NOT run the agent loop or tools.
        It just asks the LLM to rewrite the given text.
        """
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "Invalid JSON")

        session_id = body.get("session_id")
        original_text = body.get("original_text", "")
        instruction = body.get("instruction", "")

        if not session_id or not original_text or not instruction:
            raise HTTPException(400, "session_id, original_text, and instruction are required")

        _verify_session_owner(request, session_id)

        try:
            sess = session_manager.get_session(session_id)
        except (KeyError, SessionNotFoundError):
            raise HTTPException(404, "Session not found")

        messages = [
            {"role": "system", "content": (
                "You are rewriting a previous response. Follow the instruction exactly. "
                "Output ONLY the rewritten text — no preamble, no explanation, no meta-commentary. "
                "Preserve any formatting (markdown, code blocks, lists) from the original."
            )},
            {"role": "user", "content": (
                f"Here is the original response:\n\n{original_text}\n\n"
                f"Instruction: {instruction}"
            )},
        ]

        async def stream_rewrite() -> AsyncGenerator[str, None]:
            full_response = ""
            try:
                async for chunk in stream_llm(
                    sess.endpoint_url,
                    sess.model,
                    messages,
                    headers=sess.headers,
                    temperature=0.7,
                    # 0 = let the server decide (no cap). A hardcoded 4096 made
                    # local reasoning models (Qwen3 / R1) burn the whole budget
                    # inside <think> and emit no rewrite — the bubble just hung
                    # on "Rewriting...". Same fix as the chat max_tokens cap.
                    max_tokens=0,
                    tools=None,
                ):
                    if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                        try:
                            data = json.loads(chunk[6:])
                            if "delta" in data:
                                # Forward the chunk (so the client can show a
                                # thinking indicator) but DON'T fold reasoning
                                # tokens into the saved rewrite — only real
                                # content. reasoning_content arrives flagged
                                # with thinking:true.
                                if not data.get("thinking"):
                                    full_response += data["delta"]
                                yield chunk
                        except json.JSONDecodeError:
                            yield chunk
                    elif chunk.startswith("event: "):
                        yield chunk
                    elif chunk == "data: [DONE]\n\n":
                        # Update the last assistant message in session history.
                        # Strip reasoning-model <think> blocks so the persisted
                        # rewrite is just the rewritten text, not its scratchpad.
                        from src.research_utils import strip_thinking
                        full_response = strip_thinking(full_response).strip() or full_response
                        if full_response:
                            for msg in reversed(sess.history):
                                if (isinstance(msg, ChatMessage) and msg.role == 'assistant') or \
                                   (isinstance(msg, dict) and msg.get('role') == 'assistant'):
                                    if isinstance(msg, ChatMessage):
                                        msg.content = full_response
                                    else:
                                        msg['content'] = full_response
                                    break
                            # Update in DB too
                            db = SessionLocal()
                            try:
                                db_msg = (
                                    db.query(DBChatMessage)
                                    .filter(DBChatMessage.session_id == session_id, DBChatMessage.role == 'assistant')
                                    .order_by(DBChatMessage.timestamp.desc())
                                    .first()
                                )
                                if db_msg:
                                    db_msg.content = full_response
                                    db.commit()
                            except Exception as e:
                                logger.warning("Failed to update rewritten message in DB: %s", e)
                                db.rollback()
                            finally:
                                db.close()
                            session_manager.save_sessions()
                        yield chunk
            except Exception as e:
                logger.error("Rewrite stream error: %s", e)
                yield f'event: error\ndata: {json.dumps({"error": str(e), "status": 500})}\n\n'

        return StreamingResponse(stream_rewrite(), media_type="text/event-stream")

    return router
