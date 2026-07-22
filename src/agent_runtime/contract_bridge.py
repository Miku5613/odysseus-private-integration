from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import AsyncGenerator


_TRUE_VALUES = {"1", "true", "yes", "on", "contract"}


def contract_runtime_enabled() -> bool:
    return (os.getenv("ODYSSEUS_ENABLE_CONTRACT_RUNTIME", "") or "").strip().lower() in _TRUE_VALUES


def looks_like_readonly_contract_task(message: str | None) -> bool:
    """Conservative v0.12 gate for the experimental contract runtime.

    This deliberately accepts only read-only lookup/inspection wording.
    No write/edit/create/delete/patch/apply tasks are allowed through here.
    """

    q = (message or "").strip().lower()
    if not q:
        return False

    blocked = [
        "write ",
        "edit ",
        "create ",
        "delete ",
        "remove ",
        "modify ",
        "patch ",
        "apply ",
        "rename ",
        "move ",
        "install ",
        "run ",
        "execute ",
        "fix ",
        "change ",
        "update ",
    ]
    if any(x in q for x in blocked):
        return False

    lookup_words = [
        "find ",
        "where ",
        "locate ",
        "search ",
        "show me where",
        "inspect ",
        "trace ",
        "identify ",
    ]
    code_words = [
        "frontend",
        "backend",
        "route",
        "routes",
        "chat_routes",
        "agent_loop",
        "handler",
        "event",
        "stream",
        "streaming",
        "function",
        "call",
        "callsite",
        "file",
        "code",
        ".py",
        ".js",
        "/api/",
    ]

    return any(x in q for x in lookup_words) and any(x in q for x in code_words)


async def stream_contract_runtime_legacy_sse(
    question: str,
    *,
    workspace: str | None = None,
) -> AsyncGenerator[str, None]:
    """Run the deterministic contract runner and yield normal SSE strings.

    The contract runner currently prints debug-prefixed legacy-wrapper frames:

        [legacy-wrapper-frame] data: {...}\n\n

    This bridge strips that prefix and yields:

        data: {...}\n\n

    so chat_routes/frontend can consume the same event shape as other streams.
    """

    if not workspace:
        raise ValueError(
            "Contract runtime requires an explicit workspace."
        )

    project_root = Path(workspace).resolve()

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.agent_runtime.contract_runner",
        "--legacy-wrapper",
        "--workspace",
        str(project_root),
        question,
        cwd=str(project_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    assert proc.stdout is not None

    marker = "[legacy-wrapper-frame] data:"
    saw_frame = False

    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if marker not in line:
            continue

        payload = line.split(marker, 1)[1].strip()

        # contract_runner prints a literal trailing "\\n\\n" marker after
        # each JSON frame. Strip only that wrapper suffix. Do NOT globally
        # remove "\\n\\n", because valid JSON string values may contain
        # escaped newlines used in final answers or file context.
        if payload.endswith("\\n\\n"):
            payload = payload[:-4].strip()

        if not payload:
            continue

        saw_frame = True

        if payload == "[DONE]":
            yield "data: [DONE]\n\n"
            continue

        # Validate JSON-ish frames before yielding them. If validation fails,
        # surface an error event instead of crashing the app stream.
        try:
            json.loads(payload)
        except Exception as e:
            err = {"error": f"contract runtime emitted malformed frame: {e}", "status": 500}
            yield f"event: error\ndata: {json.dumps(err)}\n\n"
            continue

        yield f"data: {payload}\n\n"

    code = await proc.wait()
    if code != 0:
        err = {"error": f"contract runtime exited with code {code}", "status": 500}
        yield f"event: error\ndata: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
    elif not saw_frame:
        err = {"error": "contract runtime produced no SSE frames", "status": 500}
        yield f"event: error\ndata: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
