
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from .general_graph import plan_task


def general_runtime_enabled() -> bool:
    return os.getenv("ODYSSEUS_ENABLE_GENERAL_RUNTIME", "").strip().lower() in {
        "1", "true", "yes", "on", "general"
    }


def general_runtime_supports(message: str) -> bool:
    task = plan_task(message or "")
    return bool(task.is_readonly and not task.should_fallback)


async def stream_general_runtime_legacy_sse(message: str, workspace: str | None = None):
    if not workspace:
        raise ValueError(
            "General runtime requires an explicit workspace."
        )

    root = Path(workspace).resolve()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.agent_runtime.general_runner",
        "--legacy-wrapper",
        "--workspace",
        str(root),
        message,
        cwd=str(root),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    emitted = 0
    marker = "[legacy-wrapper-frame] data:"

    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if marker not in line:
            continue

        payload = line.split(marker, 1)[1].strip()
        if payload.endswith("\\n\\n"):
            payload = payload[:-4].strip()

        if payload == "[DONE]":
            yield "data: [DONE]\n\n"
            emitted += 1
            continue

        try:
            json.loads(payload)
        except Exception:
            yield "data: " + json.dumps({
                "type": "error",
                "message": "General runtime emitted malformed JSON frame.",
                "raw": payload[:500],
            }) + "\n\n"
            emitted += 1
            continue

        yield "data: " + payload + "\n\n"
        emitted += 1

    stderr = ""
    if proc.stderr is not None:
        err = await proc.stderr.read()
        stderr = err.decode("utf-8", errors="replace").strip()

    code = await proc.wait()
    if code != 0:
        yield "data: " + json.dumps({
            "type": "error",
            "message": f"General runtime exited with code {code}",
            "stderr": stderr[-2000:],
        }) + "\n\n"
        yield "data: [DONE]\n\n"
    elif emitted == 0:
        yield "data: " + json.dumps({
            "type": "error",
            "message": "General runtime produced no frames.",
            "stderr": stderr[-2000:],
        }) + "\n\n"
        yield "data: [DONE]\n\n"
