from __future__ import annotations

import time
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def start_span(
    *,
    span_id: str,
    kind: str,
    name: str,
    input: dict[str, Any] | None = None,
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = now_ms()
    return {
        "span_id": span_id,
        "parent_id": parent_id,
        "kind": kind,
        "name": name,
        "input": input or {},
        "output_preview": "",
        "ok": None,
        "started_at_ms": started,
        "ended_at_ms": None,
        "latency_ms": None,
        "metadata": metadata or {},
    }


def complete_span(
    span: dict[str, Any],
    *,
    output: Any,
    ok: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ended = now_ms()
    started = int(span.get("started_at_ms") or ended)
    text = str(output or "")

    new_span = dict(span)
    new_span.update(
        {
            "output_preview": text.replace("\n", " ")[:500],
            "ok": bool(ok),
            "ended_at_ms": ended,
            "latency_ms": max(0, ended - started),
        }
    )

    md = dict(new_span.get("metadata") or {})
    md.update(metadata or {})
    new_span["metadata"] = md
    return new_span
