from __future__ import annotations

import time
import uuid
from typing import Any

TRACKER_VERSION = "evidence_tracker_v1"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _short(value: Any, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def new_evidence_tracker(
    *,
    user_request: str,
    runtime: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "version": TRACKER_VERSION,
        "run_id": run_id or ("ev_" + uuid.uuid4().hex[:12]),
        "runtime": str(runtime or ""),
        "user_request": str(user_request or ""),
        "created_at_ms": _now_ms(),
        "route": {},
        "evidence": {
            "items": [],
            "source_paths": [],
            "source_count": 0,
            "top_confidence": 0,
            "strength": "none",
        },
        "sufficiency": {
            "status": "unknown",
            "reason": "No sufficiency decision recorded yet.",
        },
        "human_gate": {
            "required": False,
            "reason": "",
            "ambiguity": [],
        },
        "fallback": {
            "used": False,
            "target_runtime": "",
            "reason": "",
        },
        "final": {
            "answer_mode": "",
            "source_basis": "",
            "preview": "",
        },
    }


def record_route(
    tracker: dict[str, Any],
    *,
    intent: str,
    target_runtime: str,
    reason: str,
    confidence: str = "",
    safe_readonly: bool | None = None,
    needs_human: bool = False,
) -> dict[str, Any]:
    tracker["route"] = {
        "intent": str(intent or ""),
        "target_runtime": str(target_runtime or ""),
        "reason": str(reason or ""),
        "confidence": str(confidence or ""),
        "safe_readonly": safe_readonly,
        "needs_human": bool(needs_human),
    }
    return tracker


def _obs_path(obs: dict[str, Any]) -> str:
    args = obs.get("args") or {}
    return str(
        obs.get("source_path")
        or obs.get("path")
        or obs.get("file")
        or args.get("path")
        or ""
    )


def _obs_preview(obs: dict[str, Any]) -> str:
    return _short(
        obs.get("content")
        or obs.get("output_preview")
        or obs.get("summary")
        or obs.get("text")
        or "",
        240,
    )


def record_observations(
    tracker: dict[str, Any],
    observations: list[dict[str, Any]] | None,
    *,
    label: str = "observation",
    limit: int = 8,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    source_paths: list[str] = []
    top_confidence = 0

    for index, obs in enumerate(observations or [], start=1):
        if not isinstance(obs, dict):
            continue

        path = _obs_path(obs)
        confidence = int(obs.get("confidence") or 0)
        top_confidence = max(top_confidence, confidence)

        if path and path not in source_paths:
            source_paths.append(path)

        items.append({
            "id": str(obs.get("observation_id") or f"{label}_{index}"),
            "kind": str(obs.get("kind") or obs.get("tool") or label),
            "source_path": path,
            "source_tool": str(obs.get("source_tool") or obs.get("tool") or ""),
            "source_authority": str(obs.get("source_authority") or ""),
            "line_start": obs.get("line_start"),
            "line_end": obs.get("line_end"),
            "confidence": confidence,
            "preview": _obs_preview(obs),
        })

    if top_confidence >= 30:
        strength = "strong"
    elif top_confidence >= 15 or source_paths:
        strength = "partial"
    elif items:
        strength = "weak"
    else:
        strength = "none"

    tracker["evidence"] = {
        "items": items[:limit],
        "source_paths": source_paths[:limit],
        "source_count": len(source_paths),
        "top_confidence": top_confidence,
        "strength": strength,
    }
    return tracker


def record_sufficiency(
    tracker: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    tracker["sufficiency"] = {
        "status": str(status or "unknown"),
        "reason": str(reason or ""),
    }
    return tracker


def record_human_gate(
    tracker: dict[str, Any],
    *,
    required: bool,
    reason: str = "",
    ambiguity: list[str] | None = None,
) -> dict[str, Any]:
    tracker["human_gate"] = {
        "required": bool(required),
        "reason": str(reason or ""),
        "ambiguity": [str(x) for x in (ambiguity or [])],
    }
    return tracker


def record_fallback(
    tracker: dict[str, Any],
    *,
    used: bool,
    target_runtime: str = "",
    reason: str = "",
) -> dict[str, Any]:
    tracker["fallback"] = {
        "used": bool(used),
        "target_runtime": str(target_runtime or ""),
        "reason": str(reason or ""),
    }
    return tracker


def record_final(
    tracker: dict[str, Any],
    *,
    answer_mode: str,
    source_basis: str,
    final_answer: str,
) -> dict[str, Any]:
    tracker["final"] = {
        "answer_mode": str(answer_mode or ""),
        "source_basis": str(source_basis or ""),
        "preview": _short(final_answer, 700),
    }
    return tracker


def compact_tracker_summary(tracker: dict[str, Any]) -> dict[str, Any]:
    route = tracker.get("route") or {}
    evidence = tracker.get("evidence") or {}
    sufficiency = tracker.get("sufficiency") or {}
    human_gate = tracker.get("human_gate") or {}
    fallback = tracker.get("fallback") or {}
    final = tracker.get("final") or {}

    return {
        "version": tracker.get("version"),
        "runtime": tracker.get("runtime"),
        "intent": route.get("intent"),
        "target_runtime": route.get("target_runtime"),
        "confidence": route.get("confidence"),
        "evidence_strength": evidence.get("strength"),
        "source_count": evidence.get("source_count"),
        "sufficiency": sufficiency.get("status"),
        "needs_human": human_gate.get("required"),
        "fallback_used": fallback.get("used"),
        "answer_mode": final.get("answer_mode"),
    }


__all__ = [
    "TRACKER_VERSION",
    "compact_tracker_summary",
    "new_evidence_tracker",
    "record_fallback",
    "record_final",
    "record_human_gate",
    "record_observations",
    "record_route",
    "record_sufficiency",
]