from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.agent_runtime.capability_registry import build_capability_registry


@dataclass(frozen=True)
class SupervisorDecision:
    """Validated orchestration decision produced by the Supervisor model."""

    selected_capabilities: tuple[str, ...]
    missing_information: tuple[str, ...] = ()
    clarification_needed: bool = False
    reasoning_summary: str = ""


def _dedupe_preserve_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return tuple(result)


def validate_supervisor_decision(
    payload: dict[str, Any],
    *,
    known_capabilities: set[str] | None = None,
) -> SupervisorDecision:
    """Validate model output without trusting capability names blindly."""

    registry = build_capability_registry()
    known = known_capabilities or set(registry)

    raw_selected = payload.get("selected_capabilities", [])
    raw_missing = payload.get("missing_information", [])
    raw_clarification = payload.get("clarification_needed", False)
    raw_summary = payload.get("reasoning_summary", "")

    if not isinstance(raw_selected, list):
        raise ValueError("selected_capabilities must be a list")

    if not isinstance(raw_missing, list):
        raise ValueError("missing_information must be a list")

    if not isinstance(raw_clarification, bool):
        raise ValueError("clarification_needed must be a boolean")

    if not isinstance(raw_summary, str):
        raise ValueError("reasoning_summary must be a string")

    selected: list[str] = []

    for value in raw_selected:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("selected capability names must be non-empty strings")

        name = value.strip()

        if name not in known:
            raise ValueError(f"Unknown capability: {name}")

        selected.append(name)

    missing: list[str] = []

    for value in raw_missing:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing_information entries must be non-empty strings")

        missing.append(value.strip())

    return SupervisorDecision(
        selected_capabilities=_dedupe_preserve_order(selected),
        missing_information=_dedupe_preserve_order(missing),
        clarification_needed=raw_clarification,
        reasoning_summary=raw_summary.strip(),
    )


def parse_supervisor_decision(
    text: str,
    *,
    known_capabilities: set[str] | None = None,
) -> SupervisorDecision:
    """Parse strict JSON produced by the Supervisor model."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Supervisor decision must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Supervisor decision must be a JSON object")

    return validate_supervisor_decision(
        payload,
        known_capabilities=known_capabilities,
    )


def build_supervisor_capability_context() -> list[dict[str, Any]]:
    """Return model-facing semantic capability descriptions."""

    registry = build_capability_registry()

    return [
        {
            "name": capability.name,
            "description": capability.description,
            "kind": capability.kind,
            "discoverable_inputs": list(capability.discoverable_inputs),
            "supported_actions": list(capability.supported_actions),
        }
        for capability in registry.values()
    ]


__all__ = [
    "SupervisorDecision",
    "build_supervisor_capability_context",
    "parse_supervisor_decision",
    "validate_supervisor_decision",
]
