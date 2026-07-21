from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Protocol

from src.agent_runtime.capability_registry import build_capability_registry
from src.agent_runtime.supervisor_contracts import CapabilityDescriptor
from src.agent_runtime.supervisor_decision import (
    SupervisorDecision,
    parse_supervisor_decision,
)


class SupervisorModel(Protocol):
    """Minimal model interface required by the offline Supervisor reasoner."""

    def invoke(self, prompt: str) -> Any:
        ...


def build_supervisor_prompt(
    raw_query: str,
    *,
    extra_capabilities: Iterable[CapabilityDescriptor] = (),
) -> str:
    """Build a semantic capability-selection prompt.

    The prompt intentionally describes abilities rather than keyword rules.
    It permits multiple capabilities and allows the model to report
    unresolved ambiguity without forcing clarification for information
    that safe discovery capabilities may reasonably resolve.
    """

    registry = build_capability_registry(extra_capabilities)

    capability_context = [
        {
            "name": capability.name,
            "description": capability.description,
            "kind": capability.kind,
            "provider": capability.provider,
            "discoverable_inputs": list(capability.discoverable_inputs),
            "supported_actions": list(capability.supported_actions),
        }
        for capability in registry.values()
    ]

    schema = {
        "selected_capabilities": ["capability_name"],
        "missing_information": ["description of unresolved information"],
        "clarification_needed": False,
        "reasoning_summary": "Brief orchestration rationale.",
    }

    instructions = [
        "You are the general agent Supervisor for Odysseus.",
        "",
        "Your job is orchestration, not domain execution.",
        "",
        "Evaluate the user request against the semantic capability registry.",
        "",
        "Rules:",
        "1. Select zero, one, or multiple capabilities as needed.",
        "2. Do not force every request into exactly one worker.",
        "3. Do not use keyword matching as your reasoning method.",
        "4. Prefer capabilities whose descriptions and supported actions match the actual objective.",
        "5. Missing explicit identifiers do not automatically require clarification.",
        "6. Consider whether safe discovery/search capabilities can reasonably resolve missing details.",
        "7. Set clarification_needed=true only when unresolved ambiguity materially prevents safe or correct progress.",
        "8. Do not invent capability names that are not present in the registry.",
        "9. Do not execute tools or answer the user request itself.",
        "10. Return only one JSON object matching the required schema.",
        "",
        "AVAILABLE_CAPABILITIES:",
        json.dumps(capability_context, ensure_ascii=False, indent=2),
        "",
        "USER_QUERY:",
        raw_query,
        "",
        "REQUIRED_OUTPUT_SCHEMA:",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ]

    return "\n".join(instructions)


def _extract_model_text(response: Any) -> str:
    """Normalize common model response shapes into plain text."""

    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(response, dict):
        value = response.get("content")
        if isinstance(value, str):
            return value

    raise ValueError("Supervisor model response does not contain textual content")


def reason_about_capabilities(
    model: SupervisorModel,
    raw_query: str,
    *,
    extra_capabilities: Iterable[CapabilityDescriptor] = (),
) -> SupervisorDecision:
    """Run one Supervisor reasoning turn and validate its decision.

    This function performs no tool execution and no live routing.
    """

    extras = tuple(extra_capabilities)
    registry = build_capability_registry(extras)

    prompt = build_supervisor_prompt(
        raw_query,
        extra_capabilities=extras,
    )

    response = model.invoke(prompt)
    text = _extract_model_text(response).strip()

    return parse_supervisor_decision(
        text,
        known_capabilities=set(registry),
    )


__all__ = [
    "SupervisorModel",
    "build_supervisor_prompt",
    "reason_about_capabilities",
]
