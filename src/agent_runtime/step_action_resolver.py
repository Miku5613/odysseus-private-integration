from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Mapping, Protocol

from src.agent_runtime.supervisor_contracts import CapabilityDescriptor
from src.agent_runtime.structured_llm import (
    invoke_structured_model_async,
)


@dataclass(frozen=True)
class CapabilityActionResolution:
    """Validated operation selection for one already-resolved capability."""

    action: str | None
    clarification_needed: bool = False
    missing_information: tuple[str, ...] = ()
    reasoning_summary: str = ""


class CapabilityActionModel(Protocol):
    """Minimal model seam for selecting one supported capability action."""

    def invoke(self, prompt: str) -> Any | Awaitable[Any]:
        ...


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _clean_string_tuple(
    value: Any,
    *,
    field: str,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")

    result: list[str] = []
    seen: set[str] = set()

    for item in value:
        normalized = _required_text(
            item,
            field=f"{field} item",
        )
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)

    return tuple(result)


def _response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        text = response.strip()
    else:
        content = getattr(response, "content", None)
        if isinstance(content, str):
            text = content.strip()
        else:
            value = getattr(response, "text", None)
            text = value.strip() if isinstance(value, str) else str(response or "").strip()

    if not text:
        raise ValueError("Capability action model returned an empty response")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                "Capability action model did not return a JSON object"
            )
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Capability action model returned invalid JSON"
            ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Capability action response must be a JSON object"
        )

    return payload


def build_capability_action_prompt(
    objective: str,
    capability: CapabilityDescriptor,
    *,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Ask only for the operation of an already-selected capability."""

    normalized_objective = _required_text(
        objective,
        field="PlanStep objective",
    )
    supported_actions = tuple(
        str(action).strip()
        for action in capability.supported_actions
        if str(action).strip()
    )

    if not supported_actions:
        raise ValueError(
            f"Capability {capability.name} does not advertise actions"
        )

    schema = {
        "action": supported_actions[0],
        "clarification_needed": False,
        "missing_information": [],
        "reasoning_summary": "brief operation-selection rationale",
    }

    dependency_outputs = dict(
        (context or {}).get("dependency_outputs", {})
        if isinstance(context, Mapping)
        else {}
    )

    return "\n".join([
        "You are the Odysseus capability-action resolver.",
        "",
        "A downstream execution surface and capability have already been selected.",
        "Choose only the supported operation required to execute this PlanStep.",
        "Do not select another capability, worker, tool, provider, or runtime.",
        "Do not execute the action.",
        "Set clarification_needed=true only when the operation cannot be chosen",
        "safely from the objective and completed dependency outputs.",
        "Return only one JSON object matching the required schema.",
        "",
        "PLAN STEP OBJECTIVE:",
        normalized_objective,
        "",
        "SELECTED CAPABILITY:",
        json.dumps(
            {
                "name": capability.name,
                "description": capability.description,
                "supported_actions": list(supported_actions),
            },
            ensure_ascii=False,
            default=str,
        ),
        "",
        "COMPLETED DEPENDENCY OUTPUTS:",
        json.dumps(
            dependency_outputs,
            ensure_ascii=False,
            default=str,
        ),
        "",
        "REQUIRED JSON SCHEMA:",
        json.dumps(schema, ensure_ascii=False, indent=2),
    ])


def parse_capability_action_response(
    response: Any,
    *,
    capability: CapabilityDescriptor,
) -> CapabilityActionResolution:
    """Validate an action against the capability's advertised contract."""

    payload = _response_payload(response)
    allowed = {
        "action",
        "clarification_needed",
        "missing_information",
        "reasoning_summary",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            "Capability action response contains unsupported fields: "
            + ", ".join(unknown)
        )

    clarification_needed = payload.get(
        "clarification_needed",
        False,
    )
    if not isinstance(clarification_needed, bool):
        raise ValueError(
            "clarification_needed must be a boolean"
        )

    missing_information = _clean_string_tuple(
        payload.get("missing_information", []),
        field="missing_information",
    )

    reasoning_summary = payload.get(
        "reasoning_summary",
        "",
    )
    if not isinstance(reasoning_summary, str):
        raise ValueError(
            "reasoning_summary must be a string"
        )

    raw_action = payload.get("action")
    action = (
        raw_action.strip()
        if isinstance(raw_action, str)
        else None
    )
    if action == "":
        action = None

    supported = {
        str(value).strip()
        for value in capability.supported_actions
        if str(value).strip()
    }

    if clarification_needed:
        if action is not None and action not in supported:
            raise ValueError(
                f"Unsupported action for {capability.name}: {action}"
            )
        return CapabilityActionResolution(
            action=action,
            clarification_needed=True,
            missing_information=missing_information,
            reasoning_summary=reasoning_summary.strip(),
        )

    if not supported:
        if action is not None:
            raise ValueError(
                f"Capability {capability.name} does not support actions"
            )
        return CapabilityActionResolution(
            action=None,
            reasoning_summary=reasoning_summary.strip(),
        )

    if action is None:
        raise ValueError(
            f"Capability action is required for {capability.name}"
        )

    if action not in supported:
        raise ValueError(
            f"Unsupported action for {capability.name}: {action}"
        )

    return CapabilityActionResolution(
        action=action,
        missing_information=missing_information,
        reasoning_summary=reasoning_summary.strip(),
    )


async def resolve_capability_action_async(
    objective: str,
    capability: CapabilityDescriptor,
    *,
    model: CapabilityActionModel,
    context: Mapping[str, Any] | None = None,
) -> CapabilityActionResolution:
    """Resolve one supported action without blocking an async runtime."""

    if not capability.supported_actions:
        return CapabilityActionResolution(
            action=None,
            reasoning_summary=(
                "The selected capability does not require an advertised action."
            ),
        )

    prompt = build_capability_action_prompt(
        objective,
        capability,
        context=context,
    )
    return await invoke_structured_model_async(
        model,
        prompt,
        lambda response: parse_capability_action_response(
            response,
            capability=capability,
        ),
    )


__all__ = [
    "CapabilityActionModel",
    "CapabilityActionResolution",
    "build_capability_action_prompt",
    "parse_capability_action_response",
    "resolve_capability_action_async",
]
