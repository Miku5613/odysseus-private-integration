from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from src.agent_runtime.intent_complexity import assess_intent_complexity


AdmissionPath = Literal[
    "normal_chat",
    "single_graph",
    "multi_graph",
    "legacy_agent",
]


@dataclass(frozen=True)
class AgentAdmissionDecision:
    """Admission result for a turn where the user explicitly selected Agent.

    This contract does not own or alter upstream Chat-to-Agent auto-escalation.
    """

    path: AdmissionPath
    assessment: Any | None = None

    @property
    def use_normal_chat(self) -> bool:
        return self.path == "normal_chat"

    @property
    def use_single_graph(self) -> bool:
        return self.path == "single_graph"

    @property
    def use_multi_graph(self) -> bool:
        return self.path == "multi_graph"

    @property
    def use_legacy_agent(self) -> bool:
        return self.path == "legacy_agent"


async def assess_explicit_agent_admission(
    query: str,
    *,
    langgraph_enabled: bool,
    llm_candidates: Sequence[Any],
    prompt_type: str | None = None,
) -> AgentAdmissionDecision:
    """Classify explicit Agent work without deterministic demotion."""

    if not langgraph_enabled:
        return AgentAdmissionDecision(
            path="legacy_agent",
        )

    assessment = await assess_intent_complexity(
        query,
        llm_candidates=llm_candidates,
        prompt_type=prompt_type,
    )

    path_by_class: dict[str, AdmissionPath] = {
        "conversation": "normal_chat",
        "single_graph": "single_graph",
        "multi_graph": "multi_graph",
    }

    try:
        path = path_by_class[
            assessment.execution_class
        ]
    except KeyError as exc:
        raise ValueError(
            "Unsupported Intent/Complexity execution_class: "
            f"{assessment.execution_class!r}"
        ) from exc

    return AgentAdmissionDecision(
        path=path,
        assessment=assessment,
    )


__all__ = [
    "AdmissionPath",
    "AgentAdmissionDecision",
    "assess_explicit_agent_admission",
]
