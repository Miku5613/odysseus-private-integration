
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


GeneralTaskKind = Literal[
    "code_lookup",
    "file_lookup",
    "route_lookup",
    "event_lookup",
    "general_readonly",
    "unsupported",
]


@dataclass
class GeneralTask:
    raw_query: str
    kind: GeneralTaskKind
    target: str = ""
    reason: str = ""
    is_readonly: bool = True
    should_fallback: bool = False


@dataclass
class GeneralObservation:
    tool: str
    args: dict[str, Any]
    output: str
    success: bool = True


@dataclass
class GeneralState:
    raw_query: str
    workspace: str
    task: GeneralTask | None = None
    observations: list[GeneralObservation] = field(default_factory=list)
    final_answer: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
