from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SubTaskKind = Literal[
    "exact_lookup",
    "file_lookup",
    "route_lookup",
    "stream_generator_lookup",
    "handler_lookup",
    "callsite_lookup",
    "control_flow_lookup",
    "save_callsite_lookup",
    "backend_event_emit_lookup",
    "streaming_event_emit_lookup",
    "analysis",
    "verification",
]

EvidenceType = Literal[
    "file_line",
    "code_context",
    "symbol_match",
    "negative_match",
    "summary",
]


class SubTask(BaseModel):
    id: str = Field(..., description="Deterministic subtask id, e.g. task_001")
    kind: SubTaskKind
    target: str
    source_scope: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=lambda: ["grep", "read_file_around"])
    is_lookup_only: bool = True
    required_evidence: str
    evidence_type: EvidenceType = "code_context"
    dependencies: list[str] = Field(default_factory=list)

    is_completed: bool = False
    is_unresolved: bool = False
    tool_attempts: int = 0
    max_attempts: int = 3

    evidence_payload: Any | None = None
    internal_verification_report: str | None = None


class ContractState(BaseModel):
    raw_query: str
    task_contract: dict[str, SubTask] = Field(default_factory=dict)
    current_subtask_id: str | None = None
    user_facing_answer: str | None = None
    internal_events: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def complete_or_unresolved(self) -> bool:
        if not self.task_contract:
            return False
        return all(t.is_completed or t.is_unresolved for t in self.task_contract.values())

    def incomplete_ready_tasks(self) -> list[SubTask]:
        ready: list[SubTask] = []
        for task in self.task_contract.values():
            if task.is_completed or task.is_unresolved:
                continue
            deps_ok = all(
                dep in self.task_contract
                and (
                    self.task_contract[dep].is_completed
                    or self.task_contract[dep].is_unresolved
                )
                for dep in task.dependencies
            )
            if deps_ok:
                ready.append(task)
        return ready
