from __future__ import annotations

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_rounds: int
    workspace: str
    scope: str
    events: list[dict]
    evidence_hits: list[dict]
    anchors: list[dict]
    expanded_anchors: list[str]
    claim_support: dict
    action_searches_done: list[str]
    task_contract: dict
    terminal_answer: dict
    expanded_observations: list[str]
    observations: list[dict]
    trace_spans: list[dict]
