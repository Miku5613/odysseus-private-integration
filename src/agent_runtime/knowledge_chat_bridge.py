
from __future__ import annotations

import logging

import json
import time
import uuid
from typing import Any

from .knowledge_graph_router import (
    KnowledgeGraphDecision,
    build_human_clarification_prompt,
    evaluate_knowledge_request,
)
from .obsidian_read_graph import run_obsidian_read_graph
from .hitl_policy import build_hitl_policy
from .legacy_fallback_contract import build_legacy_fallback_contract
from .knowledge_llm_response_bridge import enforce_knowledge_llm_final_answer, try_knowledge_llm_response
from .evidence_tracker import (

    compact_tracker_summary,
    new_evidence_tracker,
    record_fallback,
    record_final,
    record_human_gate,
    record_observations,
    record_route,
    record_sufficiency,
)

_LOGGER = logging.getLogger(__name__)

def sse_frame(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _decision_output(decision: KnowledgeGraphDecision) -> str:
    return "KNOWLEDGE_DECISION\n" + decision.to_trace_text()


def _tool_event(
    *,
    tool: str,
    command: str,
    output: str,
    exit_code: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "type": "tool_output",
        "tool": tool,
        "command": command,
        "output": str(output or ""),
        "exit_code": int(exit_code),
    }
    if extra:
        ev.update(extra)
    return ev


def _natural_clarification(decision: KnowledgeGraphDecision) -> str:
    # The runtime decides that human input is needed.
    # This text gives the LLM/UI enough context without pretending certainty.
    if decision.intent == "needs_clarification":
        return "\n".join([
            "I need a bit more context before choosing the right graph.",
            "",
            decision.reason,
            "",
            "The unclear part is:",
            *[f"- {item}" for item in decision.ambiguity],
            "",
            "Do you mean your Ideaverse/Obsidian note, an uploaded file, Odysseus code/docs, or something else?",
        ])

    if decision.intent == "knowledge_propose":
        return "\n".join([
            "I understand this as a knowledge-base change request, but direct writes are disabled right now.",
            "",
            "I can only stage or draft a proposed change later; I will not edit Ideaverse files from this read wiring.",
            "",
            f"Runtime reason: {decision.reason}",
        ])

    if decision.intent == "knowledge_research":
        return "\n".join([
            "I understand this as a knowledge-base research request, but the research graph is not wired yet.",
            "",
            "No web search, external lookup, note draft, or file write was executed.",
            "",
            f"Runtime reason: {decision.reason}",
        ])

    if decision.intent == "unsupported_mutation":
        return "\n".join([
            "I’m blocking this knowledge-base operation.",
            "",
            decision.reason,
            "",
            "No note was edited, moved, renamed, deleted, cleaned up, or written.",
        ])

    return "\n".join([
        "I could not confidently route this as a knowledge-base task.",
        "",
        decision.reason,
        "",
        "No knowledge graph was executed.",
    ])


def run_knowledge_chat_turn(message: str, *, llm_config: dict[str, Any] | None = None) -> tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]:
    """Run one knowledge-router chat turn.

    Returns:
    - final assistant text
    - SSE frames to stream
    - metadata.tool_events compatible list
    - metrics dict

    Safety:
    - Only high-confidence knowledge_read executes the Obsidian Read Graph.
    - All propose/research/destructive/ambiguous outcomes produce text only.
    """
    start = time.time()
    decision = evaluate_knowledge_request(message)
    evidence_tracker = new_evidence_tracker(
        user_request=message,
        runtime="knowledge_chat_bridge",
    )
    record_route(
        evidence_tracker,
        intent=decision.intent,
        target_runtime=decision.target_runtime,
        reason=decision.reason,
        confidence=decision.confidence,
        safe_readonly=decision.is_read_only,
        needs_human=decision.needs_human,
    )
    hitl_policy = build_hitl_policy(decision)
    legacy_fallback_contract = build_legacy_fallback_contract(
        decision,
        hitl_policy=hitl_policy,
    )

    frames: list[str] = []
    tool_events: list[dict[str, Any]] = []

    # Frontend lifecycle note:
    # Emit an assistant lifecycle frame before any tool bubbles. If the stream
    # starts with tool_start/tool_output, the UI can render the tool thread but
    # fail to attach the later delta to a visible assistant text bubble until
    # hard refresh. This mirrors the confirmed deterministic runtime shape.
    frames.append(sse_frame({
        "type": "agent_prep",
        "title": "Knowledge runtime",
        "content": "Preparing a read-only Ideaverse/knowledge task.",
    }))
    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Routing knowledge request",
        "content": "Checking intent, confidence, and graph safety before execution.",
    }))

    call_id = "knowledge_decision_" + uuid.uuid4().hex[:10]
    decision_text = _decision_output(decision)

    frames.append(sse_frame({
        "type": "tool_start",
        "id": call_id,
        "call_id": call_id,
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
    }))
    frames.append(sse_frame({
        "type": "tool_output",
        "id": call_id,
        "call_id": call_id,
        "tool": "knowledge_decision",
        "command": "Knowledge graph decision",
        "output": decision_text,
        "exit_code": 0,
        "knowledge_decision": decision.to_dict(),
    }))

    tool_events.append(_tool_event(
        tool="knowledge_decision",
        command="Knowledge graph decision",
        output=decision_text,
        exit_code=0,
        extra={"knowledge_decision": decision.to_dict()},
    ))

    final_answer = ""

    if decision.intent == "knowledge_read" and decision.confidence == "high":
        frames.append(sse_frame({
            "type": "agent_step",
            "title": "Executing Obsidian Read Graph",
            "content": "High-confidence read-only knowledge request.",
        }))

        state = run_obsidian_read_graph(message)

        for idx, obs in enumerate(state.observations, start=1):
            obs_id = f"obsidian_read_{idx}_" + uuid.uuid4().hex[:8]
            output = obs.output_preview

            frames.append(sse_frame({
                "type": "tool_start",
                "id": obs_id,
                "call_id": obs_id,
                "tool": obs.tool,
                "command": obs.tool,
            }))
            frames.append(sse_frame({
                "type": "tool_output",
                "id": obs_id,
                "call_id": obs_id,
                "tool": obs.tool,
                "command": obs.tool,
                "output": output,
                "exit_code": 0 if obs.success else 1,
                "args": obs.args,
            }))

            tool_events.append(_tool_event(
                tool=obs.tool,
                command=obs.tool,
                output=output,
                exit_code=0 if obs.success else 1,
                extra={"args": obs.args},
            ))

        final_answer = state.final_answer

        record_observations(
            evidence_tracker,
            [
                {
                    "kind": obs.tool,
                    "source_tool": obs.tool,
                    "content": obs.output_preview,
                    "output_preview": obs.output_preview,
                    "success": obs.success,
                    "args": obs.args,
                }
                for obs in state.observations
            ],
            label="obsidian_read",
        )
        record_sufficiency(
            evidence_tracker,
            status="sufficient" if state.observations else "unknown",
            reason="High-confidence read-only knowledge graph executed." if state.observations else "Read graph returned no observations.",
        )
        record_human_gate(
            evidence_tracker,
            required=False,
            reason="High-confidence read-only knowledge route.",
        )
        record_fallback(evidence_tracker, used=False)
        record_final(
            evidence_tracker,
            answer_mode="knowledge_read",
            source_basis="obsidian_read_graph",
            final_answer=final_answer,
        )

        evidence_summary = compact_tracker_summary(evidence_tracker)
        hitl_policy["evidence_summary"] = evidence_summary
        legacy_fallback_contract["evidence_summary"] = evidence_summary
        legacy_fallback_contract["hitl_policy"] = hitl_policy

        metrics = {
            **state.metrics,
            "knowledge_runtime": "v0.16-D",
            "knowledge_intent": decision.intent,
            "knowledge_confidence": decision.confidence,
            "needs_human": decision.needs_human,
            "elapsed_s": round(time.time() - start, 3),
            "evidence_tracker": evidence_tracker,
            "evidence_tracker_summary": compact_tracker_summary(evidence_tracker),
            "hitl_policy": hitl_policy,
            "legacy_fallback_contract": legacy_fallback_contract,
        }

    else:
        final_answer = _natural_clarification(decision)
        record_observations(evidence_tracker, [])
        record_sufficiency(
            evidence_tracker,
            status="insufficient" if decision.needs_human else "not_applicable",
            reason=decision.reason,
        )
        record_human_gate(
            evidence_tracker,
            required=decision.needs_human,
            reason=decision.reason if decision.needs_human else "No human clarification required.",
            ambiguity=decision.ambiguity,
        )
        record_fallback(
            evidence_tracker,
            used=decision.target_runtime != "obsidian_read_graph",
            target_runtime=decision.target_runtime,
            reason=decision.reason,
        )
        record_final(
            evidence_tracker,
            answer_mode="clarification" if decision.needs_human else decision.intent,
            source_basis="knowledge_router_decision",
            final_answer=final_answer,
        )
        evidence_summary = compact_tracker_summary(evidence_tracker)
        hitl_policy["evidence_summary"] = evidence_summary
        legacy_fallback_contract["evidence_summary"] = evidence_summary
        legacy_fallback_contract["hitl_policy"] = hitl_policy

        metrics = {
            "runtime": "v0.16-D-knowledge-chat-bridge",
            "model": "deterministic-knowledge-router",
            "knowledge_intent": decision.intent,
            "knowledge_confidence": decision.confidence,
            "target_runtime": decision.target_runtime,
            "needs_human": decision.needs_human,
            "read_only": decision.is_read_only,
            "write_allowed": decision.write_allowed,
            "tool_calls": 1,
            "elapsed_s": round(time.time() - start, 3),
            "evidence_tracker": evidence_tracker,
            "evidence_tracker_summary": compact_tracker_summary(evidence_tracker),
            "hitl_policy": hitl_policy,
            "legacy_fallback_contract": legacy_fallback_contract,
        }

    # Important for the current frontend lifecycle:
    # tool bubbles alone do not guarantee that a visible assistant text holder
    # is active. Emit a finalizing agent_step before the delta, matching the
    # confirmed deterministic runtimes.
    llm_response = try_knowledge_llm_response(
        llm_config=llm_config,
        user_message=message,
        deterministic_answer=final_answer,
        runtime_decision=decision.to_dict(),
        evidence_summary=compact_tracker_summary(evidence_tracker),
        hitl_policy=hitl_policy,
        legacy_fallback_contract=legacy_fallback_contract,
    )
    if llm_config is not None:
        final_answer = enforce_knowledge_llm_final_answer(
            final_answer=final_answer,
            llm_response=llm_response,
            llm_config=llm_config,
        )
    elif llm_response.get("used_llm") and llm_response.get("final_answer"):
        final_answer = str(llm_response.get("final_answer") or final_answer)

    metrics["llm_response_bridge"] = llm_response
    _LOGGER.info(
        "[knowledge-runtime] llm_response_bridge status=%s used_llm=%s model=%s url_source=%s error_type=%s error_message=%s",
        llm_response.get("status"),
        llm_response.get("used_llm"),
        llm_response.get("model"),
        llm_response.get("url_source"),
        llm_response.get("error_type"),
        llm_response.get("error_message"),
    )
    metrics["evidence_tracker_summary"] = compact_tracker_summary(evidence_tracker)

    frames.append(sse_frame({
        "type": "agent_step",
        "title": "Finalizing knowledge response",
        "content": "Rendering the final assistant answer.",
    }))

    # Use the canonical live-chat delta shape expected by chat.js.
    # Saved refresh works through round_texts, but live rendering expects
    # a top-level "delta" field, not {"type": "delta", "content": ...}.
    frames.append(sse_frame({
        "delta": final_answer,
    }))

    frames.append(sse_frame({
        "type": "metrics",
        **metrics,
    }))

    return final_answer, frames, tool_events, metrics


def should_intercept_knowledge_chat(message: str) -> bool:
    decision = evaluate_knowledge_request(message)
    return decision.intent != "not_knowledge"
