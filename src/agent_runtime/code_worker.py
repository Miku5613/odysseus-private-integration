from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import inspect
import re
from typing import Any

from src.agent_runtime.claim_support import (
    evaluate_code_location_support,
    has_strong_claim_support,
)
from src.agent_runtime.contract_verifier import (
    contract_support_report,
    format_contract_report,
)
from src.agent_runtime.evidence_ledger import (
    format_observations,
    normalize_path,
    observations_from_tool_result,
)
from src.agent_runtime.finalizer import (
    build_grounded_worker_finalizer_prompt,
    parse_grounded_worker_finalizer_response,
)
from src.agent_runtime.general_graph import run_code_inspection_recovery
from src.agent_runtime.supervisor_contracts import WorkerHandoff, WorkerResult
from src.agent_runtime.task_contract import (
    build_task_contract,
    format_task_contract,
)
from src.llm_core import llm_call_with_fallback


CODE_WORKER_CAPABILITY = "code_inspection"

_SEMANTIC_COMPLETION_LEVELS = frozenset({
    "strong",
    "negative_complete",
})


def _semantic_support_complete(level: str | None) -> bool:
    return str(level or "").strip().lower() in _SEMANTIC_COMPLETION_LEVELS


def _normalize_workspace(handoff: WorkerHandoff) -> str | None:
    workspace = handoff.context.get("workspace")

    if workspace is None:
        return None

    workspace_text = str(workspace).strip()
    return workspace_text or None


def _collect_worker_result(
    handoff: WorkerHandoff,
    events: Iterable[dict[str, Any]],
) -> WorkerResult:
    """Normalize specialist-graph output without discarding semantic support.

    The specialist graph already owns claim-support and task-contract checking.
    Worker completion must consume that semantic verdict when it is available;
    the mere presence of an evidence event is not enough to prove that the
    PlanStep objective was fulfilled.
    """

    final_answer = ""
    evidence: list[dict[str, Any]] = []
    error: str | None = None
    event_count = 0
    claim_support_level: str | None = None
    semantic_support_seen = False
    strong_evidence = False
    runtime_events: list[dict[str, Any]] = []

    for event in events:
        event_count += 1
        event_type = str(event.get("type") or "")
        if event_type in {
            "stream_start",
            "tool_call_start",
            "tool_call_result",
            "tool_confirmation_required",
            "tool_risk_blocked",
            "stream_error",
            "stream_done",
        }:
            runtime_events.append(dict(event))

        if event_type == "evidence_hit":
            evidence.append(dict(event))

        if event_type == "finalizer_start":
            raw_support = event.get("claim_support")
            if raw_support is not None:
                claim_support_level = (
                    str(raw_support).strip().lower() or "unknown"
                )
                semantic_support_seen = True
            strong_evidence = bool(
                event.get("strong_evidence", strong_evidence)
            )

        if event_type == "final_answer":
            content = event.get("content")
            if isinstance(content, str):
                final_answer = content

        if event_type == "stream_done":
            content = event.get("final_answer")
            if isinstance(content, str) and content.strip():
                final_answer = content

        if event_type == "stream_error":
            message = event.get("error")
            error = str(message or "Code Worker runtime failed.")

    semantic_complete = _semantic_support_complete(
        claim_support_level
    )
    source_tool_execution_seen = any(
        event.get("type") == "tool_call_result"
        and bool(event.get("ok", True))
        for event in runtime_events
    )
    base_data = {
        "objective": handoff.objective,
        "events_processed": event_count,
        "supports_active_doc": True,
        "claim_support": claim_support_level,
        "semantic_support_seen": semantic_support_seen,
        "strong_evidence": strong_evidence,
        "evidence_complete": (
            semantic_complete
            if semantic_support_seen
            else bool(evidence)
        ),
        "runtime_events": runtime_events,
        "source_tool_execution_seen": source_tool_execution_seen,
    }

    if error is not None:
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="failed",
            summary="Code inspection worker failed.",
            data=base_data,
            evidence=tuple(evidence),
            error=error,
        )

    if not final_answer.strip():
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="partial",
            summary="Code inspection completed without a final answer.",
            data=base_data,
            evidence=tuple(evidence),
        )

    if semantic_support_seen:
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status=(
                "completed"
                if semantic_complete
                else "partial"
            ),
            summary=final_answer,
            data=base_data,
            evidence=tuple(evidence),
        )

    # Compatibility seam for injected/offline stream factories that predate
    # finalizer_start semantic metadata. Production run_stream emits the
    # semantic verdict, so real worker completion does not use this branch.
    if not evidence:
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="partial",
            summary=final_answer,
            data=base_data,
        )

    return WorkerResult(
        capability=CODE_WORKER_CAPABILITY,
        status="completed",
        summary=final_answer,
        data=base_data,
        evidence=tuple(evidence),
    )


_GENERAL_EVIDENCE_TOOLS = frozenset({
    "grep",
    "read_file",
    "read_file_around",
})


def _handoff_semantic_dimensions(
    handoff: WorkerHandoff,
) -> tuple[str, ...]:
    """Preserve explicit responsibility dimensions beyond planner concepts."""

    dimensions = list(
        handoff.context.get("retrieval_concepts", ()) or ()
    )
    if not dimensions:
        return ()
    objective = handoff.objective.lower()
    aliases = (
        (("planning", "planner"), "plan"),
        (("coordination", "coordinator", "orchestration"), "coordinat"),
        (("replanning", "replan"), "replan"),
    )
    for phrases, dimension in aliases:
        already_covered = any(
            dimension in str(existing).lower()
            for existing in dimensions
        )
        if (
            not already_covered
            and any(phrase in objective for phrase in phrases)
        ):
            dimensions.append(dimension)
    return tuple(dict.fromkeys(
        str(value).strip()
        for value in dimensions
        if str(value).strip()
    ))


def _finalize_recovered_step(
    handoff: WorkerHandoff,
    *,
    claim_support: dict[str, Any],
    ledger_observations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    fallback_summary: str,
    finalizer_candidates: Sequence[tuple] | None,
    finalizer_prompt_type: str | None,
    finalizer_caller: Callable[..., str],
) -> WorkerResult:
    """Turn strong recovery evidence into a verified PlanStep result.

    Recovery evidence may prove that relevant source exists without itself
    fulfilling the worker objective. A model-backed grounded finalizer therefore
    produces the step answer and explicitly verifies semantic fulfillment.
    """

    task_contract = build_task_contract(
        handoff.objective,
        semantic_dimensions=_handoff_semantic_dimensions(handoff),
    )
    contract_report = contract_support_report(
        contract=task_contract,
        observations=ledger_observations,
        claim_support=claim_support,
    )
    contract_ready = bool(contract_report.get("allow_stop"))
    evidence_ready = bool(
        has_strong_claim_support(claim_support)
        and contract_ready
    )
    dimension_statuses = [
        item
        for item in contract_report.get("obligations", ())
        if item.get("kind") == "evidence_dimension"
    ]
    missing_dimensions = [
        str(item.get("term") or "").strip()
        for item in dimension_statuses
        if item.get("status") != "satisfied"
        and str(item.get("term") or "").strip()
    ]

    base_data = {
        "objective": handoff.objective,
        "supports_active_doc": True,
        "evidence_complete": evidence_ready,
        "evidence_backend": (
            "deterministic_code_inspection_recovery"
        ),
        "claim_support": claim_support.get("support_level"),
        "claim_support_reason": claim_support.get("reason"),
        "contract_support": contract_report.get("support_level"),
        "contract_support_reason": contract_report.get("reason"),
        "evidence_dimension_coverage": dimension_statuses,
        "missing_evidence_dimensions": missing_dimensions,
        "grounded_finalizer_used": False,
        "task_fulfillment_complete": False,
    }

    if not evidence_ready:
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="partial",
            summary=fallback_summary,
            data=base_data,
            evidence=tuple(evidence),
            missing_information=tuple(
                f"Missing evidence dimension: {dimension}"
                for dimension in missing_dimensions
            ),
        )

    candidates = tuple(finalizer_candidates or ())
    if not candidates:
        data = dict(base_data)
        data["grounded_finalizer_unavailable"] = True
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="partial",
            summary=fallback_summary,
            data=data,
            evidence=tuple(evidence),
            missing_information=(
                "Grounded PlanStep finalization is unavailable for "
                "recovered source evidence.",
            ),
        )

    prioritized_evidence = sorted(
        evidence,
        key=lambda item: {
            "read_file_around": 0,
            "read_file": 1,
            "grep": 2,
        }.get(str(item.get("tool") or ""), 3),
    )
    raw_tool_evidence = "\n\n---\n\n".join(
        (
            f"{item.get('tool')}: "
            f"{item.get('args')}\n"
            f"{item.get('output')}"
        )
        for item in prioritized_evidence
    )[:24000]

    prompt = build_grounded_worker_finalizer_prompt(
        objective=handoff.objective,
        task_contract=format_task_contract(task_contract),
        claim_support=(
            "Claim support:\n"
            f"- support_level: {claim_support.get('support_level')}\n"
            f"- reason: {claim_support.get('reason', '')}"
        ),
        contract_report=format_contract_report(contract_report),
        observations_summary=format_observations(
            ledger_observations,
            limit=10,
        ),
        raw_tool_evidence=raw_tool_evidence,
    )

    try:
        response = finalizer_caller(
            candidates,
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=1600,
            prompt_type=finalizer_prompt_type,
            structured_output=True,
            reasoning_effort="none",
            response_validator=(
                parse_grounded_worker_finalizer_response
            ),
        )
        payload = parse_grounded_worker_finalizer_response(
            response
        )
    except Exception as exc:
        data = dict(base_data)
        data["grounded_finalizer_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return WorkerResult(
            capability=CODE_WORKER_CAPABILITY,
            status="partial",
            summary=fallback_summary,
            data=data,
            evidence=tuple(evidence),
            missing_information=(
                "Recovered evidence could not be converted into a "
                "verified step-specific answer.",
            ),
        )

    # Ground finalizer citations against the evidence ledger's canonical
    # path representation. The ledger normalizes Windows source paths with
    # backslashes, while model output commonly uses forward slashes.
    observed_paths = {
        normalize_path(
            str(observation.get("source_path") or "").strip()
        )
        for observation in ledger_observations
        if normalize_path(
            str(observation.get("source_path") or "").strip()
        )
    }
    reported_path_pairs = [
        (
            str(path).strip(),
            normalize_path(str(path).strip()),
        )
        for path in payload["evidence_paths"]
        if str(path).strip()
    ]
    ungrounded_paths = sorted(
        original
        for original, canonical in reported_path_pairs
        if not canonical or canonical not in observed_paths
    )

    missing_requirements = list(
        payload["missing_requirements"]
    )
    if ungrounded_paths:
        missing_requirements.append(
            "Finalizer cited source paths not present in the "
            "runtime evidence: "
            + ", ".join(ungrounded_paths)
        )

    fulfills = bool(
        payload["fulfills_objective"]
        and not ungrounded_paths
    )
    data = dict(base_data)
    data.update({
        "grounded_finalizer_used": True,
        "task_fulfillment_complete": fulfills,
        "grounded_evidence_paths": list(
            payload["evidence_paths"]
        ),
        "missing_requirements": missing_requirements,
    })

    return WorkerResult(
        capability=CODE_WORKER_CAPABILITY,
        status="completed" if fulfills else "partial",
        summary=payload["answer"],
        data=data,
        evidence=tuple(evidence),
        missing_information=(
            tuple(missing_requirements)
            if not fulfills
            else ()
        ),
    )


def _recovery_worker_result(
    handoff: WorkerHandoff,
    state: Any,
    *,
    finalizer_candidates: Sequence[tuple] | None = None,
    finalizer_prompt_type: str | None = None,
    finalizer_caller: Callable[..., str] = llm_call_with_fallback,
) -> WorkerResult | None:
    """Normalize capability-locked recovery through the existing semantic checker.

    File inventory alone is discovery, not evidence that a code-inspection
    objective was fulfilled. Recovery may report completion only when the
    shared claim-support evaluator finds strong source support.
    """

    task = getattr(state, "task", None)
    if task is None or bool(getattr(task, "should_fallback", True)):
        return None

    final_answer = str(
        getattr(state, "final_answer", "") or ""
    ).strip()
    if not final_answer:
        return None

    evidence: list[dict[str, Any]] = []
    ledger_observations: list[dict[str, Any]] = []

    for index, observation in enumerate(
        tuple(getattr(state, "observations", ()) or ()),
        start=1,
    ):
        tool = str(
            getattr(observation, "tool", "") or ""
        ).strip()
        output = str(
            getattr(observation, "output", "") or ""
        )
        success = bool(
            getattr(observation, "success", False)
        )
        args = dict(
            getattr(observation, "args", {}) or {}
        )

        if (
            not success
            or tool not in _GENERAL_EVIDENCE_TOOLS
            or not output.strip()
        ):
            continue

        evidence.append({
            "kind": "general_runtime_observation",
            "tool": tool,
            "args": args,
            "output": output[:8000],
            "success": True,
        })

        ledger_observations.extend(
            observations_from_tool_result(
                tool_name=tool,
                args=args,
                content=output,
                span_id=f"general_recovery_{index}",
                query=handoff.objective,
                start_index=len(ledger_observations),
            )
        )

    if not evidence:
        return None

    claim_support = evaluate_code_location_support(
        query=handoff.objective,
        observations=ledger_observations,
    )
    return _finalize_recovered_step(
        handoff,
        claim_support=claim_support,
        ledger_observations=ledger_observations,
        evidence=evidence,
        fallback_summary=final_answer,
        finalizer_candidates=finalizer_candidates,
        finalizer_prompt_type=finalizer_prompt_type,
        finalizer_caller=finalizer_caller,
    )


def run_code_worker(
    handoff: WorkerHandoff,
    *,
    session_id: str | None = None,
    stream_factory: Callable[..., Iterable[dict[str, Any]]] | None = None,
    recovery_factory: Callable[..., Any] = run_code_inspection_recovery,
    finalizer_candidates: Sequence[tuple] | None = None,
    finalizer_prompt_type: str | None = None,
    finalizer_caller: Callable[..., str] = llm_call_with_fallback,
) -> WorkerResult:
    "Execute Code Worker while preventing strong primary support from bypassing PlanStep fulfillment verification."

    if stream_factory is None:
        from src.agent_runtime.streaming import run_stream
        stream_factory = run_stream

    if handoff.capability != CODE_WORKER_CAPABILITY:
        raise ValueError(
            "Code Worker requires capability=code_inspection"
        )

    objective = str(handoff.objective or "").strip()

    if not objective:
        raise ValueError("Code Worker requires a non-empty objective")

    workspace = _normalize_workspace(handoff)

    stream_kwargs = {
        "workspace": workspace,
        "session_id": session_id,
        "owner": handoff.context.get("owner"),
        "semantic_dimensions": tuple(
            handoff.context.get("retrieval_concepts", ()) or ()
        ),
    }
    try:
        signature = inspect.signature(stream_factory)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            stream_kwargs = {
                key: value
                for key, value in stream_kwargs.items()
                if key in signature.parameters
            }
    except (TypeError, ValueError):
        # Some injected callables do not expose a signature. Preserve the older
        # workspace/session contract in that compatibility case.
        stream_kwargs.pop("owner", None)

    events = stream_factory(objective, **stream_kwargs)

    primary = _collect_worker_result(
        handoff,
        events,
    )

    if primary.status == "failed" and not workspace:
        return primary

    grounded_primary_required = bool(
        primary.status == "completed"
        and workspace
        and finalizer_candidates
        and not primary.evidence
        and not bool(
            primary.data.get("source_tool_execution_seen")
        )
    )

    if primary.status == "completed" and not grounded_primary_required:
        return primary

    if workspace:
        try:
            recovery_kwargs = {}
            try:
                recovery_signature = inspect.signature(
                    recovery_factory
                )
                if (
                    "semantic_terms"
                    in recovery_signature.parameters
                    or any(
                        parameter.kind
                        == inspect.Parameter.VAR_KEYWORD
                        for parameter
                        in recovery_signature.parameters.values()
                    )
                ):
                    concepts = tuple(
                        handoff.context.get(
                            "retrieval_concepts",
                            (),
                        )
                        or ()
                    )
                    objective_terms = {
                        term.lower()
                        for term in re.findall(
                            r"[A-Za-z][A-Za-z0-9]{3,}",
                            objective,
                        )
                    }
                    ranked_concepts = sorted(
                        enumerate(concepts),
                        key=lambda item: (
                            -len(
                                objective_terms
                                & {
                                    term.lower()
                                    for term in re.findall(
                                        r"[A-Za-z][A-Za-z0-9]{3,}",
                                        str(item[1]),
                                    )
                                }
                            ),
                            item[0],
                        ),
                    )
                    relevant_concepts = [
                        concept
                        for _, concept in ranked_concepts
                        if objective_terms
                        & {
                            term.lower()
                            for term in re.findall(
                                r"[A-Za-z][A-Za-z0-9]{3,}",
                                str(concept),
                            )
                        }
                    ][:4]
                    semantic_term_counts: dict[str, int] = {}
                    for concept in (relevant_concepts or concepts):
                        for term in re.findall(
                            r"[A-Za-z][A-Za-z0-9]{3,}",
                            str(concept),
                        ):
                            normalized = term.lower()
                            if (
                                normalized.endswith("s")
                                and len(normalized) > 4
                            ):
                                normalized = normalized[:-1]
                            semantic_term_counts[normalized] = (
                                semantic_term_counts.get(normalized, 0) + 1
                            )
                    repeated_terms = [
                        term
                        for term, count
                        in semantic_term_counts.items()
                        if count > 1
                    ]
                    ordered_terms = [
                        *repeated_terms,
                        *(
                            term
                            for term in semantic_term_counts
                            if term not in repeated_terms
                        ),
                    ]
                    recovery_kwargs["semantic_terms"] = tuple(
                        ordered_terms
                    )
            except (TypeError, ValueError):
                pass

            recovery_state = recovery_factory(
                objective,
                workspace,
                **recovery_kwargs,
            )
            recovered = _recovery_worker_result(
                handoff,
                recovery_state,
                finalizer_candidates=finalizer_candidates,
                finalizer_prompt_type=finalizer_prompt_type,
                finalizer_caller=finalizer_caller,
            )
            if recovered is not None:
                data = dict(recovered.data)
                if grounded_primary_required:
                    data["primary_specialist_completion_reverified"] = True
                return WorkerResult(
                    capability=recovered.capability,
                    status=recovered.status,
                    summary=recovered.summary,
                    data=data,
                    evidence=recovered.evidence,
                    missing_information=recovered.missing_information,
                    error=recovered.error,
                )
        except Exception as exc:
            data = dict(primary.data)
            data["evidence_recovery_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            if grounded_primary_required:
                data["primary_specialist_completion_reverified"] = False
                data["task_fulfillment_complete"] = False
            return WorkerResult(
                capability=primary.capability,
                status="partial",
                summary=primary.summary,
                data=data,
                evidence=primary.evidence,
                missing_information=(
                    primary.missing_information
                    or (
                        "Strong specialist evidence could not be reverified "
                        "against the PlanStep objective.",
                    )
                ),
                error=primary.error,
            )

    if grounded_primary_required:
        data = dict(primary.data)
        data["primary_specialist_completion_reverified"] = False
        data["task_fulfillment_complete"] = False
        return WorkerResult(
            capability=primary.capability,
            status="partial",
            summary=primary.summary,
            data=data,
            evidence=primary.evidence,
            missing_information=(
                "Strong specialist evidence did not receive grounded "
                "PlanStep fulfillment verification.",
            ),
            error=primary.error,
        )

    return primary



__all__ = [
    "CODE_WORKER_CAPABILITY",
    "run_code_worker",
]
