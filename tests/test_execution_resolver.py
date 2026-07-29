from __future__ import annotations

import json

import pytest

from src.agent_runtime.execution_resolver import (
    ExecutionSurfaceDescriptor,
    build_execution_resolver_prompt,
    build_execution_surface_catalog,
    parse_execution_compatibility_response,
    resolve_plan_step,
)
from src.agent_runtime.supervisor_contracts import (
    CapabilityDescriptor,
)
from src.agent_runtime.task_plan import PlanStep


class _Model:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.payload


def _step(
    *,
    requires_external_state=True,
    requires_mutation=False,
):
    return PlanStep(
        id="step_1",
        objective="Retrieve and analyze the required information.",
        depends_on=(),
        work_domains=("semantic_domain",),
        requires_external_state=requires_external_state,
        requires_mutation=requires_mutation,
    )


def _surface(
    surface_id,
    kind,
    *,
    capability_name=None,
    supports_external_state=True,
    supports_mutation=False,
    priority=0,
):
    return ExecutionSurfaceDescriptor(
        id=surface_id,
        kind=kind,
        description=f"{surface_id} description",
        capability_name=capability_name,
        supports_external_state=supports_external_state,
        supports_mutation=supports_mutation,
        priority=priority,
    )


def test_catalog_exposes_only_registered_implemented_native_workers():
    registry = {
        "native_ok": CapabilityDescriptor(
            name="native_ok",
            description="Implemented native capability.",
            kind="specialist",
            provider="test",
        ),
        "registered_only": CapabilityDescriptor(
            name="registered_only",
            description="No worker runner exists.",
            kind="specialist",
            provider="test",
        ),
    }

    catalog = build_execution_surface_catalog(
        include_legacy=False,
        capability_registry=registry,
        worker_runners={
            "native_ok": object(),
            "runner_only": object(),
        },
    )

    assert tuple(
        surface.id
        for surface in catalog
    ) == (
        "langgraph:native_ok",
    )

    assert catalog[0].capability_name == "native_ok"


def test_catalog_includes_legacy_as_deliberate_surface():
    catalog = build_execution_surface_catalog(
        include_legacy=True,
        capability_registry={},
        worker_runners={},
    )

    assert len(catalog) == 1
    assert catalog[0].id == "legacy:agent_loop"
    assert catalog[0].kind == "legacy"
    assert catalog[0].supports_external_state is True
    assert catalog[0].supports_mutation is True


def test_duplicate_surface_ids_are_rejected():
    duplicate = _surface(
        "legacy:agent_loop",
        "registered",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate execution surface id",
    ):
        build_execution_surface_catalog(
            capability_registry={},
            worker_runners={},
            include_legacy=True,
            extra_surfaces=(duplicate,),
        )


def test_resolver_prompt_defines_canonical_source_ownership_boundaries():
    step = PlanStep(
        id="step_saved_research",
        objective=(
            "Retrieve the most recent completed saved research report."
        ),
        depends_on=(),
        work_domains=("user_saved_research",),
        requires_external_state=True,
        requires_mutation=False,
    )
    surfaces = (
        _surface(
            "langgraph:deep_research",
            "langgraph",
            capability_name="deep_research",
        ),
        _surface(
            "langgraph:knowledge_search",
            "langgraph",
            capability_name="knowledge_search",
        ),
        _surface(
            "langgraph:code_inspection",
            "langgraph",
            capability_name="code_inspection",
        ),
        _surface(
            "langgraph:dependency_reasoning",
            "langgraph",
            capability_name="dependency_reasoning",
            supports_external_state=False,
        ),
    )

    prompt = build_execution_resolver_prompt(
        step,
        surfaces,
    )

    assert (
        "Saved research reports do not belong to workspace "
        "knowledge search."
        in prompt
    )
    assert (
        "It requires the relevant semantic index and does not own "
        "the persistent deep-research report library."
        in prompt
    )
    assert (
        "It does not require a workspace semantic knowledge index."
        in prompt
    )
    assert (
        "Similar verbs such as retrieve, inspect, or read do not "
        "make two different source systems interchangeable."
        in prompt
    )
    assert (
        "already-completed PlanStep dependency outputs"
        in prompt
    )


def test_resolver_prompt_exposes_dependencies_for_dependency_reasoning_selection():
    step = PlanStep(
        id="compare",
        objective="Compare the completed research and code findings.",
        depends_on=("research", "code"),
        work_domains=("cross_domain_synthesis",),
        requires_external_state=False,
        requires_mutation=False,
    )
    prompt = build_execution_resolver_prompt(
        step,
        (
            _surface(
                "langgraph:dependency_reasoning",
                "langgraph",
                capability_name="dependency_reasoning",
                supports_external_state=False,
            ),
        ),
    )

    assert '"depends_on": ["research", "code"]' in prompt
    assert (
        "dependency_reasoning is the canonical surface"
        in prompt
    )


def test_resolver_prompt_leaves_unknown_surfaces_without_fake_boundaries():
    prompt = build_execution_resolver_prompt(
        _step(),
        (
            _surface(
                "registered:custom",
                "registered",
            ),
        ),
    )

    payload_text = prompt.split(
        "AVAILABLE EXECUTION SURFACES:",
        1,
    )[1].split(
        "REQUIRED JSON SCHEMA:",
        1,
    )[0]
    payload = json.loads(
        payload_text.strip()
    )

    assert payload[0]["id"] == "registered:custom"
    assert payload[0]["semantic_boundary"] == ""


def test_langgraph_surface_requires_capability_name():
    with pytest.raises(
        ValueError,
        match="requires capability_name",
    ):
        resolve_plan_step(
            _step(),
            model=_Model({}),
            surfaces=(
                _surface(
                    "langgraph:broken",
                    "langgraph",
                ),
            ),
        )


def test_parser_rejects_unknown_surface_id():
    with pytest.raises(
        ValueError,
        match="unknown surfaces",
    ):
        parse_execution_compatibility_response(
            {
                "compatible_surface_ids": [
                    "unknown"
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": "",
            },
            known_surface_ids={
                "legacy:agent_loop"
            },
        )


def test_native_langgraph_wins_when_native_and_legacy_are_compatible():
    native = _surface(
        "langgraph:knowledge_search",
        "langgraph",
        capability_name="knowledge_search",
    )
    legacy = _surface(
        "legacy:agent_loop",
        "legacy",
        supports_mutation=True,
    )

    model = _Model(
        json.dumps(
            {
                "compatible_surface_ids": [
                    "legacy:agent_loop",
                    "langgraph:knowledge_search",
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": (
                    "Both can perform the semantic work."
                ),
            }
        )
    )

    resolution = resolve_plan_step(
        _step(),
        model=model,
        surfaces=(legacy, native),
    )

    assert resolution.status == "resolved"
    assert resolution.selected_surface == native
    assert len(model.prompts) == 1


def test_legacy_is_selected_when_it_is_only_compatible_surface():
    legacy = _surface(
        "legacy:agent_loop",
        "legacy",
        supports_mutation=True,
    )
    registered = _surface(
        "registered:external",
        "registered",
    )

    resolution = resolve_plan_step(
        _step(),
        model=_Model(
            {
                "compatible_surface_ids": [
                    "legacy:agent_loop"
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": (
                    "Legacy owns this uncovered step."
                ),
            }
        ),
        surfaces=(registered, legacy),
    )

    assert resolution.status == "resolved"
    assert resolution.selected_surface == legacy


def test_registered_surface_is_used_when_it_is_only_compatible_surface():
    registered = _surface(
        "registered:external",
        "registered",
    )

    resolution = resolve_plan_step(
        _step(),
        model=_Model(
            {
                "compatible_surface_ids": [
                    "registered:external"
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": "",
            }
        ),
        surfaces=(registered,),
    )

    assert resolution.status == "resolved"
    assert resolution.selected_surface == registered


def test_hard_mutation_requirement_filters_nonmutation_native_surface():
    native = _surface(
        "langgraph:readonly",
        "langgraph",
        capability_name="readonly",
        supports_mutation=False,
    )
    legacy = _surface(
        "legacy:agent_loop",
        "legacy",
        supports_mutation=True,
    )

    resolution = resolve_plan_step(
        _step(
            requires_mutation=True,
        ),
        model=_Model(
            {
                "compatible_surface_ids": [
                    "langgraph:readonly",
                    "legacy:agent_loop",
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": "",
            }
        ),
        surfaces=(native, legacy),
    )

    assert resolution.status == "resolved"
    assert resolution.selected_surface == legacy
    assert resolution.rejected_surface_ids == (
        "langgraph:readonly",
    )


def test_hard_external_state_requirement_can_make_step_unsupported():
    local_only = _surface(
        "registered:local_only",
        "registered",
        supports_external_state=False,
    )

    resolution = resolve_plan_step(
        _step(
            requires_external_state=True,
        ),
        model=_Model(
            {
                "compatible_surface_ids": [
                    "registered:local_only"
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": "",
            }
        ),
        surfaces=(local_only,),
    )

    assert resolution.status == "unsupported"
    assert resolution.selected_surface is None
    assert resolution.rejected_surface_ids == (
        "registered:local_only",
    )


def test_clarification_precedes_surface_selection():
    native = _surface(
        "langgraph:knowledge_search",
        "langgraph",
        capability_name="knowledge_search",
    )

    resolution = resolve_plan_step(
        _step(),
        model=_Model(
            {
                "compatible_surface_ids": [
                    "langgraph:knowledge_search"
                ],
                "clarification_needed": True,
                "missing_information": [
                    "Which workspace should be used?"
                ],
                "reasoning_summary": (
                    "The source scope is ambiguous."
                ),
            }
        ),
        surfaces=(native,),
    )

    assert resolution.status == "clarification"
    assert resolution.selected_surface is None
    assert resolution.missing_information == (
        "Which workspace should be used?",
    )


def test_no_compatible_surface_returns_unsupported():
    native = _surface(
        "langgraph:knowledge_search",
        "langgraph",
        capability_name="knowledge_search",
    )

    resolution = resolve_plan_step(
        _step(),
        model=_Model(
            {
                "compatible_surface_ids": [],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": (
                    "No surface can perform this work."
                ),
            }
        ),
        surfaces=(native,),
    )

    assert resolution.status == "unsupported"
    assert resolution.selected_surface is None


def test_kind_priority_is_native_then_legacy_then_registered():
    native = _surface(
        "langgraph:native",
        "langgraph",
        capability_name="native",
        priority=999,
    )
    legacy = _surface(
        "legacy:agent_loop",
        "legacy",
        supports_mutation=True,
        priority=-999,
    )
    registered = _surface(
        "registered:other",
        "registered",
        priority=-999,
    )

    resolution = resolve_plan_step(
        _step(),
        model=_Model(
            {
                "compatible_surface_ids": [
                    "registered:other",
                    "legacy:agent_loop",
                    "langgraph:native",
                ],
                "clarification_needed": False,
                "missing_information": [],
                "reasoning_summary": "",
            }
        ),
        surfaces=(
            registered,
            legacy,
            native,
        ),
    )

    assert resolution.selected_surface == native


def test_resolver_does_not_execute_worker_or_legacy_runtime():
    import src.agent_runtime.execution_resolver as module

    text = open(
        module.__file__,
        "r",
        encoding="utf-8",
    ).read()

    forbidden = (
        "stream_agent_loop",
        "stream_langgraph_agent_legacy_sse",
        "WorkerHandoff(",
        "WorkerResult(",
        "dispatch_supervisor_decision(",
        "dispatch_supervisor_decision_async(",
    )

    for marker in forbidden:
        assert marker not in text


def test_plan_step_remains_backend_independent():
    fields = set(
        PlanStep.__dataclass_fields__
    )

    assert fields == {
        "id",
        "objective",
        "depends_on",
        "work_domains",
        "uses_dependency_state",
        "requires_external_state",
        "requires_mutation",
    }


def test_dependency_only_step_routes_canonically_without_model_call():
    class _MustNotRun:
        def invoke(self, prompt):
            raise AssertionError(
                "canonical dependency routing should not invoke the model"
            )

    step = PlanStep(
        id="goal_3",
        objective="Compare the completed research and code findings.",
        depends_on=("goal_1", "goal_2"),
        work_domains=("research_comparison", "code_analysis"),
        uses_dependency_state=True,
        requires_external_state=False,
        requires_mutation=False,
    )
    surfaces = (
        ExecutionSurfaceDescriptor(
            id="langgraph:dependency_reasoning",
            kind="langgraph",
            description="Reason over completed dependency outputs.",
            capability_name="dependency_reasoning",
            supports_external_state=False,
            supports_mutation=False,
            priority=-100,
        ),
        ExecutionSurfaceDescriptor(
            id="langgraph:code_inspection",
            kind="langgraph",
            description="Inspect repository source.",
            capability_name="code_inspection",
            supports_external_state=True,
            supports_mutation=False,
        ),
    )

    result = resolve_plan_step(
        step,
        model=_MustNotRun(),
        surfaces=surfaces,
    )

    assert result.status == "resolved"
    assert result.selected_surface is not None
    assert result.selected_surface.id == "langgraph:dependency_reasoning"
    assert result.compatible_surface_ids == (
        "langgraph:dependency_reasoning",
    )


def test_dependency_state_does_not_hide_new_external_state_requirement():
    step = PlanStep(
        id="followup",
        objective=(
            "Use the prior findings to inspect additional repository state."
        ),
        depends_on=("goal_1",),
        work_domains=("codebase_inspection",),
        uses_dependency_state=True,
        requires_external_state=True,
        requires_mutation=False,
    )
    surfaces = (
        ExecutionSurfaceDescriptor(
            id="langgraph:dependency_reasoning",
            kind="langgraph",
            description="Reason over completed dependency outputs.",
            capability_name="dependency_reasoning",
            supports_external_state=False,
            supports_mutation=False,
            priority=-100,
        ),
        ExecutionSurfaceDescriptor(
            id="langgraph:code_inspection",
            kind="langgraph",
            description="Inspect repository source.",
            capability_name="code_inspection",
            supports_external_state=True,
            supports_mutation=False,
        ),
    )
    model = _Model(
        {
            "compatible_surface_ids": [
                "langgraph:dependency_reasoning",
                "langgraph:code_inspection",
            ],
            "clarification_needed": False,
            "missing_information": [],
            "reasoning_summary": "New repository state is still required.",
        }
    )

    result = resolve_plan_step(
        step,
        model=model,
        surfaces=surfaces,
    )

    assert result.status == "resolved"
    assert result.selected_surface is not None
    assert result.selected_surface.id == "langgraph:code_inspection"
    assert "langgraph:dependency_reasoning" in result.rejected_surface_ids


def test_resolver_prompt_exposes_dependency_state_semantics():
    step = PlanStep(
        id="goal_3",
        objective="Synthesize the completed dependency outputs.",
        depends_on=("goal_1", "goal_2"),
        uses_dependency_state=True,
        requires_external_state=False,
        requires_mutation=False,
    )
    surfaces = (
        ExecutionSurfaceDescriptor(
            id="langgraph:dependency_reasoning",
            kind="langgraph",
            description="Reason over completed dependency outputs.",
            supports_external_state=False,
            supports_mutation=False,
        ),
    )

    prompt = build_execution_resolver_prompt(
        step,
        surfaces,
    )

    assert '"uses_dependency_state": true' in prompt
    assert "uses_dependency_state is authoritative semantic metadata" in prompt


def test_workspace_source_contract_locks_external_step_to_code_inspection():
    step = PlanStep(
        id="goal_1",
        objective="Inspect the implementation in the active workspace.",
        work_domains=("codebase_inspection",),
        requires_external_state=True,
        requires_mutation=False,
    )
    code = ExecutionSurfaceDescriptor(
        id="langgraph:code_inspection",
        kind="langgraph",
        description="Inspect repository source.",
        capability_name="code_inspection",
    )
    web = ExecutionSurfaceDescriptor(
        id="langgraph:web_search",
        kind="langgraph",
        description="Search the public web.",
        capability_name="web_search",
    )

    model = _Model({})
    result = resolve_plan_step(
        step,
        model=model,
        surfaces=(web, code),
        source_contract={
            "requested_sources": ["authorized_workspace_files"],
            "excluded_sources": ["public_web"],
        },
    )

    assert result.status == "resolved"
    assert result.selected_surface == code
    assert result.compatible_surface_ids == ("langgraph:code_inspection",)
    assert model.prompts == []


def test_workspace_source_contract_fails_closed_when_owner_is_unavailable():
    step = PlanStep(
        id="goal_1",
        objective="Inspect the implementation in the active workspace.",
        requires_external_state=True,
        requires_mutation=False,
    )
    web = ExecutionSurfaceDescriptor(
        id="langgraph:web_search",
        kind="langgraph",
        description="Search the public web.",
        capability_name="web_search",
    )

    model = _Model({})
    result = resolve_plan_step(
        step,
        model=model,
        surfaces=(web,),
        source_contract={
            "requested_sources": ["authorized_workspace_files"],
        },
    )

    assert result.status == "unsupported"
    assert result.selected_surface is None
    assert "langgraph:code_inspection is unavailable" in result.reasoning_summary
    assert model.prompts == []
