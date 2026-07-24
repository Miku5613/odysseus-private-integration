from __future__ import annotations

import asyncio

from src.agent_runtime.dependency_reasoning import (
    DEPENDENCY_REASONING_CAPABILITY,
    build_dependency_reasoning_prompt,
    dependency_reasoning_capability,
    run_dependency_reasoning_worker,
)
from src.agent_runtime.supervisor_contracts import WorkerHandoff


class Model:
    def __init__(self, value):
        self.value = value
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        async def result():
            return self.value

        return result()


def test_dependency_reasoning_capability_is_readonly_and_actionless():
    descriptor = dependency_reasoning_capability()

    assert descriptor.name == DEPENDENCY_REASONING_CAPABILITY
    assert descriptor.supported_actions == ()
    assert descriptor.metadata["supports_external_state"] is False
    assert descriptor.metadata["supports_mutation"] is False
    assert descriptor.metadata["execution_priority"] < 0


def test_dependency_reasoning_prompt_is_grounded_in_completed_outputs():
    prompt = build_dependency_reasoning_prompt(
        "Compare the findings.",
        {
            "research": {
                "status": "completed",
                "summary": "Research result.",
            },
            "code": {
                "status": "completed",
                "summary": "Code result.",
            },
        },
    )

    assert "Compare the findings." in prompt
    assert "Research result." in prompt
    assert "Code result." in prompt
    assert "Do not retrieve new external information" in prompt


def test_dependency_reasoning_worker_completes_without_external_action_selection():
    model = Model("The implementation differs from the saved report.")
    handoff = WorkerHandoff(
        capability=DEPENDENCY_REASONING_CAPABILITY,
        objective="Compare the saved report with code findings.",
        context={
            "dependency_outputs": {
                "research": {
                    "status": "completed",
                    "summary": "Research result.",
                },
                "code": {
                    "status": "completed",
                    "summary": "Code result.",
                },
            }
        },
    )

    result = asyncio.run(
        run_dependency_reasoning_worker(
            handoff,
            model=model,
        )
    )

    assert result.status == "completed"
    assert (
        result.summary
        == "The implementation differs from the saved report."
    )
    assert result.data["uses_external_state"] is False
    assert result.data["dependency_ids"] == [
        "research",
        "code",
    ]
    assert len(model.prompts) == 1


def test_dependency_reasoning_worker_requires_completed_dependency_outputs():
    result = asyncio.run(
        run_dependency_reasoning_worker(
            WorkerHandoff(
                capability=DEPENDENCY_REASONING_CAPABILITY,
                objective="Compare the findings.",
            ),
            model=Model("unused"),
        )
    )

    assert result.status == "needs_clarification"
    assert result.missing_information == (
        "completed dependency outputs",
    )
