from pathlib import Path

import pytest

from src.agent_runtime.clarification_store import (
    _clear_all_clarifications_for_tests,
    consume_pending_clarification,
    get_pending_clarification,
    register_pending_clarification,
)

ROOT = Path(__file__).resolve().parents[1]

graph = (ROOT / 'src/agent_runtime/graph.py').read_text(
    encoding='utf-8',
    errors='replace',
)
streaming = (ROOT / 'src/agent_runtime/streaming.py').read_text(
    encoding='utf-8',
    errors='replace',
)
legacy = (ROOT / 'src/agent_loop.py').read_text(
    encoding='utf-8',
    errors='replace',
)


@pytest.fixture(autouse=True)
def clean_store():
    _clear_all_clarifications_for_tests()
    yield
    _clear_all_clarifications_for_tests()


def test_pending_clarification_is_session_bound():
    register_pending_clarification(
        session_id='session-a',
        original_request='List my notes',
        question='Which source?',
    )

    assert get_pending_clarification('session-a') is not None
    assert get_pending_clarification('session-b') is None


def test_clarification_rehydrates_original_request():
    register_pending_clarification(
        session_id='session-a',
        original_request='List my notes',
        question='Which source?',
    )

    combined = consume_pending_clarification(
        session_id='session-a',
        answer='Obsidian',
    )

    assert combined is not None
    assert 'List my notes' in combined
    assert 'Which source?' in combined
    assert 'Obsidian' in combined
    assert get_pending_clarification('session-a') is None


def test_clarification_consumption_is_one_shot():
    register_pending_clarification(
        session_id='session-a',
        original_request='List my notes',
        question='Which source?',
    )

    first = consume_pending_clarification(
        session_id='session-a',
        answer='Obsidian',
    )
    second = consume_pending_clarification(
        session_id='session-a',
        answer='Anything else',
    )

    assert first is not None
    assert second is None


def test_graph_registers_pending_clarification():
    assert 'register_pending_clarification(' in graph
    assert 'original_request=query' in graph
    assert 'question=question' in graph


def test_streaming_consumes_clarification_before_scope_routing():
    consume_pos = streaming.index(
        'clarified_question = consume_pending_clarification('
    )
    scope_pos = streaming.index(
        'scope = infer_tool_scope(question, workspace_path)'
    )
    assert consume_pos < scope_pos


def test_next_turn_replaces_question_with_combined_context():
    assert 'if clarified_question:' in streaming
    assert 'question = clarified_question' in streaming


def test_legacy_runtime_remains_outside_clarification_store():
    assert 'register_pending_clarification' not in legacy
    assert 'consume_pending_clarification' not in legacy
