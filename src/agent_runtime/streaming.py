from __future__ import annotations

from src.agent_runtime.clarification_store import consume_pending_clarification

import queue
import threading
from pathlib import Path
from typing import Iterator

from langchain_core.messages import HumanMessage

from src.agent_runtime.events import make_event, reset_event_sink, set_event_sink
from src.agent_runtime.graph import RUNTIME_VERSION, build_graph
from src.agent_runtime.durable_checkpoint import (
    get_default_checkpoint_store,
)
from src.agent_runtime.run_lifecycle import LangGraphRunIdentity
from src.agent_runtime.tool_adapters import bind_workspace
from src.agent_runtime.tool_policy import infer_tool_scope
from src.agent_runtime.task_contract import build_task_contract


_SENTINEL = object()


def run_stream(
    question: str,
    workspace: str | None = None,
    session_id: str | None = None,
    owner: str | None = None,
    semantic_dimensions: tuple[str, ...] = (),
) -> Iterator[dict]:
    """Run the LangGraph runtime and yield event dicts as they happen.

    This is the bridge shape that /api/chat_stream can later consume.
    """

    if not workspace:
        raise ValueError(
            "LangGraph runtime requires an explicit workspace."
        )

    workspace_path = str(
        Path(workspace).resolve()
    )
    clarified_question = consume_pending_clarification(
        session_id=str(session_id or ""),
        answer=question,
        owner=owner,
    )
    if clarified_question:
        question = clarified_question

    scope = infer_tool_scope(question, workspace_path)
    identity = LangGraphRunIdentity.create(
        session_id=session_id,
        owner=owner,
        runtime_mode="single_graph",
    )
    event_queue: queue.Queue = queue.Queue()

    def sink(event: dict) -> None:
        event_queue.put(event)

    def worker() -> None:
        bind_workspace(workspace_path)
        store = get_default_checkpoint_store()
        executor_id = f"executor_thread_{threading.get_ident()}"
        store.register_run_identity(
            run_id=identity.run_id,
            owner=identity.owner,
            session_id=identity.session_id,
            thread_id=identity.thread_id,
            checkpoint_ns=identity.checkpoint_ns,
            status="PENDING",
        )
        store.claim_executor(
            run_id=identity.run_id,
            executor_id=executor_id,
        )

        tokens = set_event_sink(
            sink,
            suppress_print=True,
            suppress_debug=True,
        )

        try:
            sink(
                make_event(
                    "stream_start",
                    runtime=RUNTIME_VERSION,
                    workspace=workspace_path,
                    scope=scope,
                    question=question,
                    run_id=identity.run_id,
                    thread_id=identity.thread_id,
                    checkpoint_ns=identity.checkpoint_ns,
                )
            )

            graph = build_graph(
                scope=scope,
                session_id=session_id,
                run_id=identity.run_id,
                owner=identity.owner,
            )

            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "tool_rounds": 0,
                    "workspace": workspace_path,
                    "scope": scope,
                    "events": [],
                    "evidence_hits": [],
                    "task_contract": build_task_contract(
                        question,
                        semantic_dimensions=semantic_dimensions,
                    ),
                },
                config={
                    "recursion_limit": 30,
                    **store.config(
                        thread_id=identity.thread_id,
                        checkpoint_ns=identity.checkpoint_ns,
                    ),
                },
            )

            final = str(result["messages"][-1].content)

            sink(
                make_event(
                    "stream_done",
                    runtime=RUNTIME_VERSION,
                    reason="completed",
                    final_answer=final,
                )
            )
            store.set_run_status(identity.run_id, "COMPLETED")

        except Exception as e:
            try:
                store.set_run_status(identity.run_id, "FAILED")
            except Exception:
                pass
            sink(
                make_event(
                    "stream_error",
                    runtime=RUNTIME_VERSION,
                    error_type=type(e).__name__,
                    error=str(e),
                )
            )

        finally:
            try:
                store.release_executor(
                    run_id=identity.run_id,
                    executor_id=executor_id,
                )
            except Exception:
                pass
            reset_event_sink(tokens)
            event_queue.put(_SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = event_queue.get()
        if item is _SENTINEL:
            break
        yield item
