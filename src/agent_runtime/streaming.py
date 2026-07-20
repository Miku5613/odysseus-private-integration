from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Iterator

from langchain_core.messages import HumanMessage

from src.agent_runtime.events import make_event, reset_event_sink, set_event_sink
from src.agent_runtime.graph import RUNTIME_VERSION, build_graph
from src.agent_runtime.tool_adapters import DEFAULT_WORKSPACE
from src.agent_runtime.tool_policy import infer_tool_scope


_SENTINEL = object()


def run_stream(
    question: str,
    workspace: str | None = None,
    session_id: str | None = None,
) -> Iterator[dict]:
    """Run the LangGraph runtime and yield event dicts as they happen.

    This is the bridge shape that /api/chat_stream can later consume.
    """

    workspace_path = str(Path(workspace).resolve()) if workspace else str(DEFAULT_WORKSPACE)
    scope = infer_tool_scope(question, workspace_path)
    event_queue: queue.Queue = queue.Queue()

    def sink(event: dict) -> None:
        event_queue.put(event)

    def worker() -> None:
        tokens = set_event_sink(sink, suppress_print=True, suppress_debug=True)

        try:
            sink(
                make_event(
                    "stream_start",
                    runtime=RUNTIME_VERSION,
                    workspace=workspace_path,
                    scope=scope,
                    question=question,
                )
            )

            graph = build_graph(scope=scope, session_id=session_id)

            result = graph.invoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "tool_rounds": 0,
                    "workspace": workspace_path,
                    "scope": scope,
                    "events": [],
                    "evidence_hits": [],
                },
                config={"recursion_limit": 30},
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

        except Exception as e:
            sink(
                make_event(
                    "stream_error",
                    runtime=RUNTIME_VERSION,
                    error_type=type(e).__name__,
                    error=str(e),
                )
            )

        finally:
            reset_event_sink(tokens)
            event_queue.put(_SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = event_queue.get()
        if item is _SENTINEL:
            break
        yield item
