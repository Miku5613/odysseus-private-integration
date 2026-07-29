from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agent_runtime.graph import RUNTIME_VERSION, build_graph
from src.agent_runtime.durable_checkpoint import (
    get_default_checkpoint_store,
)
from src.agent_runtime.run_lifecycle import LangGraphRunIdentity
from src.agent_runtime.legacy_sse_adapter import to_legacy_sse_frame
from src.agent_runtime.legacy_runner import stream_langgraph_agent_legacy_sse
from src.agent_runtime.sse_adapter import to_sse_frame, to_sse_payload
from src.agent_runtime.streaming import run_stream
from src.agent_runtime.tool_adapters import bind_workspace
from src.agent_runtime.tool_policy import infer_tool_scope


def run_once(
    question: str,
    workspace: str | None = None,
) -> str:
    if not workspace:
        raise ValueError(
            "run_once requires an explicit workspace."
        )

    workspace_path = str(
        Path(workspace).resolve()
    )

    bind_workspace(workspace_path)

    scope = infer_tool_scope(question, workspace_path)
    identity = LangGraphRunIdentity.create(
        session_id=None,
        runtime_mode="single_graph",
    )

    graph = build_graph(scope=scope)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "tool_rounds": 0,
            "workspace": workspace_path,
            "scope": scope,
            "events": [],
            "evidence_hits": [],
        },
        config={
            "recursion_limit": 30,
            **get_default_checkpoint_store().config(
                thread_id=identity.thread_id,
                checkpoint_ns=identity.checkpoint_ns,
            ),
        },
    )

    final = result["messages"][-1]
    return str(final.content)


def main():
    args = sys.argv[1:]
    stream_mode = False
    sse_mode = False
    legacy_sse_mode = False
    legacy_wrapper_mode = False

    if args and args[0] == "--stream":
        stream_mode = True
        args = args[1:]

    if args and args[0] == "--sse":
        sse_mode = True
        args = args[1:]

    if args and args[0] == "--legacy-sse":
        legacy_sse_mode = True
        args = args[1:]

    if args and args[0] == "--legacy-wrapper":
        legacy_wrapper_mode = True
        args = args[1:]

    question = " ".join(args).strip()
    if not question:
        question = "Find where the main chat streaming route is implemented in this Odysseus project. Use tools as needed. Do not guess."

    workspace = str(
        Path.cwd().resolve()
    )

    print(f"Workspace: {workspace}")
    print(f"Question: {question}")
    print(f"\n--- Running Odysseus LangGraph Runtime {RUNTIME_VERSION} ---\n")

    if stream_mode:
        for event in run_stream(question, workspace=workspace):
            print("[stream] " + json.dumps(event, ensure_ascii=False, default=str))
        return

    if sse_mode:
        for event in run_stream(question, workspace=workspace):
            payload = to_sse_payload(event)
            print("[sse-payload] " + json.dumps(payload, ensure_ascii=False, default=str))
        return

    if legacy_sse_mode:
        for event in run_stream(question, workspace=workspace):
            frame = to_legacy_sse_frame(event)
            if frame is not None:
                printable = frame.replace("\n", "\\n")
                print("[legacy-sse-frame] " + printable)
        return

    if legacy_wrapper_mode:
        messages = [{"role": "user", "content": question}]
        async def _main():
            async for frame in stream_langgraph_agent_legacy_sse(
                endpoint_url="http://127.0.0.1:11434/v1",
                model="batiai/qwen3.5-9b:q6",
                messages=messages,
                workspace=workspace,
            ):
                printable = frame.replace("\n", "\\n")
                print("[legacy-wrapper-frame] " + printable)
        asyncio.run(_main())
        return

    answer = run_once(question, workspace=workspace)

    print("\n--- Final Answer ---\n")
    print(answer)


if __name__ == "__main__":
    main()
