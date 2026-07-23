from __future__ import annotations

from pathlib import Path

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


import json
import re
import time
from typing import Any

from src.agent_runtime.contract_models import ContractState, SubTask
from src.agent_runtime.tool_adapters import (
    TOOL_MAP,
    bind_workspace,
    reset_workspace,
)


RUNTIME_VERSION = "v0.11.3-contract"


def _norm(text: str | None) -> str:
    return str(text or "").lower().replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")


def _emit(event_type: str, **payload: Any) -> dict[str, Any]:
    event = {"type": event_type, **payload}
    print(f"[legacy-wrapper-frame] data: {json.dumps(event, ensure_ascii=False)}\\n\\n")
    return event


def _emit_delta(text: str) -> None:
    print(f"[legacy-wrapper-frame] data: {json.dumps({'delta': text}, ensure_ascii=False)}\\n\\n")


def _emit_done() -> None:
    print("[legacy-wrapper-frame] data: [DONE]\\n\\n")


def _extract_exact_terms(query: str) -> list[str]:
    terms: list[str] = []

    for m in re.findall(r'/[A-Za-z0-9_./:-]+', query):
        terms.append(m)

    for m in re.findall(r'\[[A-Za-z0-9_:-]+\]', query):
        terms.append(m)

    for m in re.findall(r'data:\s*\[[A-Za-z0-9_:-]+\]', query, flags=re.I):
        terms.append(m)

    for m in re.findall(r"[\"']([^\"']{3,80})[\"']", query):
        if any(ch in m for ch in "_:/[]-."):
            terms.append(m)

    for m in re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', query):
        if "_" in m and len(m) >= 4 and m.lower() not in {"chat_routes"}:
            terms.append(m)

    seen = set()
    out = []
    for t in terms:
        k = t.lower()
        if k not in seen:
            out.append(t)
            seen.add(k)
    return out


def _source_scope_for_query(query: str) -> list[str]:
    q = _norm(query)

    if any(x in q for x in ["frontend", "front end", "client", "browser", "ui"]):
        return ["static\\js"]

    if "chat routes" in q or "chat_routes" in q or "/api/" in query.lower() or "chat stream" in q or "chat_stream" in q:
        return ["routes\\chat_routes.py"]

    if "agent loop" in q:
        return ["src\\agent_loop.py"]

    return ["."]


def _infer_event_name(query: str) -> str | None:
    q = _norm(query)
    exact_terms = _extract_exact_terms(query)

    # Direct snake_case event names.
    for term in exact_terms:
        if "_" in term and not term.startswith("/"):
            return term

    # Common natural-language aliases for known SSE events.
    aliases = [
        (["document stream", "opens"], "doc_stream_open"),
        (["document stream", "open"], "doc_stream_open"),
        (["doc stream", "opens"], "doc_stream_open"),
        (["doc stream", "open"], "doc_stream_open"),
        (["document stream", "delta"], "doc_stream_delta"),
        (["doc stream", "delta"], "doc_stream_delta"),
        (["tool output"], "tool_output"),
        (["agent step"], "agent_step"),
        (["message saved"], "message_saved"),
        (["tool start"], "tool_start"),
        (["tool progress"], "tool_progress"),
    ]
    for needles, event in aliases:
        if all(n in q for n in needles):
            return event

    return None


def _backend_streaming_scope(query: str) -> list[str]:
    q = _norm(query)
    if "agent loop" in q or "tool output" in q or "tool_output" in q:
        return ["src\\agent_loop.py"]
    if "chat routes" in q or "chat_routes" in q or "message saved" in q or "message_saved" in q:
        return ["routes\\chat_routes.py"]
    return ["routes\\chat_routes.py", "src\\agent_loop.py"]


def deconstruct_contract(query: str) -> ContractState:
    """Hybrid deconstructor, deterministic layer first.

    v0.11.2 expands deterministic coverage for common Odysseus runtime
    questions: frontend event handlers, backend event emissions, streaming
    emissions, route/generator lookup, save callsites, and exact negatives.
    """

    q = _norm(query)
    exact_terms = _extract_exact_terms(query)
    inferred_event = _infer_event_name(query)
    scope = _source_scope_for_query(query)
    tasks: dict[str, SubTask] = {}

    def add(task: SubTask) -> None:
        tasks[task.id] = task

    # Frontend event handler lookup. Supports both exact event names and
    # natural phrases like "opens document stream events".
    if any(x in q for x in ["frontend", "front end", "client", "browser", "ui"]) and any(x in q for x in ["handle", "handles", "handling", "event", "events", "opens", "open"]):
        target = inferred_event or (exact_terms[0] if exact_terms else "event handler")
        add(SubTask(
            id="task_001",
            kind="handler_lookup",
            target=target,
            source_scope=["static\\js"],
            required_evidence=f"Find a frontend handler branch that directly references {target}.",
            evidence_type="code_context",
        ))
        return ContractState(raw_query=query, task_contract=tasks)

    # Backend event emission lookup. Important: "emits message_saved" is not
    # the same as "where is the save callsite"; it needs yield/SSE evidence.
    if "backend" in q and any(x in q for x in ["emit", "emits", "emitted", "yield", "sse"]) and inferred_event:
        kind = "streaming_event_emit_lookup" if ("stream" in q or "streaming" in q or inferred_event in {"tool_output", "tool_start", "agent_step"}) else "backend_event_emit_lookup"
        add(SubTask(
            id="task_001",
            kind=kind,
            target=inferred_event,
            source_scope=_backend_streaming_scope(query),
            required_evidence=f"Find backend streaming/SSE emission code that emits {inferred_event}.",
            evidence_type="code_context",
        ))
        return ContractState(raw_query=query, task_contract=tasks)

    # API route + stream generator compound lookup.
    if "/api/" in query.lower() and ("registered" in q or "route" in q) and ("generator" in q or "stream" in q):
        target = exact_terms[0] if exact_terms else "/api/chat_stream"
        add(SubTask(
            id="task_001",
            kind="route_lookup",
            target=target,
            source_scope=["routes\\chat_routes.py"],
            required_evidence=f"Find the backend route decorator and handler for {target}.",
            evidence_type="code_context",
        ))
        add(SubTask(
            id="task_002",
            kind="stream_generator_lookup",
            target="stream generator",
            source_scope=["routes\\chat_routes.py"],
            required_evidence="Verify the backend handler returns or creates a StreamingResponse and/or yields SSE data.",
            evidence_type="code_context",
            dependencies=["task_001"],
        ))
        return ContractState(raw_query=query, task_contract=tasks)

    # Exact data marker / control-flow lookup.
    if "[done]" in q or "data: [done]" in q:
        target = "data: [DONE]"
        add(SubTask(
            id="task_001",
            kind="control_flow_lookup",
            target=target,
            source_scope=scope,
            required_evidence="Find code that handles, filters, emits, or branches on data: [DONE].",
            evidence_type="code_context",
        ))
        return ContractState(raw_query=query, task_contract=tasks)

    # Agent loop execution/callsite lookup.
    if "agent loop" in q and any(x in q for x in ["execute", "executes", "executing", "tool call", "tool calls"]):
        add(SubTask(
            id="task_001",
            kind="callsite_lookup",
            target="execute_tool_block",
            source_scope=["src\\agent_loop.py"],
            required_evidence="Find the concrete callsite where the agent loop executes tool blocks.",
            evidence_type="code_context",
        ))
        return ContractState(raw_query=query, task_contract=tasks)

    # Backend save-call lookup.
    if any(x in q for x in ["save", "saves", "saved", "saving"]) and "frontend" not in q:
        add(SubTask(
            id="task_001",
            kind="save_callsite_lookup",
            target="save_assistant_response",
            source_scope=scope,
            required_evidence="Find the backend callsite that saves the assistant response.",
            evidence_type="code_context",
        ))
        return ContractState(raw_query=query, task_contract=tasks)

    # Generic exact lookup.
    if exact_terms:
        add(SubTask(
            id="task_001",
            kind="exact_lookup",
            target=exact_terms[0],
            source_scope=scope,
            required_evidence=f"Find an exact source match for {exact_terms[0]}.",
            evidence_type="code_context",
        ))
        return ContractState(raw_query=query, task_contract=tasks)

    # Conservative fallback.
    add(SubTask(
        id="task_001",
        kind="exact_lookup",
        target=query,
        source_scope=scope,
        required_evidence="Find the most directly relevant source location.",
        evidence_type="code_context",
    ))
    return ContractState(raw_query=query, task_contract=tasks)

def _call_tool(name: str, args: dict[str, Any], call_id: str, state: ContractState) -> str:
    _emit("tool_start", tool=name, command=args, call_id=call_id)

    tool = TOOL_MAP.get(name)
    if not tool:
        out = f"ERROR: tool not found: {name}"
        _emit("tool_output", tool=name, output=out, ok=False, call_id=call_id, chars=len(out))
        return out

    workspace_token = None

    try:
        if not state.workspace:
            raise RuntimeError(
                "Contract tool execution requires an explicit workspace."
            )

        workspace_token = bind_workspace(
            state.workspace
        )

        result = str(
            tool.invoke(args)
        )

        ok = not result.startswith("ERROR:")
    except Exception as e:
        result = f"ERROR executing {name}: {e}"
        ok = False
    finally:
        if workspace_token is not None:
            reset_workspace(
                workspace_token
            )

    _emit("tool_output", tool=name, output=result[:900], ok=ok, call_id=call_id, chars=len(result))
    state.internal_events.append({
        "type": "tool_result",
        "tool": name,
        "args": args,
        "ok": ok,
        "output": result,
    })
    return result


def _all_matches(output: str) -> list[tuple[str, int, str]]:
    matches: list[tuple[str, int, str]] = []
    for line in str(output or "").splitlines():
        m = re.match(r'([^:\n]+):(\d+):(.*)$', line.strip())
        if m:
            matches.append((m.group(1), int(m.group(2)), m.group(3).strip()))
    return matches


def _rank_match_for_task(task: SubTask, match: tuple[str, int, str]) -> int:
    path, line, content = match
    low = content.lower()
    target = str(task.target or "").lower()
    score = 0

    # Prefer implementation locations over headers/imports/comments.
    if line <= 25:
        score -= 20
    if low.startswith(chr(34) * 3) or low.startswith(chr(39) * 3):
        score -= 50
    if low.startswith("from ") or low.startswith("import "):
        score -= 45
    if low.startswith("#"):
        score -= 20

    if target and target in low:
        score += 20

    if task.kind == "exact_lookup":
        # Exact lookup should prefer executable use sites over registries,
        # aliases, test vectors, and string-only declarations of the symbol.
        if re.search(r"\b(?:if|elif|for|while|def|async\s+def)\b", low):
            score += 25
        if any(token in low for token in ("yield ", "await ", "return ", "addeventlistener")):
            score += 30
        if re.search(r"[\"'][^\"']*[\"']\s*[,):]", low) and not any(
            token in low for token in ("yield ", "if ", "elif ", "await ")
        ):
            score -= 18
        norm_path = path.lower().replace("/", "\\")
        if "src\\agent_runtime\\contract_runtime.py" in norm_path:
            score -= 20

    if task.kind == "route_lookup":
        if "@router." in low:
            score += 100
        if "@router.post" in low or "@router.get" in low:
            score += 40
        if "async def" in low or "def " in low:
            score += 50

    elif task.kind == "stream_generator_lookup":
        if "async def chat_stream" in low:
            score += 120
        if "streamingresponse" in low and not low.startswith("from "):
            score += 80
        if "yield" in low and "data:" in low:
            score += 60
        if low.startswith("from ") and "streamingresponse" in low:
            score -= 100

    elif task.kind == "handler_lookup":
        if target and target in low:
            score += 50
        if "json.type" in low:
            score += 40
        if "else if" in low or "if (" in low:
            score += 30
        if "addeventlistener" in low:
            score += 15

    elif task.kind == "callsite_lookup":
        if "execute_tool_block" in low:
            score += 80
        if "return await" in low or "await" in low:
            score += 50
        if low.startswith("from ") or "def " in low:
            score -= 40

    elif task.kind == "save_callsite_lookup":
        if "save_assistant_response(" in low:
            score += 90
        if "message_saved" in low:
            score += 40
        if low.startswith("from "):
            score -= 40

    elif task.kind == "control_flow_lookup":
        if "[done]" in low:
            score += 70
        if "chunk.startswith" in low:
            score += 40
        if "yield" in low and "data:" in low:
            score += 30

    elif task.kind == "backend_event_emit_lookup":
        norm_path = path.lower().replace("/", "\\")
        exact_type_single = "'type': '" + target + "'"
        exact_type_double = '"type": "' + target + '"'

        if target and target in low:
            score += 80
        if exact_type_single in low or exact_type_double in low:
            score += 150
        if "yield" in low and "data:" in low:
            score += 60
        if "json.dumps" in low:
            score += 35
        if "save_assistant_response" in low:
            score += 20
        if "routes\\chat_routes.py" in norm_path:
            score += 35

        # Generic relay lines are weaker unless they contain the exact target.
        if "yield f" in low and "json.dumps(data)" in low and target not in low:
            score -= 130

        if "patch_" in norm_path or "install_" in norm_path or "inspect" in norm_path:
            score -= 250

    elif task.kind == "streaming_event_emit_lookup":
        norm_path = path.lower().replace("/", "\\")
        exact_type_single = "'type': '" + target + "'"
        exact_type_double = '"type": "' + target + '"'

        if target and target in low:
            score += 90
        if exact_type_single in low or exact_type_double in low:
            score += 140
        if "emit tool_output" in low or "emit tool output" in low:
            score += 90
        if "tool_output below" in low or "tool_output (" in low:
            score += 70

        # Generic yield lines are useful, but should not beat exact event refs.
        if "yield" in low and "data:" in low:
            score += 35
        if "json.dumps" in low:
            score += 20

        if "src\\agent_loop.py" in norm_path:
            score += 35
        if "routes\\chat_routes.py" in norm_path:
            score += 15
        if "skills_routes" in norm_path:
            score -= 80
        if "patch_" in norm_path or "install_" in norm_path or "inspect" in norm_path:
            score -= 250

        # Avoid generic relay lines when they do not contain the target event.
        if "yield f" in low and "json.dumps(data)" in low and target not in low:
            score -= 120

    # Handler-specific refinement: direct event branches beat broad allow-lists.
    if task.kind == "handler_lookup":
        norm_path = path.lower().replace("/", "\\")
        direct_single = "json.type === '" + target + "'"
        direct_double = 'json.type === "' + target + '"'

        if target and (direct_single in low or direct_double in low):
            score += 140

        # Canonical frontend runtime handlers should beat secondary views.
        if "static\\js\\chat.js" in norm_path:
            score += 65
        if "static\\js\\compare\\" in norm_path:
            score -= 45
        if "static\\js\\group.js" in norm_path:
            score -= 35
        if "static\\js\\skills.js" in norm_path:
            score -= 35

        # Broad allow-list lines are useful as evidence of passthrough, but they
        # should not outrank concrete handler branches.
        if "json.delta ||" in low:
            score -= 55
        if "|| json.type ===" in low and target in low:
            score -= 25

    return score


def _contextual_match_bonus(
    workspace: str | None,
    path: str,
    repository_context: dict | None = None,
) -> float:
    """Return soft provenance weight for an already-valid lexical match.

    Provenance never determines eligibility. It only helps rank multiple
    matches that have already satisfied the Contract grep/search step.
    """
    if not workspace:
        return 0.0

    try:
        from src.agent_runtime.general_graph import (
            _build_repository_context,
            _source_provenance_context,
        )

        if repository_context is None:
            repository_context = _build_repository_context(
                workspace,
            )

        provenance = _source_provenance_context(
            workspace,
            path,
            repository_context,
        )
    except Exception:
        return 0.0

    score = 0.0

    role = provenance.get("role")

    if role == "source":
        score += 50.0
    elif role == "test":
        score -= 5.0
    elif role == "documentation_or_report":
        score -= 10.0
    elif role == "historical_backup":
        score -= 25.0
    elif role == "generated_report":
        score -= 35.0
    elif role == "local_script":
        score -= 40.0

    if provenance.get("tracked") is True:
        score += 30.0

    if provenance.get("modified") is True:
        score += 6.0

    if provenance.get("ignored") is True:
        score -= 4.0

    return score


def _first_match(
    output: str,
    task: SubTask | None = None,
    workspace: str | None = None,
) -> tuple[str, int] | None:
    """DORMANT COMPATIBILITY WRAPPER; production uses the audited ranker.

    Older evals and external diagnostics import this helper directly. The live
    Contract execution path calls ``_rank_matches_with_provenance`` so it can
    retain the complete score/provenance audit.
    """
    ranked = _rank_matches_with_provenance(
        output,
        task,
        workspace,
    )
    if not ranked:
        return None

    return ranked[0]["path"], ranked[0]["line"]


def _rank_matches_with_provenance(
    output: str,
    task: SubTask | None = None,
    workspace: str | None = None,
) -> list[dict[str, Any]]:
    """Rank lexical matches and retain an auditable score breakdown.

    Low-authority locations remain eligible. Their path role and Git context
    influence ranking only after the candidate has satisfied the lexical
    Contract lookup.
    """
    matches = _all_matches(output)
    if not matches:
        return []

    if task is None:
        return [
            {
                "path": path,
                "line": line,
                "content": content,
                "task_score": 0,
                "provenance_score": 0.0,
                "total_score": 0.0,
                "provenance": {},
            }
            for path, line, content in matches
        ]

    repository_context = None
    provenance_resolver = None

    if workspace:
        try:
            from src.agent_runtime.general_graph import (
                _build_repository_context,
                _source_provenance_context,
            )

            repository_context = _build_repository_context(
                workspace,
            )
            provenance_resolver = _source_provenance_context
        except Exception:
            repository_context = None
            provenance_resolver = None

    ranked: list[dict[str, Any]] = []
    for path, line, content in matches:
        task_score = _rank_match_for_task(
            task,
            (path, line, content),
        )
        provenance_score = _contextual_match_bonus(
            workspace,
            path,
            repository_context,
        )
        provenance: dict[str, Any] = {}
        if workspace and provenance_resolver is not None:
            try:
                provenance = provenance_resolver(
                    workspace,
                    path,
                    repository_context,
                )
            except Exception:
                provenance = {}

        ranked.append({
            "path": path,
            "line": line,
            "content": content,
            "task_score": task_score,
            "provenance_score": provenance_score,
            "total_score": task_score + provenance_score,
            "provenance": provenance,
        })

    ranked.sort(
        key=lambda item: (
            -float(item["total_score"]),
            int(item["line"]),
        ),
    )
    return ranked

def _path_for_task(task: SubTask) -> str:
    return task.source_scope[0] if task.source_scope else "."


def _grep_pattern_for_task(task: SubTask) -> str:
    target = task.target

    if task.kind == "route_lookup":
        return re.escape(target)

    if task.kind == "stream_generator_lookup":
        return r"StreamingResponse|yield\s+f?['\"]data:|yield\s+['\"]data:|async\s+def\s+chat_stream"

    if task.kind == "control_flow_lookup":
        return r"data:\s*\[DONE\]|\[DONE\]"

    if task.kind == "callsite_lookup":
        return r"return\s+await\s+execute_tool_block|await\s+execute_tool_block|execute_tool_block\s*\("

    if task.kind == "save_callsite_lookup":
        return r"save_assistant_response|message_saved|full_response"

    if task.kind == "backend_event_emit_lookup":
        return re.escape(target) + r"|yield\s+f?['\"]data:|yield\s+['\"]data:"

    if task.kind == "streaming_event_emit_lookup":
        # Target first. Generic yield lines are useful only after exact event
        # references, otherwise they outrank real emission blocks.
        return re.escape(target) + r"|['\"]type['\"]\s*:\s*['\"]" + re.escape(target) + r"['\"]|yield\s+f?['\"]data:|yield\s+['\"]data:"

    return re.escape(target)

def _context_from_match(task: SubTask, match: tuple[str, int], state: ContractState, call_prefix: str) -> dict[str, Any]:
    path, line = match
    context = _call_tool(
        "read_file_around",
        {"path": path, "line": line, "context": 60},
        f"{call_prefix}_read_context",
        state,
    )
    return {
        "status": "FOUND",
        "path": path,
        "line": line,
        "context": context,
    }


def execute_subtask(state: ContractState, task: SubTask) -> ContractState:
    state.current_subtask_id = task.id
    task.tool_attempts += 1

    state.internal_events.append({
        "type": "subtask_start",
        "task_id": task.id,
        "kind": task.kind,
        "target": task.target,
        "scope": task.source_scope,
    })

    pattern = _grep_pattern_for_task(task)
    path = _path_for_task(task)

    grep_out = _call_tool(
        "grep",
        {
            "pattern": pattern,
            "path": path,
            "max_matches": 60,
            "include_low_authority": True,
        },
        f"{task.id}_grep_1",
        state,
    )

    if "No matches." in grep_out:
        task.evidence_payload = {
            "status": "CRITICAL_NO_MATCH",
            "target": task.target,
            "source_scope": task.source_scope,
            "reason": "Exact/primary lookup returned no matches.",
        }
        return verify_subtask(state, task)

    ranked_matches = _rank_matches_with_provenance(
        grep_out,
        task,
        state.workspace,
    )
    match = (
        (ranked_matches[0]["path"], ranked_matches[0]["line"])
        if ranked_matches
        else None
    )

    if ranked_matches:
        audited_candidates = list(ranked_matches[:10])
        audited_paths = {
            (item.get("path"), item.get("line"))
            for item in audited_candidates
        }
        for item in ranked_matches:
            provenance = item.get("provenance") or {}
            identity = (item.get("path"), item.get("line"))
            if (
                provenance.get("role") != "source"
                and identity not in audited_paths
            ):
                audited_candidates.append(item)
                audited_paths.add(identity)
                if len(audited_candidates) >= 12:
                    break

        provenance_audit = {
            "type": "source_provenance_audit",
            "task_id": task.id,
            "selected": {
                key: ranked_matches[0].get(key)
                for key in (
                    "path",
                    "line",
                    "task_score",
                    "provenance_score",
                    "total_score",
                    "provenance",
                )
            },
            "candidates": [
                {
                    key: item.get(key)
                    for key in (
                        "path",
                        "line",
                        "task_score",
                        "provenance_score",
                        "total_score",
                        "provenance",
                    )
                }
                for item in audited_candidates
            ],
        }
        state.internal_events.append(provenance_audit)
        _emit(
            "source_provenance_audit",
            data=provenance_audit,
        )

    # For stream generator, prefer context around a concrete handler line from route task if available.
    if task.kind == "stream_generator_lookup":
        route_task = state.task_contract.get("task_001")
        if route_task and isinstance(route_task.evidence_payload, dict):
            route_line = route_task.evidence_payload.get("line")
            route_path = route_task.evidence_payload.get("path")
            if route_path and route_line:
                match = (str(route_path), int(route_line))

    if match:
        task.evidence_payload = _context_from_match(task, match, state, task.id)
    else:
        task.evidence_payload = {
            "status": "FOUND_UNEXPANDED",
            "grep_output": grep_out,
            "target": task.target,
            "source_scope": task.source_scope,
        }

    return verify_subtask(state, task)


def _payload_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str) if not isinstance(payload, str) else payload


def verify_subtask(state: ContractState, task: SubTask) -> ContractState:
    payload_text = _payload_text(task.evidence_payload)

    if "CRITICAL_NO_MATCH" in payload_text:
        task.is_unresolved = True
        task.internal_verification_report = "Unresolved: primary lookup returned no matches."
        state.task_contract[task.id] = task
        return state

    ok = False

    if task.kind == "handler_lookup":
        ok = task.target.lower() in payload_text.lower() and ("json.type" in payload_text or "addEventListener" in payload_text)

    elif task.kind == "route_lookup":
        ok = task.target.lower() in payload_text.lower() and ("@router." in payload_text or "async def" in payload_text)

    elif task.kind == "stream_generator_lookup":
        lower = payload_text.lower()
        ok = "streamingresponse" in lower or ("yield" in lower and "data:" in lower)

    elif task.kind == "control_flow_lookup":
        ok = "[done]" in payload_text.lower()

    elif task.kind == "callsite_lookup":
        lower = payload_text.lower()
        ok = "execute_tool_block" in lower and ("await" in lower or "return" in lower)

    elif task.kind == "save_callsite_lookup":
        ok = "save_assistant_response" in payload_text or "message_saved" in payload_text

    elif task.kind == "backend_event_emit_lookup":
        lower = payload_text.lower()
        target = task.target.lower()
        lines = lower.splitlines()
        ok = any(
            target in line and (
                "yield" in line or "data:" in line or "json.dumps" in line or '"type"' in line or "'type'" in line
            )
            for line in lines
        )

    elif task.kind == "streaming_event_emit_lookup":
        lower = payload_text.lower()
        target = task.target.lower()
        lines = lower.splitlines()
        ok = any(
            target in line and (
                "yield" in line or "data:" in line or "json.dumps" in line or '"type"' in line or "'type'" in line
            )
            for line in lines
        )

    else:
        ok = bool(task.evidence_payload)

    if ok:
        task.is_completed = True
        task.internal_verification_report = "Verification passed by deterministic boundary checks."
    else:
        # v0.11.2a: deterministic lookup-only tasks should not repeat the
        # exact same grep/context path three times. If the first selected
        # context fails verification, mark unresolved instead of looping.
        if task.is_lookup_only:
            task.is_unresolved = True
            task.internal_verification_report = "Unresolved: selected evidence did not satisfy deterministic checks."
        elif task.tool_attempts >= task.max_attempts:
            task.is_unresolved = True
            task.internal_verification_report = "Unresolved: max attempts reached without satisfying deterministic checks."
        else:
            task.internal_verification_report = "Verification failed; another attempt may be required."

    state.task_contract[task.id] = task
    return state


def route_contract(state: ContractState) -> str:
    if state.complete_or_unresolved:
        return "finalize"

    ready = state.incomplete_ready_tasks()
    if not ready:
        return "finalize"

    return ready[0].id


def _clean_fact_from_task(task: SubTask) -> dict[str, Any]:
    payload = task.evidence_payload if isinstance(task.evidence_payload, dict) else {"raw": task.evidence_payload}
    return {
        "id": task.id,
        "kind": task.kind,
        "target": task.target,
        "status": "completed" if task.is_completed else "unresolved",
        "path": payload.get("path"),
        "line": payload.get("line"),
        "reason": payload.get("reason"),
        "context": payload.get("context"),
    }


def clean_room_finalizer(state: ContractState) -> str:
    completed = [_clean_fact_from_task(t) for t in state.task_contract.values() if t.is_completed]
    unresolved = [_clean_fact_from_task(t) for t in state.task_contract.values() if t.is_unresolved]

    # Exact no-match / unresolved-only case.
    if unresolved and not completed:
        target = unresolved[0].get("target")
        scope = state.task_contract[next(iter(state.task_contract))].source_scope
        return (
            f"No matching code location was found for `{target}` in the searched scope "
            f"({', '.join(scope) if scope else 'project'})."
        )

    lines: list[str] = []

    if completed:
        if len(completed) == 1:
            fact = completed[0]
            path = fact.get("path") or "unknown path"
            line = fact.get("line")
            target = fact.get("target")
            if line:
                lines.append(f"The relevant location is `{path}` around line {line}, for `{target}`.")
            else:
                lines.append(f"The relevant location is `{path}` for `{target}`.")
        else:
            lines.append("I found the requested locations:")
            for fact in completed:
                path = fact.get("path") or "unknown path"
                line = fact.get("line")
                target = fact.get("target")
                if line:
                    lines.append(f"- `{target}`: `{path}` around line {line}.")
                else:
                    lines.append(f"- `{target}`: `{path}`.")

    if unresolved:
        lines.append("")
        lines.append("Some requested parts could not be verified:")
        for fact in unresolved:
            lines.append(f"- `{fact.get('target')}`: no matching evidence found.")

    # Add short code evidence snippets, stripped of internal verifier chatter.
    snippets: list[str] = []
    for fact in completed:
        ctx = str(fact.get("context") or "")
        interesting = []
        target = str(fact.get("target") or "").lower()
        for raw_line in ctx.splitlines():
            low = raw_line.lower()
            if target and target in low:
                interesting.append(raw_line.strip())
            elif fact.get("kind") == "stream_generator_lookup" and ("streamingresponse" in low or "yield" in low and "data:" in low):
                interesting.append(raw_line.strip())
            elif fact.get("kind") == "callsite_lookup" and "execute_tool_block" in low:
                interesting.append(raw_line.strip())
            elif fact.get("kind") in {"backend_event_emit_lookup", "streaming_event_emit_lookup"}:
                event_target = str(fact.get("target") or "").lower()

                # For backend event emission claims, visible evidence should
                # prioritize the exact event line, not nearby generic yield lines.
                if event_target and event_target in low:
                    interesting.insert(0, raw_line.strip())
                elif (
                    len(interesting) == 0
                    and ("yield" in low or "json.dumps" in low)
                ):
                    # Keep one fallback generic emission line only if no exact
                    # event line has appeared yet.
                    interesting.append(raw_line.strip())
            if len(interesting) >= 3:
                break
        if interesting:
            snippets.extend(interesting)

    if snippets:
        lines.append("")
        lines.append("Evidence:")
        for s in snippets[:6]:
            lines.append(f"- `{s}`")

    return "\n".join(lines).strip()


def run_contract_runtime(
    question: str,
    *,
    legacy_wrapper: bool = True,
    workspace: str | None = None,
) -> None:
    started = time.time()

    if not workspace:
        raise ValueError(
            "Contract runtime requires an explicit workspace."
        )

    resolved_workspace = str(
        Path(workspace).resolve()
    )

    _emit("agent_prep", data={
        "runtime": RUNTIME_VERSION,
        "scope": "file_inspection",
        "workspace": resolved_workspace,
    })

    state = deconstruct_contract(question)
    state.workspace = resolved_workspace

    _emit("task_contract", data={
        "runtime": RUNTIME_VERSION,
        "subtasks": [
            {
                "id": t.id,
                "kind": t.kind,
                "target": t.target,
                "source_scope": t.source_scope,
                "dependencies": t.dependencies,
                "is_lookup_only": t.is_lookup_only,
                "required_evidence": t.required_evidence,
            }
            for t in state.task_contract.values()
        ],
    })

    rounds = 0
    tool_calls_before = len(state.internal_events)

    while True:
        route = route_contract(state)
        if route == "finalize":
            break

        task = state.task_contract[route]
        _emit("agent_step", data={
            "label": "Executing subtask",
            "runtime": RUNTIME_VERSION,
            "subtask_id": task.id,
            "kind": task.kind,
            "target": task.target,
            "tool_rounds": rounds,
        })
        state = execute_subtask(state, task)
        rounds += 1

        if rounds > 20:
            break

    _emit("agent_step", data={
        "label": "Finalizing",
        "runtime": RUNTIME_VERSION,
        "tool_rounds": rounds,
    })

    answer = clean_room_finalizer(state)
    state.user_facing_answer = answer
    _emit_delta(answer)

    tool_call_count = sum(1 for e in state.internal_events if e.get("type") == "tool_result")
    _emit("metrics", data={
        "runtime": RUNTIME_VERSION,
        "model": "deterministic-contract-runtime",
        "requested_model": "deterministic-contract-runtime",
        "elapsed_s": round(time.time() - started, 3),
        "agent_rounds": rounds,
        "agent_tool_calls": tool_call_count,
        "final_chars": len(answer),
        "langgraph_runtime": True,
        "contract_runtime": True,
        "completed_subtasks": sum(1 for t in state.task_contract.values() if t.is_completed),
        "unresolved_subtasks": sum(1 for t in state.task_contract.values() if t.is_unresolved),
        "total_subtasks": len(state.task_contract),
    })

    _emit_done()
