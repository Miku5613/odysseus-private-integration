from __future__ import annotations

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.agent_runtime.action_search import (
    build_assist_searches,
    search_assist_needed,
)
from src.agent_runtime.claim_support import (
    evaluate_code_location_support,
    format_claim_support,
    has_strong_claim_support,
)
from src.agent_runtime.contract_verifier import (
    contract_support_report,
    exact_negative_ready,
    filter_observations_for_contract,
    format_contract_report,
)
from src.agent_runtime.events import debug_print, make_event, print_event
from src.agent_runtime.evidence import (
    analyze_tool_result,
    anchor_key,
    extract_anchors,
    format_evidence_hits,
    has_strong_evidence,
    pick_unexpanded_anchor,
)
from src.agent_runtime.evidence_ledger import (
    format_observations,
    observations_from_tool_result,
    pick_context_candidate,
)
from src.agent_runtime.finalizer import (
    FINALIZER_SYSTEM,
    build_structured_finalizer_prompt,
    build_terminal_fallback_answer,
    clean_negative_lookup_wording,
    collect_evidence,
    enforce_finalizer_honesty,
    sanitize_final_answer,
)
from src.agent_runtime.state import AgentState
from src.agent_runtime.tool_adapters import TOOL_MAP, get_tools_for_scope
from src.agent_runtime.task_contract import (
    build_task_contract,
    format_task_contract,
)
from src.agent_runtime.trace_spans import complete_span, start_span


MAX_TOOL_ROUNDS = 10
RUNTIME_VERSION = "v0.10.2"


SYSTEM = SystemMessage(
    content=(
        "You are a read-only local code/file inspection agent. "
        "Use tools when needed. Do not guess. "
        "You may only inspect files inside the workspace. "
        "Do not write, edit, delete, run shell commands, or use web search. "
        "Prefer targeted search: use grep/glob before reading many chunks. "
        "When a tool result gives a file:line anchor, nearby context may be expanded automatically by the runtime. "
        "When enough source-backed evidence exists, stop searching and give the final answer. "
        "Your final answer must include: file path, function/route/event/call if found, exact evidence, and tools used."
    )
)


def _message_content_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def _state_question(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in {"human", "user"}:
            text = _message_content_text(msg).strip()
            if text:
                return text
    return ""


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="batiai/qwen3.5-9b:q6",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        temperature=0.1,
    )


def _emit_evidence_hits(events, hits):
    for hit in hits:
        event = make_event(
            "evidence_hit",
            runtime=RUNTIME_VERSION,
            kind=hit.get("kind"),
            strength=hit.get("strength"),
            source_tool=hit.get("source_tool"),
            summary=hit.get("summary"),
            file=hit.get("file"),
            line=hit.get("line"),
            route=hit.get("route"),
            symbol=hit.get("symbol"),
        )
        print_event(event)
        events.append(event)


def _merge_observations(existing, new):
    rows = list(existing or [])
    seen = {
        (
            o.get("kind"),
            o.get("source_path"),
            o.get("line_start"),
            str(o.get("content", ""))[:120],
        )
        for o in rows
    }

    for obs in new or []:
        key = (
            obs.get("kind"),
            obs.get("source_path"),
            obs.get("line_start"),
            str(obs.get("content", ""))[:120],
        )
        if key not in seen:
            rows.append(obs)
            seen.add(key)

    return rows


def _evaluate_support(state: AgentState) -> dict:
    return evaluate_code_location_support(
        query=_state_question(state),
        observations=state.get("observations", []),
    )


def build_graph():
    llm = make_llm()
    tools = get_tools_for_scope("file_inspection")
    llm_with_tools = llm.bind_tools(tools)

    def task_contract_node(state: AgentState):
        events = list(state.get("events", []))
        query = _state_question(state)
        contract = build_task_contract(query)

        event = make_event(
            "task_contract",
            runtime=RUNTIME_VERSION,
            task_kind=contract.get("task_kind"),
            source_intent=contract.get("source_intent"),
            exact_terms=contract.get("exact_terms"),
            obligations=contract.get("obligations"),
            allow_direct_end=contract.get("allow_direct_end"),
        )
        print_event(event)
        events.append(event)

        return {
            "events": events,
            "task_contract": contract,
        }


    def call_model(state: AgentState):
        events = list(state.get("events", []))
        trace_spans = list(state.get("trace_spans", []))

        span = start_span(
            span_id=f"span_model_{len(trace_spans) + 1}",
            kind="llm",
            name="agent_model_call",
            input={
                "tool_rounds": state.get("tool_rounds", 0),
                "scope": state.get("scope"),
            },
        )

        event = make_event(
            "agent_model_call_start",
            runtime=RUNTIME_VERSION,
            scope=state.get("scope"),
            tool_rounds=state.get("tool_rounds", 0),
        )
        print_event(event)
        events.append(event)

        messages = [SYSTEM] + state["messages"]
        response = llm_with_tools.invoke(messages)

        tool_calls = getattr(response, "tool_calls", None) or []
        span = complete_span(
            span,
            output=str(response.content or "")[:1000],
            ok=True,
            metadata={"tool_calls": [tc.get("name") for tc in tool_calls]},
        )
        trace_spans.append(span)

        if tool_calls:
            names = [tc.get("name") for tc in tool_calls]
            debug_print(f"\n[agent] tool calls requested: {names}")

            event = make_event(
                "agent_tool_calls_requested",
                runtime=RUNTIME_VERSION,
                tools=names,
                count=len(names),
            )
            print_event(event)
            events.append(event)
        else:
            event = make_event(
                "agent_model_final_candidate",
                runtime=RUNTIME_VERSION,
                preview=str(response.content or "")[:500],
            )
            print_event(event)
            events.append(event)

        return {
            "messages": [response],
            "events": events,
            "trace_spans": trace_spans,
        }

    def execute_tools(state: AgentState):
        last = state["messages"][-1]
        tool_messages = []
        events = list(state.get("events", []))
        evidence_hits = list(state.get("evidence_hits", []))
        anchors = list(state.get("anchors", []))
        observations = list(state.get("observations", []))
        trace_spans = list(state.get("trace_spans", []))
        query = _state_question(state)

        for call in getattr(last, "tool_calls", []) or []:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or f"call_{len(trace_spans) + 1}"

            debug_print(f"[tools] executing {name}({args})")

            span = start_span(
                span_id=f"span_tool_{len(trace_spans) + 1}",
                kind="tool",
                name=name or "unknown_tool",
                input=args,
                metadata={"call_id": call_id},
            )

            event = make_event(
                "tool_call_start",
                runtime=RUNTIME_VERSION,
                name=name,
                args=args,
                call_id=call_id,
            )
            print_event(event)
            events.append(event)

            tool_obj = TOOL_MAP.get(name)
            if not tool_obj:
                content = f"ERROR: Tool not allowed or not found: {name}"
                ok = False
            else:
                try:
                    content = tool_obj.invoke(args)
                    ok = not str(content).startswith("ERROR:")
                except Exception as e:
                    content = f"ERROR executing {name}: {e}"
                    ok = False

            content_text = str(content)
            span = complete_span(span, output=content_text, ok=ok)
            trace_spans.append(span)

            preview = content_text.replace("\n", " ")[:260]
            debug_print(f"[tools] result preview: {preview}")

            event = make_event(
                "tool_call_result",
                runtime=RUNTIME_VERSION,
                name=name,
                ok=ok,
                call_id=call_id,
                preview=preview,
                chars=len(content_text),
            )
            print_event(event)
            events.append(event)

            new_observations = observations_from_tool_result(
                tool_name=name or "",
                args=args,
                content=content_text,
                span_id=span["span_id"],
                query=query,
                start_index=len(observations),
            )
            observations = _merge_observations(observations, new_observations)

            new_anchors = extract_anchors(name, args, content_text, query=query)
            if new_anchors:
                seen_anchor_keys = {anchor_key(a) for a in anchors}
                for anchor in new_anchors:
                    key = anchor_key(anchor)
                    if key not in seen_anchor_keys:
                        anchors.append(anchor)
                        seen_anchor_keys.add(key)

            new_hits = analyze_tool_result(name, args, content_text)
            if new_hits:
                evidence_hits.extend(new_hits)
                _emit_evidence_hits(events, new_hits)

            tool_messages.append(
                ToolMessage(
                    content=content_text,
                    tool_call_id=call_id,
                    name=name,
                )
            )

        claim_support = evaluate_code_location_support(
            query=query,
            observations=observations,
        )

        return {
            "messages": tool_messages,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "events": events,
            "evidence_hits": evidence_hits,
            "anchors": anchors,
            "observations": observations,
            "claim_support": claim_support,
            "trace_spans": trace_spans,
        }

    def expand_observation_context(state: AgentState):
        events = list(state.get("events", []))
        evidence_hits = list(state.get("evidence_hits", []))
        anchors = list(state.get("anchors", []))
        observations = list(state.get("observations", []))
        expanded_observations = list(state.get("expanded_observations", []))
        trace_spans = list(state.get("trace_spans", []))
        query = _state_question(state)

        candidate = pick_context_candidate(
            observations,
            expanded_observations,
            query=query,
            min_confidence=18,
        )

        # Fallback compatibility with v0.8.2 anchors.
        if not candidate:
            anchor = pick_unexpanded_anchor(
                anchors,
                state.get("expanded_anchors", []),
                min_score=12,
            )
            if anchor:
                candidate = {
                    "observation_id": anchor_key(anchor),
                    "source_path": anchor.get("file"),
                    "line_start": anchor.get("line"),
                    "content": anchor.get("text", ""),
                    "confidence": anchor.get("score", 0),
                }

        if not candidate:
            claim_support = evaluate_code_location_support(
                query=query,
                observations=observations,
            )
            return {
                "events": events,
                "evidence_hits": evidence_hits,
                "anchors": anchors,
                "observations": observations,
                "expanded_observations": expanded_observations,
                "claim_support": claim_support,
                "trace_spans": trace_spans,
            }

        obs_id = str(candidate.get("observation_id"))
        expanded_observations.append(obs_id)

        name = "read_file_around"
        args = {
            "path": candidate.get("source_path"),
            "line": int(candidate.get("line_start") or 1),
            "context": 36,
        }
        call_id = f"auto_expand_obs_{len(expanded_observations)}"

        event = make_event(
            "runtime_followup",
            runtime=RUNTIME_VERSION,
            reason="expand_best_observation",
            observation_id=obs_id,
            file=candidate.get("source_path"),
            line=candidate.get("line_start"),
            confidence=candidate.get("confidence"),
            tool_rounds=state.get("tool_rounds", 0),
        )
        print_event(event)
        events.append(event)

        span = start_span(
            span_id=f"span_tool_{len(trace_spans) + 1}",
            kind="tool",
            name=name,
            input=args,
            metadata={"call_id": call_id, "automatic": True, "derived_from_observation": obs_id},
        )

        event = make_event(
            "tool_call_start",
            runtime=RUNTIME_VERSION,
            name=name,
            args=args,
            call_id=call_id,
            automatic=True,
            reason="expand_best_observation",
        )
        print_event(event)
        events.append(event)

        tool_obj = TOOL_MAP.get(name)
        if not tool_obj:
            content = f"ERROR: Tool not allowed or not found: {name}"
            ok = False
        else:
            try:
                content = tool_obj.invoke(args)
                ok = not str(content).startswith("ERROR:")
            except Exception as e:
                content = f"ERROR executing {name}: {e}"
                ok = False

        content_text = str(content)
        span = complete_span(span, output=content_text, ok=ok)
        trace_spans.append(span)

        event = make_event(
            "tool_call_result",
            runtime=RUNTIME_VERSION,
            name=name,
            ok=ok,
            call_id=call_id,
            preview=content_text.replace("\n", " ")[:260],
            chars=len(content_text),
            automatic=True,
        )
        print_event(event)
        events.append(event)

        new_observations = observations_from_tool_result(
            tool_name=name,
            args=args,
            content=content_text,
            span_id=span["span_id"],
            query=query,
            start_index=len(observations),
        )
        observations = _merge_observations(observations, new_observations)

        new_anchors = extract_anchors(name, args, content_text, query=query)
        if new_anchors:
            seen_anchor_keys = {anchor_key(a) for a in anchors}
            for new_anchor in new_anchors:
                new_key = anchor_key(new_anchor)
                if new_key not in seen_anchor_keys:
                    anchors.append(new_anchor)
                    seen_anchor_keys.add(new_key)

        new_hits = analyze_tool_result(name, args, content_text)
        if new_hits:
            evidence_hits.extend(new_hits)
            _emit_evidence_hits(events, new_hits)

        claim_support = evaluate_code_location_support(
            query=query,
            observations=observations,
        )

        return {
            "messages": [
                ToolMessage(
                    content=content_text,
                    tool_call_id=call_id,
                    name=name,
                )
            ],
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "events": events,
            "evidence_hits": evidence_hits,
            "anchors": anchors,
            "observations": observations,
            "expanded_observations": expanded_observations,
            "expanded_anchors": list(state.get("expanded_anchors", [])),
            "claim_support": claim_support,
            "trace_spans": trace_spans,
        }

    def targeted_action_search(state: AgentState):
        """Runtime assist node for action/implementation questions.

        This is not a task-specific answer. It performs targeted grep searches
        when the current observations are insufficient for action-like queries.
        """

        events = list(state.get("events", []))
        evidence_hits = list(state.get("evidence_hits", []))
        anchors = list(state.get("anchors", []))
        observations = list(state.get("observations", []))
        trace_spans = list(state.get("trace_spans", []))
        query = _state_question(state)
        action_searches_done = list(state.get("action_searches_done", []))

        searches = build_assist_searches(query=query, observations=observations)
        searches = [
            s for s in searches
            if f"{s['args'].get('path')}::{s['args'].get('pattern')}" not in set(action_searches_done)
        ][:3]

        if not searches:
            claim_support, contract_report, observations = _evaluate_support_with_contract(state | {"observations": observations})
            return {
                "events": events,
                "evidence_hits": evidence_hits,
                "anchors": anchors,
                "observations": observations,
                "trace_spans": trace_spans,
                "claim_support": claim_support,
                "action_searches_done": action_searches_done,
            }

        tool_messages = []

        for search in searches:
            name = search["tool"]
            args = search["args"]
            key = f"{args.get('path')}::{args.get('pattern')}"
            action_searches_done.append(key)
            call_id = f"auto_action_search_{len(action_searches_done)}"

            event = make_event(
                "runtime_followup",
                runtime=RUNTIME_VERSION,
                reason=search.get("reason", "targeted_action_search"),
                tool=name,
                args=args,
                tool_rounds=state.get("tool_rounds", 0),
            )
            print_event(event)
            events.append(event)

            span = start_span(
                span_id=f"span_tool_{len(trace_spans) + 1}",
                kind="tool",
                name=name,
                input=args,
                metadata={"call_id": call_id, "automatic": True, "reason": search.get("reason")},
            )

            event = make_event(
                "tool_call_start",
                runtime=RUNTIME_VERSION,
                name=name,
                args=args,
                call_id=call_id,
                automatic=True,
                reason=search.get("reason"),
            )
            print_event(event)
            events.append(event)

            tool_obj = TOOL_MAP.get(name)
            if not tool_obj:
                content = f"ERROR: Tool not allowed or not found: {name}"
                ok = False
            else:
                try:
                    content = tool_obj.invoke(args)
                    ok = not str(content).startswith("ERROR:")
                except Exception as e:
                    content = f"ERROR executing {name}: {e}"
                    ok = False

            content_text = str(content)
            span = complete_span(span, output=content_text, ok=ok)
            trace_spans.append(span)

            event = make_event(
                "tool_call_result",
                runtime=RUNTIME_VERSION,
                name=name,
                ok=ok,
                call_id=call_id,
                preview=content_text.replace("\n", " ")[:260],
                chars=len(content_text),
                automatic=True,
            )
            print_event(event)
            events.append(event)

            new_observations = observations_from_tool_result(
                tool_name=name,
                args=args,
                content=content_text,
                span_id=span["span_id"],
                query=query,
                start_index=len(observations),
            )
            observations = _merge_observations(observations, new_observations)

            new_anchors = extract_anchors(name, args, content_text, query=query)
            if new_anchors:
                seen_anchor_keys = {anchor_key(a) for a in anchors}
                for anchor in new_anchors:
                    k = anchor_key(anchor)
                    if k not in seen_anchor_keys:
                        anchors.append(anchor)
                        seen_anchor_keys.add(k)

            new_hits = analyze_tool_result(name, args, content_text)
            if new_hits:
                evidence_hits.extend(new_hits)
                _emit_evidence_hits(events, new_hits)

            tool_messages.append(
                ToolMessage(
                    content=content_text,
                    tool_call_id=call_id,
                    name=name,
                )
            )

        claim_support, contract_report, observations = _evaluate_support_with_contract(state | {"observations": observations})

        return {
            "messages": tool_messages,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "events": events,
            "evidence_hits": evidence_hits,
            "anchors": anchors,
            "observations": observations,
            "trace_spans": trace_spans,
            "claim_support": claim_support,
            "action_searches_done": action_searches_done,
        }


    def finalizer(state: AgentState):
        debug_print("\n[agent] finalizing from structured evidence ledger.\n")

        events = list(state.get("events", []))
        claim_support = state.get("claim_support") or _evaluate_support(state)
        strong = has_strong_claim_support(claim_support) or has_strong_evidence(state.get("evidence_hits", []))

        event = make_event(
            "finalizer_start",
            runtime=RUNTIME_VERSION,
            tool_rounds=state.get("tool_rounds", 0),
            strong_evidence=strong,
            claim_support=claim_support.get("support_level"),
            observations=len(state.get("observations", [])),
            trace_spans=len(state.get("trace_spans", [])),
        )
        print_event(event)
        events.append(event)

        structured_claim_support = format_claim_support(claim_support)
        observations_summary = format_observations(state.get("observations", []), limit=14)
        verified_hits = format_evidence_hits(state.get("evidence_hits", []))
        raw_tool_evidence = collect_evidence(state["messages"])
        contract_summary = format_task_contract(state.get("task_contract", {}))
        contract_report_summary = format_contract_report(claim_support.get("contract_report", {}))

        final_prompt = build_structured_finalizer_prompt(
            claim_support=contract_summary + "\n\n" + contract_report_summary + "\n\n" + structured_claim_support + "\n\n" + verified_hits,
            observations=observations_summary,
            raw_tool_evidence=raw_tool_evidence,
        )
        response = llm.invoke([FINALIZER_SYSTEM, final_prompt])

        evidence_for_sanitizer = (
            structured_claim_support
            + "\n\n"
            + observations_summary
            + "\n\n"
            + verified_hits
            + "\n\n"
            + raw_tool_evidence
        )
        final_message = sanitize_final_answer(str(response.content or ""), evidence_for_sanitizer)
        honest_content = enforce_finalizer_honesty(
            answer=str(final_message.content or ""),
            claim_support=claim_support,
            observations_summary=observations_summary,
        )

        if not str(honest_content or "").strip():
            honest_content = build_terminal_fallback_answer(
                claim_support=claim_support,
                observations_summary=observations_summary,
                task_contract=contract_summary,
            )

        honest_content = clean_negative_lookup_wording(
            answer=honest_content,
            claim_support=claim_support,
        )

        final_message.content = honest_content

        event = make_event(
            "final_answer",
            runtime=RUNTIME_VERSION,
            preview=str(final_message.content or "")[:700],
            content=str(final_message.content or ""),
            chars=len(str(final_message.content or "")),
        )
        print_event(event)
        events.append(event)

        event = make_event(
            "agent_done",
            runtime=RUNTIME_VERSION,
            reason="finalized",
        )
        print_event(event)
        events.append(event)

        return {
            "messages": [final_message],
            "events": events,
            "claim_support": claim_support,
            "terminal_answer": {
                "content": str(final_message.content or ""),
                "chars": len(str(final_message.content or "")),
                "claim_support": claim_support.get("support_level"),
                "runtime": RUNTIME_VERSION,
            },
        }

    def _evaluate_support_with_contract(state: AgentState) -> tuple[dict, dict, list]:
        query = _state_question(state)
        contract = state.get("task_contract", {})
        raw_observations = list(state.get("observations", []))
        observations = filter_observations_for_contract(
            observations=raw_observations,
            contract=contract,
            query=query,
        )
        claim_support = evaluate_code_location_support(query=query, observations=observations)
        contract_report = contract_support_report(
            contract=contract,
            observations=observations,
            claim_support=claim_support,
        )

        # Compound and exact-negative tasks are governed by the contract.
        if contract_report.get("support_level") == "strong":
            claim_support = dict(claim_support)
            claim_support["support_level"] = "strong"
            claim_support["reason"] = contract_report.get("reason")
            claim_support["contract_report"] = contract_report
        elif contract_report.get("support_level") == "negative_complete":
            claim_support = dict(claim_support)
            claim_support["support_level"] = "negative_complete"
            claim_support["reason"] = contract_report.get("reason")
            claim_support["contract_report"] = contract_report
        elif contract.get("task_kind") == "compound_code_location":
            claim_support = dict(claim_support)
            claim_support["support_level"] = "partial"
            claim_support["reason"] = contract_report.get("reason")
            claim_support["contract_report"] = contract_report

        return claim_support, contract_report, observations


    def route_after_model(state: AgentState):
        last = state["messages"][-1]
        has_tool_calls = bool(getattr(last, "tool_calls", None))

        if has_tool_calls:
            if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
                return "finalizer"
            return "tools"

        event = make_event(
            "runtime_followup",
            runtime=RUNTIME_VERSION,
            reason="terminal_finalizer_required_no_direct_end",
            tool_rounds=state.get("tool_rounds", 0),
        )
        print_event(event)
        return "finalizer"

    def route_after_tools(state: AgentState):
        claim_support = state.get("claim_support") or _evaluate_support(state)

        if has_strong_claim_support(claim_support) or has_strong_evidence(state.get("evidence_hits", [])):
            event = make_event(
                "runtime_stop_condition",
                runtime=RUNTIME_VERSION,
                reason="strong_claim_support_detected",
                tool_rounds=state.get("tool_rounds", 0),
                claim_support=claim_support.get("support_level"),
                support_reason=claim_support.get("reason"),
            )
            print_event(event)
            return "finalizer"

        candidate = pick_context_candidate(
            state.get("observations", []),
            state.get("expanded_observations", []),
            query=_state_question(state),
            min_confidence=18,
        )
        if candidate is not None:
            return "expand_observation"

        if exact_negative_ready(state.get("task_contract", {}), state.get("observations", [])):
            event = make_event(
                "runtime_stop_condition",
                runtime=RUNTIME_VERSION,
                reason="exact_negative_lookup_complete",
                tool_rounds=state.get("tool_rounds", 0),
                claim_support="negative_complete",
            )
            print_event(event)
            return "finalizer"

        if search_assist_needed(_state_question(state), claim_support) and not state.get("action_searches_done"):
            event = make_event(
                "runtime_followup",
                runtime=RUNTIME_VERSION,
                reason="claim_support_incomplete_targeted_action_search",
                claim_support=claim_support.get("support_level"),
                support_reason=claim_support.get("reason"),
                tool_rounds=state.get("tool_rounds", 0),
            )
            print_event(event)
            return "targeted_action_search"

        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            event = make_event(
                "runtime_stop_condition",
                runtime=RUNTIME_VERSION,
                reason="tool_budget_reached",
                tool_rounds=state.get("tool_rounds", 0),
                claim_support=claim_support.get("support_level"),
                support_reason=claim_support.get("reason"),
            )
            print_event(event)
            return "finalizer"

        return "agent"

    def route_after_observation_expand(state: AgentState):
        claim_support = state.get("claim_support") or _evaluate_support(state)

        if has_strong_claim_support(claim_support) or has_strong_evidence(state.get("evidence_hits", [])):
            event = make_event(
                "runtime_stop_condition",
                runtime=RUNTIME_VERSION,
                reason="strong_claim_support_detected",
                tool_rounds=state.get("tool_rounds", 0),
                claim_support=claim_support.get("support_level"),
                support_reason=claim_support.get("reason"),
            )
            print_event(event)
            return "finalizer"

        if exact_negative_ready(state.get("task_contract", {}), state.get("observations", [])):
            event = make_event(
                "runtime_stop_condition",
                runtime=RUNTIME_VERSION,
                reason="exact_negative_lookup_complete",
                tool_rounds=state.get("tool_rounds", 0),
                claim_support="negative_complete",
            )
            print_event(event)
            return "finalizer"

        if search_assist_needed(_state_question(state), claim_support) and not state.get("action_searches_done"):
            event = make_event(
                "runtime_followup",
                runtime=RUNTIME_VERSION,
                reason="claim_support_incomplete_targeted_action_search",
                claim_support=claim_support.get("support_level"),
                support_reason=claim_support.get("reason"),
                tool_rounds=state.get("tool_rounds", 0),
            )
            print_event(event)
            return "targeted_action_search"

        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            event = make_event(
                "runtime_stop_condition",
                runtime=RUNTIME_VERSION,
                reason="tool_budget_reached_after_observation_expansion",
                tool_rounds=state.get("tool_rounds", 0),
                claim_support=claim_support.get("support_level"),
                support_reason=claim_support.get("reason"),
            )
            print_event(event)
            return "finalizer"

        return "agent"

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("task_contract", task_contract_node)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", execute_tools)
    graph_builder.add_node("expand_observation", expand_observation_context)
    graph_builder.add_node("targeted_action_search", targeted_action_search)
    graph_builder.add_node("finalizer", finalizer)

    graph_builder.add_edge(START, "task_contract")
    graph_builder.add_edge("task_contract", "agent")
    graph_builder.add_conditional_edges("agent", route_after_model)
    graph_builder.add_conditional_edges("tools", route_after_tools)
    graph_builder.add_conditional_edges("expand_observation", route_after_observation_expand)
    graph_builder.add_conditional_edges("targeted_action_search", route_after_tools)
    graph_builder.add_edge("finalizer", END)

    return graph_builder.compile()
