from __future__ import annotations

import re
from typing import Any

from src.agent_runtime.exact_terms import exact_term_present


def _norm(text: str | None) -> str:
    return str(text or "").lower().replace("_", " ").replace("-", " ").replace("/", " ").replace("\\", " ")


def _all_observation_text(observations: list[dict[str, Any]] | None) -> str:
    return "\n".join(str(o.get("content") or "") for o in (observations or []))


def _paths(observations: list[dict[str, Any]] | None) -> list[str]:
    return [str(o.get("source_path") or "") for o in (observations or []) if o.get("source_path")]


def should_avoid_runtime_sources(contract: dict[str, Any] | None, query: str | None) -> bool:
    q = _norm(query)
    source_intent = (contract or {}).get("source_intent")

    # If the user asks about the LangGraph/runtime itself, src/agent_runtime is valid.
    if any(x in q for x in ["runtime", "langgraph", "claim support", "evidence ledger", "task contract", "finalizer"]):
        return False

    # If the user asks about app behavior/frontend/chat_routes/old agent loop,
    # runtime implementation files are usually not the answer source.
    return source_intent in {"frontend", "chat_routes", "agent_loop", "backend"}


def filter_observations_for_contract(
    *,
    observations: list[dict[str, Any]],
    contract: dict[str, Any] | None,
    query: str | None,
) -> list[dict[str, Any]]:
    if not should_avoid_runtime_sources(contract, query):
        return observations

    filtered = []
    for obs in observations or []:
        path = str(obs.get("source_path") or "").replace("/", "\\")
        if path.startswith("src\\agent_runtime\\"):
            continue
        filtered.append(obs)

    return filtered


def exact_negative_ready(contract: dict[str, Any] | None, observations: list[dict[str, Any]] | None) -> bool:
    c = contract or {}
    kind = c.get("task_kind")
    exact_terms = c.get("exact_terms") or []

    if kind not in {"exact_negative_lookup", "exact_symbol_lookup", "event_handler_lookup"}:
        return False

    if not exact_terms:
        return False

    # For exact negative/event lookups, if we have explicit "No matches." observations
    # for every exact term in the intended source area, the runtime should stop instead
    # of broadening to vague fragments.
    text = _all_observation_text(observations).lower()

    for term in exact_terms:
        term_l = str(term).lower()
        has_no_match_obs = "no matches" in text and term_l.replace("\\", "") in text.replace("\\", "")
        has_positive_obs = exact_term_present(text, term)
        if has_positive_obs and not has_no_match_obs:
            return False
        if not has_no_match_obs:
            # If the raw term isn't echoed in output, still allow simple no-match
            # when observations are only no-match grep outputs.
            no_match_obs = [
                o for o in observations or []
                if str(o.get("tool_name") or o.get("tool") or "").lower() == "grep"
                and "no matches" in str(o.get("content") or "").lower()
            ]
            if not no_match_obs:
                return False

    return True


def obligation_statuses(
    *,
    contract: dict[str, Any] | None,
    observations: list[dict[str, Any]] | None,
    claim_support: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    c = contract or {}
    obs = observations or []
    all_text = _all_observation_text(obs)
    all_text_norm = _norm(all_text)
    paths = _paths(obs)
    statuses: list[dict[str, Any]] = []

    for ob in c.get("obligations") or []:
        kind = ob.get("kind")
        term = str(ob.get("term") or "")
        ok = False
        reason = ""

        if kind == "exact_term":
            ok = any(exact_term_present(o.get("content", ""), term) for o in obs)
            reason = "exact term present in observations" if ok else "exact term not found in observations"

        elif kind == "route_registration":
            ok = bool(
                re.search(r'@router\.(get|post|put|delete|patch)\s*\(', all_text)
                or "confirmed post /api/chat_stream handled by chat_stream" in all_text.lower()
                or "async def chat_stream" in all_text.lower()
            )
            reason = "route registration evidence found" if ok else "route registration evidence missing"

        elif kind == "stream_generator":
            backend_paths = [p.replace("/", "\\") for p in paths if p.replace("/", "\\").startswith("routes\\")]
            ok = bool(
                backend_paths
                and (
                    "streamingresponse" in all_text_norm
                    or ("async def" in all_text_norm and "yield" in all_text_norm and "data:" in all_text_norm)
                    or "asyncgenerator" in all_text_norm
                )
            )
            reason = "backend stream generator/yield evidence found" if ok else "backend stream generator/yield evidence missing"

        elif kind == "concrete_callsite":
            ok = bool(
                re.search(r'(return\s+)?await\s+[A-Za-z_][A-Za-z0-9_]*\s*\(', all_text)
                or re.search(r'\.invoke\s*\(', all_text)
                or "execute_tool_block(" in all_text
            )
            reason = "concrete callsite found" if ok else "concrete callsite missing"

        elif kind == "frontend_handler_branch":
            ok = bool(
                re.search(r'(else\s+if|if)\s*\(?.{0,120}json\.type.{0,120}', all_text)
                or "json.type ===" in all_text
                or "addEventListener" in all_text
            )
            reason = "frontend handler branch found" if ok else "frontend handler branch missing"

        elif kind == "handler_branch":
            ok = bool(
                re.search(r'(else\s+if|elif|if)\s*\(?.{0,80}(json\.type|data\.get\(["\']type["\'])', all_text)
                or "addEventListener" in all_text
            )
            reason = "handler branch found" if ok else "handler branch missing"

        elif kind == "save_callsite":
            ok = bool("save_assistant_response(" in all_text or "message_saved" in all_text)
            reason = "save/event evidence found" if ok else "save/event evidence missing"

        else:
            ok = (claim_support or {}).get("support_level") == "strong"
            reason = "delegated to claim support"

        new_ob = dict(ob)
        new_ob["status"] = "satisfied" if ok else "missing"
        new_ob["reason"] = reason
        statuses.append(new_ob)

    return statuses


def contract_support_report(
    *,
    contract: dict[str, Any] | None,
    observations: list[dict[str, Any]] | None,
    claim_support: dict[str, Any] | None,
) -> dict[str, Any]:
    c = contract or {}
    statuses = obligation_statuses(contract=c, observations=observations, claim_support=claim_support)
    missing = [s for s in statuses if s.get("status") != "satisfied"]

    if exact_negative_ready(c, observations):
        return {
            "support_level": "negative_complete",
            "obligations": statuses,
            "missing": [],
            "reason": "exact lookup produced no matches in the intended source scope",
            "allow_stop": True,
        }

    if c.get("task_kind") == "compound_code_location":
        if missing:
            return {
                "support_level": "partial",
                "obligations": statuses,
                "missing": missing,
                "reason": "compound task has unsatisfied obligations",
                "allow_stop": False,
            }

    if missing:
        return {
            "support_level": "partial",
            "obligations": statuses,
            "missing": missing,
            "reason": "one or more task obligations are not satisfied",
            "allow_stop": False,
        }

    return {
        "support_level": "strong",
        "obligations": statuses,
        "missing": [],
        "reason": "all task obligations are satisfied",
        "allow_stop": True,
    }


def format_contract_report(report: dict[str, Any] | None) -> str:
    r = report or {}
    rows = [
        "Task-contract verification:",
        f"- support_level: {r.get('support_level', 'unknown')}",
        f"- reason: {r.get('reason', '')}",
        "- obligations:",
    ]
    for ob in r.get("obligations") or []:
        rows.append(f"  - {ob.get('kind')}: {ob.get('term')} => {ob.get('status')} ({ob.get('reason')})")
    return "\n".join(rows)
