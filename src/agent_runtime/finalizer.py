from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


FINALIZER_SYSTEM = SystemMessage(
    content=(
        "You are finalizing a local code/file inspection task. "
        "Do not call tools. Use only the evidence provided. "
        "Plain English final answer only. "
        "Mention uncertainty if evidence is incomplete. "
        "Prefer exact file paths, function names, route names, and line numbers from evidence."
    )
)


def collect_evidence(messages: list) -> str:
    chunks = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "tool")
            content = str(msg.content)
            if len(content) > 3500:
                content = content[:3500] + "\n... [truncated]"
            chunks.append(f"TOOL RESULT: {name}\n{content}")

    return "\n\n---\n\n".join(chunks) or "No tool evidence was collected."


def fallback_final_answer(evidence: str) -> str:
    return (
        "I could not confidently complete this from runtime-verified strong evidence.\\n\\n"
        "What I found may be partial or incomplete, so I should not overclaim.\\n\\n"
        f"Collected evidence:\\n{evidence[:3000]}"
    )

def build_finalizer_prompt(evidence: str) -> HumanMessage:
    return HumanMessage(
        content=(
            "Use the evidence below to answer the original task.\n"
            "If the evidence section says there are no runtime-verified evidence hits, or only raw partial evidence, say that evidence is incomplete before giving any tentative answer.\n"
            "Do not call tools. Do not output XML. Do not output JSON. Do not write <tool_call>.\n"
            "Plain English final answer only.\n\n"
            "Required answer format:\n"
            "1. File path\n"
            "2. Function/route if known\n"
            "3. Exact evidence with line numbers if available\n"
            "4. Tools used\n\n"
            f"Evidence:\n{evidence}"
        )
    )


def sanitize_final_answer(text: str, evidence: str) -> AIMessage:
    cleaned = str(text or "").strip()

    if "<tool_call" in cleaned.lower() or '"name":' in cleaned.lower() or not cleaned:
        return AIMessage(content=fallback_final_answer(evidence))

    return AIMessage(content=cleaned)



def build_structured_finalizer_prompt(
    *,
    claim_support: str,
    observations: str,
    raw_tool_evidence: str,
) -> HumanMessage:
    return HumanMessage(
        content=(
            "Use the structured claim support and observations below to answer the original task.\n"
            "Do not call tools. Do not output XML. Do not output JSON. Do not write <tool_call>.\n"
            "Plain English final answer only.\n\n"
            "Rules:\n"
            "- If claim support is strong, answer confidently and cite the file/line observations.\n"
            "- If claim support is partial or insufficient, clearly say evidence is incomplete.\n"
            "- If claim support is partial or insufficient, do not present the best weak observation as the answer. Explain what was found and what still needs verification.\n"
            "- Never answer with prompt rules, imports, docstrings, or headers when the user asked for implementation/execution behavior.\n"
            "- Prefer source-code observations over generated artifacts, logs, reports, or patch files.\n\n"
            "Required answer format:\n"
            "1. File path\n"
            "2. Function/route/event/call if known\n"
            "3. Exact evidence with line numbers if available\n"
            "4. Tools/observations used\n\n"
            f"{claim_support}\n\n"
            f"Top observations:\n{observations}\n\n"
            f"Raw tool evidence:\n{raw_tool_evidence}"
        )
    )



def enforce_finalizer_honesty(
    *,
    answer: str,
    claim_support: dict,
    observations_summary: str,
) -> str:
    text = str(answer or "").strip()
    support = (claim_support or {}).get("support_level", "unknown")
    reason = (claim_support or {}).get("reason", "")

    if not text:
        return (
            "Evidence is incomplete. I could not produce a supported final answer from the collected observations.\n\n"
            f"Claim support: {support}\n"
            f"Reason: {reason}\n\n"
            f"Top observations:\n{observations_summary[:1500]}"
        )

    if support != "strong":
        lower = text.lower()
        # If the model tries to answer as if supported, replace it with an honest
        # evidence-incomplete answer.
        if "evidence is incomplete" not in lower and "not supported" not in lower and "could not" not in lower:
            return (
                "Evidence is incomplete. I found some related observations, but they do not strongly support the requested code location yet.\n\n"
                f"Claim support: {support}\n"
                f"Reason: {reason}\n\n"
                f"Top observations:\n{observations_summary[:1500]}"
            )

    return text



def build_terminal_fallback_answer(
    *,
    claim_support: dict,
    observations_summary: str,
    task_contract: str = "",
) -> str:
    support = (claim_support or {}).get("support_level", "unknown")
    reason = (claim_support or {}).get("reason", "")

    return (
        "Evidence is incomplete. I could not produce a fully supported final answer from the collected observations.\\n\\n"
        f"{task_contract}\\n\\n"
        f"Claim support: {support}\\n"
        f"Reason: {reason}\\n\\n"
        f"Top observations:\\n{observations_summary[:1800]}"
    )



def clean_negative_lookup_wording(*, answer: str, claim_support: dict) -> str:
    text = str(answer or "").strip()
    report = (claim_support or {}).get("contract_report", {}) or {}
    if report.get("support_level") != "negative_complete":
        return text

    # Avoid making exact no-match sound like a failed search. For exact-symbol
    # lookup, no exact match is itself a valid result.
    replacements = [
        ("Further verification with a broader search scope or different query terms is required to confirm the event handler's location.", "No matching handler was found for the exact requested term."),
        ("Further verification with a broader search scope or different query terms is required.", "No matching location was found for the exact requested term."),
        ("Evidence is incomplete regarding", "No matching code location was found for"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def build_grounded_worker_finalizer_prompt(
    *,
    objective: str,
    task_contract: str,
    claim_support: str,
    contract_report: str,
    observations_summary: str,
    raw_tool_evidence: str,
) -> str:
    """Build one evidence-grounded PlanStep finalization contract.

    The worker objective is the complete semantic answer contract. Strong
    evidence is necessary but never sufficient by itself: the model must also
    produce a grounded answer that fulfills every requested deliverable.
    """

    schema = {
        "answer": "Grounded step-specific answer using only supplied evidence.",
        "fulfills_objective": True,
        "missing_requirements": [],
        "evidence_paths": ["relative/source/path.py"],
    }

    return "\n".join(
        [
            "You are finalizing exactly one already-resolved worker PlanStep.",
            "Do not answer the broader original user request.",
            "Do not call tools.",
            "Use only the supplied source evidence.",
            "",
            "The PLANSTEP OBJECTIVE below is the complete answer contract.",
            "Address every deliverable it requests, not merely the first one.",
            "For example, if it asks to identify implementation files AND explain",
            "their responsibilities, a path list or raw grep snippets alone do",
            "not fulfill the objective.",
            "",
            "Completion rules:",
            "1. Evidence strength and task fulfillment are separate gates.",
            "2. Set fulfills_objective=true only when the answer itself satisfies",
            "   every requested deliverable from the PlanStep objective.",
            "3. Never invent responsibilities, relationships, behavior, paths,",
            "   symbols, or conclusions not supported by the supplied evidence.",
            "4. If evidence is strong for only part of the objective, set",
            "   fulfills_objective=false and list the missing requirements.",
            "5. Prefer concise file-to-responsibility mappings when the objective",
            "   asks for implementation ownership or component responsibilities.",
            "6. Do not return raw retrieval debris as the final answer.",
            "7. Return one JSON object only, matching the required schema.",
            "",
            "PLANSTEP OBJECTIVE:",
            str(objective or "").strip(),
            "",
            task_contract,
            "",
            claim_support,
            "",
            contract_report,
            "",
            "TOP SOURCE OBSERVATIONS:",
            observations_summary,
            "",
            "RAW SOURCE EVIDENCE:",
            raw_tool_evidence,
            "",
            "REQUIRED OUTPUT SCHEMA:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def parse_grounded_worker_finalizer_response(
    text: str,
) -> dict[str, Any]:
    """Parse and validate one grounded worker-finalization JSON object."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Grounded worker finalizer returned an empty response")

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    payload = None
    try:
        decoded = json.loads(candidate)
        if isinstance(decoded, dict):
            payload = decoded
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                decoded, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                payload = decoded
                break

    if payload is None:
        raise ValueError(
            "Grounded worker finalizer did not return one valid JSON object"
        )

    answer = payload.get("answer")
    fulfills = payload.get("fulfills_objective")
    missing = payload.get("missing_requirements")
    paths = payload.get("evidence_paths")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            "Grounded worker finalizer answer must be a non-empty string"
        )
    if not isinstance(fulfills, bool):
        raise ValueError(
            "Grounded worker finalizer fulfills_objective must be boolean"
        )
    if not isinstance(missing, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in missing
    ):
        raise ValueError(
            "Grounded worker finalizer missing_requirements must be a list "
            "of non-empty strings"
        )
    if not isinstance(paths, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in paths
    ):
        raise ValueError(
            "Grounded worker finalizer evidence_paths must be a list "
            "of non-empty strings"
        )
    if fulfills and missing:
        raise ValueError(
            "A fulfilled objective cannot also report missing requirements"
        )
    if fulfills and not paths:
        raise ValueError(
            "A fulfilled objective must cite at least one evidence path"
        )

    return {
        "answer": answer.strip(),
        "fulfills_objective": fulfills,
        "missing_requirements": tuple(item.strip() for item in missing),
        "evidence_paths": tuple(item.strip() for item in paths),
    }
