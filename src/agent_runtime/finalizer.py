from __future__ import annotations

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
