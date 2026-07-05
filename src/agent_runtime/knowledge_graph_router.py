
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


KnowledgeIntent = Literal[
    "knowledge_read",
    "knowledge_propose",
    "knowledge_research",
    "unsupported_mutation",
    "not_knowledge",
    "needs_clarification",
]

Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class KnowledgeGraphDecision:
    intent: KnowledgeIntent
    confidence: Confidence
    action_type: str
    target_hint: str
    reason: str
    ambiguity: list[str] = field(default_factory=list)
    clarifying_context: dict[str, object] = field(default_factory=dict)
    allowed_graphs: list[str] = field(default_factory=list)
    blocked_graphs: list[str] = field(default_factory=list)
    is_read_only: bool = True
    write_allowed: bool = False
    needs_human: bool = False

    @property
    def target_runtime(self) -> str:
        if self.intent == "knowledge_read" and self.confidence == "high":
            return "obsidian_read_graph"
        if self.intent == "needs_clarification":
            return "human_in_the_loop"
        if self.intent == "knowledge_propose":
            return "knowledge_propose_staging"
        if self.intent == "knowledge_research":
            return "knowledge_research_sandbox"
        if self.intent == "unsupported_mutation":
            return "blocked"
        return "legacy_or_existing_router"

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "action_type": self.action_type,
            "target_hint": self.target_hint,
            "reason": self.reason,
            "ambiguity": list(self.ambiguity),
            "clarifying_context": dict(self.clarifying_context),
            "allowed_graphs": list(self.allowed_graphs),
            "blocked_graphs": list(self.blocked_graphs),
            "is_read_only": self.is_read_only,
            "write_allowed": self.write_allowed,
            "needs_human": self.needs_human,
            "target_runtime": self.target_runtime,
        }

    def to_trace_lines(self) -> list[str]:
        lines = [
            f"intent: {self.intent}",
            f"confidence: {self.confidence}",
            f"action_type: {self.action_type}",
            f"target_hint: {self.target_hint}",
            f"target_runtime: {self.target_runtime}",
            f"read_only: {self.is_read_only}",
            f"write_allowed: {self.write_allowed}",
            f"needs_human: {self.needs_human}",
            f"reason: {self.reason}",
        ]
        if self.ambiguity:
            lines.append("ambiguity:")
            lines.extend(f"- {x}" for x in self.ambiguity)
        return lines

    def to_trace_text(self) -> str:
        return "\n".join(self.to_trace_lines())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def _count_any(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.I))


def _extract_quoted_target(prompt: str) -> str:
    for quote in ['"', "'"]:
        if quote in prompt:
            parts = prompt.split(quote)
            if len(parts) >= 3 and parts[1].strip():
                return parts[1].strip()
    return ""


READ_PATTERNS = [
    r"\bread\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\bshow\b",
    r"\blist\b",
    r"\bfind\b",
    r"\bsearch\b",
    r"\bwhat did i\b",
    r"\bwhat have i\b",
    r"\bwhat was written\b",
    r"\brecap\b",
    r"\boverview\b",
]

KNOWLEDGE_STRONG_PATTERNS = [
    r"\bobsidian\b",
    r"\bideaverse\b",
    r"\bvault\b",
    r"\bknowledge\s*base\b",
    r"\bmy\s+notes\b",
    r"\bmy\s+note\b",
    r"\bmy\s+docs\b",
    r"\bmy\s+documents\b",
    r"\bactive\s+note\b",
    r"\bdaily\s+note\b",
    r"\bdaily\s+log\b",
    r"\bprogress\s+note\b",
    r"\bpersonal\s+notes\b",
    r"\bwhat\s+did\s+i\s+write\b",
    r"\bwhat\s+did\s+i\s+note\b",
]

KNOWN_VAULT_TARGET_PATTERNS = [
    r"\bAIOS\b",
    r"\bIdeaverse\b",
    r"\bCalendar/Daily\b",
    r"\bDaily\b",
    r"\bWIP\b",
    r"\bRAW\b",
    r"\bAtlas/\b",  # path compatibility only; not used as terminology.
    r"\.md\b",
    r"\[\[.*?\]\]",
]

VAGUE_DOC_PATTERNS = [
    r"\bthe\s+doc\b",
    r"\bthe\s+document\b",
    r"\bthe\s+note\b",
    r"\bthat\s+doc\b",
    r"\bthat\s+document\b",
    r"\bthat\s+note\b",
    r"\bthis\s+doc\b",
    r"\bthis\s+document\b",
    r"\bthis\s+note\b",
    r"\bprogress\s+report\b",
    r"\bdaily\s+log\b",
]

MUTATION_PATTERNS = [
    r"\bwrite\b",
    r"\bappend\b",
    r"\bedit\b",
    r"\bupdate\b",
    r"\bcreate\b",
    r"\badd\b",
    r"\bsave\b",
    r"\bmodify\b",
    r"\bpatch\b",
    r"\breplace\b",
    r"\bdraft\b",
    r"\bmake\s+a\s+note\b",
    r"\bnew\s+note\b",
]

DESTRUCTIVE_PATTERNS = [
    r"\bdelete\b",
    r"\bremove\b",
    r"\bpurge\b",
    r"\brename\b",
    r"\bmove\b",
    r"\bclean\s+up\b",
    r"\breorganize\b",
    r"\bwipe\b",
    r"\bdestroy\b",
    r"\bbulk\s+(?:edit|move|delete|rename)\b",
]

RESEARCH_PATTERNS = [
    r"\bweb\b",
    r"\bonline\b",
    r"\bgoogle\b",
    r"\bsearch\s+the\s+web\b",
    r"\blook\s+up\b",
    r"\bexternal\b",
    r"\bcompare\s+with\s+(?:online|web|current)\b",
    r"\bcurrent\s+docs\b",
    r"\blatest\b",
]

CODEBASE_PATTERNS = [
    r"\bcodebase\b",
    r"\bsource\s+code\b",
    r"\bimplementation\b",
    r"\bwhere\s+is\b",
    r"\bwhere\s+are\b",
    r"\bwhere\s+does\b",
    r"\bfind\s+files\s+related\s+to\b",
    r"\broutes?/",
    r"\bsrc/",
    r"\bstatic/js\b",
    r"\.py\b",
    r"\.js\b",
    r"\bchat_routes\b",
    r"\bnote_routes\b",
    r"\bdocument\s+stream\b",
    r"\btool_output\b",
    r"\bmessage_saved\b",
]

UPLOAD_DOC_PATTERNS = [
    r"\buploaded\b",
    r"\battached\b",
    r"\battachment\b",
    r"\bpdf\b",
    r"\bfile\s+i\s+uploaded\b",
]


def evaluate_knowledge_request(prompt: str, *, recent_context: str = "") -> KnowledgeGraphDecision:
    raw = str(prompt or "")
    low = _norm(raw)
    ctx = _norm(recent_context)

    quoted_target = _extract_quoted_target(raw)

    read_score = _count_any(low, READ_PATTERNS)
    knowledge_score = _count_any(low, KNOWLEDGE_STRONG_PATTERNS)
    known_target_score = _count_any(raw, KNOWN_VAULT_TARGET_PATTERNS)
    mutation_score = _count_any(low, MUTATION_PATTERNS)
    destructive_score = _count_any(low, DESTRUCTIVE_PATTERNS)
    research_score = _count_any(low, RESEARCH_PATTERNS)
    codebase_score = _count_any(low, CODEBASE_PATTERNS)
    upload_score = _count_any(low, UPLOAD_DOC_PATTERNS)
    vague_doc_score = _count_any(low, VAGUE_DOC_PATTERNS)

    context_is_knowledge = _has_any(ctx, [
        r"\bobsidian\b",
        r"\bideaverse\b",
        r"\bvault\b",
        r"\bknowledge\s*base\b",
        r"\bactive\s+note\b",
        r"\bmy\s+notes\b",
        r"\bAIOS\b",
    ])

    note_like_target = _has_any(low, [
        r"\bnote\b",
        r"\bnotes\b",
        r"\bdaily\b",
        r"\blog\b",
        r"\breport\b",
        r"\bjournal\b",
        r"\bdraft\b",
        r"\bindex\b",
    ])

    target_hint = quoted_target or ""
    if not target_hint:
        if _has_any(low, [r"\bactive\s+note\b"]):
            target_hint = "active_note"
        elif _has_any(raw, [r"\bAIOS\b"]):
            target_hint = "AIOS"
        elif _has_any(raw, [r"\bDaily\b", r"\bdaily\s+note\b", r"\bdaily\s+log\b"]):
            target_hint = "daily_notes"
        elif known_target_score:
            target_hint = "known_vault_target"
        else:
            target_hint = "unknown"

    # Clear codebase/upload false positives should not be stolen by the knowledge router.
    if codebase_score >= 2 and knowledge_score == 0 and known_target_score == 0:
        return KnowledgeGraphDecision(
            intent="not_knowledge",
            confidence="high",
            action_type="route_existing_runtime",
            target_hint=target_hint,
            reason="Prompt looks like an Odysseus codebase/runtime lookup, not a personal knowledge-base request.",
            allowed_graphs=["existing_codebase_router"],
            blocked_graphs=["obsidian_read_graph", "knowledge_propose_staging", "knowledge_research_sandbox"],
            is_read_only=True,
            write_allowed=False,
        )

    if upload_score and knowledge_score == 0 and known_target_score == 0:
        return KnowledgeGraphDecision(
            intent="not_knowledge",
            confidence="high",
            action_type="route_existing_runtime",
            target_hint=target_hint,
            reason="Prompt appears to target an uploaded/attached file rather than the Ideaverse knowledge base.",
            allowed_graphs=["uploaded_file_handler_or_existing_router"],
            blocked_graphs=["obsidian_read_graph"],
            is_read_only=True,
            write_allowed=False,
        )

    # Personal recall phrasing like "what did I write" is read-only, not mutation.
    if _has_any(low, [r"\bwhat\s+did\s+i\s+(?:write|note)\b", r"\bwhat\s+have\s+i\s+(?:written|noted)\b"]):
        return KnowledgeGraphDecision(
            intent="knowledge_read",
            confidence="high",
            action_type="read",
            target_hint=target_hint,
            reason="Personal note recall phrasing strongly indicates Ideaverse/knowledge-base reading.",
            allowed_graphs=["obsidian_read_graph"],
            blocked_graphs=["knowledge_propose_staging"],
            is_read_only=True,
            write_allowed=False,
            needs_human=False,
        )

    # Destructive actions are blocked when they appear knowledge/vault-facing or ambiguous enough to risk vault mutation.
    if destructive_score:
        if knowledge_score or known_target_score or vague_doc_score or context_is_knowledge:
            return KnowledgeGraphDecision(
                intent="unsupported_mutation",
                confidence="high",
                action_type="destructive",
                target_hint=target_hint,
                reason="Destructive knowledge-base operation detected. Live note deletion, rename, move, purge, or cleanup is blocked.",
                ambiguity=[],
                allowed_graphs=[],
                blocked_graphs=["obsidian_read_graph", "knowledge_propose_staging", "knowledge_research_sandbox"],
                is_read_only=False,
                write_allowed=False,
                needs_human=True,
            )
        return KnowledgeGraphDecision(
            intent="not_knowledge",
            confidence="medium",
            action_type="destructive_elsewhere",
            target_hint=target_hint,
            reason="Destructive wording detected, but the target does not clearly belong to the Ideaverse knowledge base.",
            ambiguity=["destructive action target is unclear"],
            allowed_graphs=["existing_router_safety_policy"],
            blocked_graphs=["obsidian_read_graph"],
            is_read_only=False,
            write_allowed=False,
            needs_human=True,
        )

    # Research should be isolated from reading/writing. It can later read context, but cannot write.
    if research_score and (knowledge_score or known_target_score or context_is_knowledge or vague_doc_score or note_like_target):
        return KnowledgeGraphDecision(
            intent="knowledge_research",
            confidence="high" if knowledge_score or known_target_score or context_is_knowledge or note_like_target else "medium",
            action_type="research",
            target_hint=target_hint,
            reason="Knowledge-base research request detected. Research must stay separate from edit/create execution.",
            ambiguity=[] if knowledge_score or known_target_score or context_is_knowledge or note_like_target else ["knowledge source is implied but not explicit"],
            allowed_graphs=["knowledge_research_sandbox"],
            blocked_graphs=["obsidian_read_graph", "knowledge_propose_staging"],
            is_read_only=False,
            write_allowed=False,
            needs_human=not (knowledge_score or known_target_score or context_is_knowledge),
        )

    # Mutation becomes Propose, never direct write.
    if mutation_score:
        if knowledge_score or known_target_score or context_is_knowledge or vague_doc_score or note_like_target:
            return KnowledgeGraphDecision(
                intent="knowledge_propose",
                confidence="high" if knowledge_score or known_target_score or context_is_knowledge or note_like_target else "medium",
                action_type="propose",
                target_hint=target_hint,
                reason="Knowledge-base modification request detected. Direct writes are disabled; route only to in-memory proposal/staging later.",
                ambiguity=[] if knowledge_score or known_target_score or context_is_knowledge or note_like_target else ["target note/document is not specific enough"],
                allowed_graphs=["knowledge_propose_staging"],
                blocked_graphs=["obsidian_read_graph"],
                is_read_only=False,
                write_allowed=False,
                needs_human=True,
            )

    # Strong read routing.
    if read_score and (knowledge_score or known_target_score or context_is_knowledge):
        # Vague target still needs human unless active note / explicit path / known folder makes it clear.
        if vague_doc_score and target_hint == "unknown" and not context_is_knowledge:
            return KnowledgeGraphDecision(
                intent="needs_clarification",
                confidence="medium",
                action_type="read",
                target_hint=target_hint,
                reason="The request appears read-only, but the target document/note source is ambiguous.",
                ambiguity=[
                    "could refer to Ideaverse/Obsidian",
                    "could refer to uploaded files",
                    "could refer to Odysseus code/docs",
                ],
                clarifying_context={
                    "question_goal": "identify_source",
                    "options": ["Ideaverse/Obsidian note", "uploaded file", "Odysseus code/docs"],
                },
                allowed_graphs=[],
                blocked_graphs=["obsidian_read_graph"],
                is_read_only=True,
                write_allowed=False,
                needs_human=True,
            )

        return KnowledgeGraphDecision(
            intent="knowledge_read",
            confidence="high",
            action_type="read",
            target_hint=target_hint,
            reason="High-confidence read-only Ideaverse/knowledge-base request.",
            allowed_graphs=["obsidian_read_graph"],
            blocked_graphs=["knowledge_propose_staging", "knowledge_research_sandbox"],
            is_read_only=True,
            write_allowed=False,
            needs_human=False,
        )

    # Vague docs/notes are clarification, not hallucinated execution.
    if read_score and vague_doc_score:
        return KnowledgeGraphDecision(
            intent="needs_clarification",
            confidence="low",
            action_type="read",
            target_hint=target_hint,
            reason="The prompt uses vague note/document language without enough source context.",
            ambiguity=[
                "source is unclear",
                "target document/note is unnamed",
                "multiple runtimes could apply",
            ],
            clarifying_context={
                "question_goal": "identify_source",
                "options": ["Ideaverse/Obsidian note", "uploaded file", "Odysseus code/docs", "active editor document"],
            },
            allowed_graphs=[],
            blocked_graphs=["obsidian_read_graph", "existing_codebase_router"],
            is_read_only=True,
            write_allowed=False,
            needs_human=True,
        )

    # Personal "what did I write/note" should usually clarify or read depending on strength.
    if _has_any(low, [r"\bwhat\s+did\s+i\s+(?:write|note)\b", r"\bmy\s+notes\b"]):
        return KnowledgeGraphDecision(
            intent="knowledge_read",
            confidence="high",
            action_type="read",
            target_hint=target_hint,
            reason="Personal note recall phrasing strongly indicates Ideaverse/knowledge-base reading.",
            allowed_graphs=["obsidian_read_graph"],
            blocked_graphs=["knowledge_propose_staging"],
            is_read_only=True,
            write_allowed=False,
            needs_human=False,
        )

    # If it mentions knowledge targets but lacks a clear read/mutate/research action, ask.
    if knowledge_score or known_target_score or context_is_knowledge:
        return KnowledgeGraphDecision(
            intent="needs_clarification",
            confidence="medium",
            action_type="unknown",
            target_hint=target_hint,
            reason="The prompt appears related to Ideaverse/knowledge-base material, but the requested action is unclear.",
            ambiguity=["action is unclear"],
            clarifying_context={
                "question_goal": "identify_action",
                "options": ["read/list/summarize", "draft/propose", "research", "something else"],
            },
            allowed_graphs=[],
            blocked_graphs=["obsidian_read_graph", "knowledge_propose_staging", "knowledge_research_sandbox"],
            is_read_only=True,
            write_allowed=False,
            needs_human=True,
        )

    return KnowledgeGraphDecision(
        intent="not_knowledge",
        confidence="low",
        action_type="route_existing_runtime",
        target_hint=target_hint,
        reason="No strong Ideaverse/knowledge-base signal detected.",
        allowed_graphs=["existing_router"],
        blocked_graphs=["obsidian_read_graph"],
        is_read_only=True,
        write_allowed=False,
    )


def build_human_clarification_prompt(decision: KnowledgeGraphDecision) -> str:
    """Generate the structured context an LLM should use to ask naturally.

    The runtime decides that human input is needed; the LLM can phrase this
    naturally using the fields here instead of hallucinating a target.
    """
    lines = [
        "The runtime needs clarification before choosing a graph.",
        "",
        f"Why: {decision.reason}",
    ]

    if decision.ambiguity:
        lines.append("")
        lines.append("Ambiguity:")
        lines.extend(f"- {item}" for item in decision.ambiguity)

    options = decision.clarifying_context.get("options")
    if options:
        lines.append("")
        lines.append("Possible interpretations:")
        lines.extend(f"- {item}" for item in options)

    lines.append("")
    lines.append("Ask the user a concise, natural clarification question before executing any graph.")
    return "\n".join(lines)
