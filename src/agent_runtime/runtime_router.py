from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal


RuntimeIntent = Literal[
    "exact_lookup",
    "file_lookup",
    "architecture_summary",
    "general_readonly",
    "write_mutation",
    "legacy_exploratory",
]

TargetRuntime = Literal[
    "contract",
    "general",
    "legacy",
]


@dataclass(frozen=True)
class RuntimeDecision:
    intent: RuntimeIntent
    target_runtime: TargetRuntime
    reason: str
    is_safe_readonly: bool

    @property
    def safety_profile(self) -> str:
        if self.is_safe_readonly and self.target_runtime in {"contract", "general"}:
            return "read_only"
        if self.intent == "write_mutation":
            return "write_blocked_or_legacy"
        if self.target_runtime == "legacy":
            return "legacy_unverified"
        return "unknown"

    @property
    def runtime_family(self) -> str:
        if self.target_runtime == "contract":
            return "deterministic_contract"
        if self.target_runtime == "general":
            return "deterministic_general"
        return "legacy_agent_loop"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["safety_profile"] = self.safety_profile
        data["runtime_family"] = self.runtime_family
        return data

    def to_status(self) -> dict:
        return {
            "intent": self.intent,
            "runtime": self.target_runtime,
            "runtime_family": self.runtime_family,
            "safe_readonly": self.is_safe_readonly,
            "safety_profile": self.safety_profile,
            "reason": self.reason,
        }

    def to_trace_lines(self) -> list[str]:
        return [
            "Runtime decision:",
            f"- intent: {self.intent}",
            f"- runtime: {self.target_runtime}",
            f"- runtime_family: {self.runtime_family}",
            f"- safety_profile: {self.safety_profile}",
            f"- safe_readonly: {self.is_safe_readonly}",
            f"- reason: {self.reason}",
        ]

    def to_trace_text(self) -> str:
        return "\n".join(self.to_trace_lines())


WRITE_PATTERNS = [
    r"\bpatch\b",
    r"\bwrite\b",
    r"\bedit\b",
    r"\bcreate\b",
    r"\bdelete\b",
    r"\bremove\b",
    r"\bmodify\b",
    r"\bupdate\b",
    r"\bchange\b",
    r"\brename\b",
    r"\bmove\b",
    r"\binstall\b",
    r"\brun\b",
    r"\bexecute\b",
    r"\bfix\b",
]

RESEARCH_OR_BROWSER_PATTERNS = [
    r"\bresearch\b",
    r"\blatest\b",
    r"\bnews\b",
    r"\bbrowser\b",
    r"\bweb\b",
    r"\bgoogle\b",
    r"\bsearch online\b",
    r"\blook up online\b",
]

FILE_LOOKUP_PATTERNS = [
    r"\bfind\s+files\b",
    r"\bshow\s+files\b",
    r"\blist\s+files\b",
    r"\brelated\s+files\b",
    r"\bfiles\s+related\b",
    r"\blist\s+modules\b",
    r"\bshow\s+modules\b",
    r"\bmodules\s+related\b",
    r"\bproject\s+structure\b",
    r"\bstructure\s+of\b",
    r"\binspect\s+the\s+structure\b",
    r"\bfiles\s+involved\b",
    r"\bcode\s+involved\b",
]

EXACT_LOOKUP_PATTERNS = [
    r"\bwhere\s+is\s+/.+?\s+defined\b",
    r"\bwhere\s+is\s+.+?\s+defined\b",
    r"\bwhere\s+does\s+.+?\s+(emit|handle|start|call|save|load|parse|render)\b",
    r"\bwhere\s+.+?\s+(emits|handles|starts|calls|saves|loads|parses|renders)\b",
    r"\bfind\s+where\s+.+?\s+(emits|handles|starts|calls|saves|loads|parses|renders)\b",
    r"\bfind\s+the\s+callsite\b",
    r"\bcallsite\b",
    r"\bhandler\b",
    r"\bemits?\b",
    r"\bhandles?\b",
    r"\bline\s+\d+\b",
    r"/api/[A-Za-z0-9_\-/{}]+",
]

EXACT_EVENT_TERMS = [
    "tool_output",
    "agent_step",
    "message_saved",
    "doc_stream_open",
    "doc_stream_delta",
    "stream_agent_loop",
    "chat_stream",
    "save_assistant_response",
]


def _has_regex(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _has_any_term(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in terms)


def _is_architecture_summary_request(text: str) -> bool:
    low = text.lower()

    architecture_terms = [
        "summarize how",
        "explain how",
        "how is",
        "how are",
        "how does",
        "how do",
        "architecture",
        "wired",
        "wiring",
        "flow",
        "overview",
        "map",
    ]

    topic_terms = [
        "memory",
        "rag",
        "mcp",
        "email",
        "calendar",
        "image generation",
        "voice",
        "stt",
        "tts",
        "chat streaming",
    ]

    return any(term in low for term in architecture_terms) and any(term in low for term in topic_terms)


def route_agent_runtime(prompt: str) -> RuntimeDecision:
    raw = prompt or ""
    low = raw.lower().strip()

    if not low:
        return RuntimeDecision(
            intent="legacy_exploratory",
            target_runtime="legacy",
            reason="Empty prompt; falling back to legacy agent loop.",
            is_safe_readonly=False,
        )

    # Safety first. Any mutation hint prevents contract/general routing.
    if _has_regex(low, WRITE_PATTERNS):
        return RuntimeDecision(
            intent="write_mutation",
            target_runtime="legacy",
            reason="Mutation keyword detected; routing away from read-only runtimes.",
            is_safe_readonly=False,
        )

    # Research/browser/current-world tasks remain legacy until dedicated graphs exist.
    if _has_regex(low, RESEARCH_OR_BROWSER_PATTERNS):
        return RuntimeDecision(
            intent="legacy_exploratory",
            target_runtime="legacy",
            reason="Research/browser/current-world intent detected; falling back to legacy.",
            is_safe_readonly=False,
        )

    # Architecture summaries are read-only project inspection and belong to general runtime.
    if _is_architecture_summary_request(low):
        return RuntimeDecision(
            intent="architecture_summary",
            target_runtime="general",
            reason="Read-only architecture summary request; routing to general runtime.",
            is_safe_readonly=True,
        )

    # Broad structural discovery belongs to general runtime.
    # This must happen before exact lookup so "files involved in X" does not get swallowed.
    if _has_regex(low, FILE_LOOKUP_PATTERNS):
        return RuntimeDecision(
            intent="file_lookup",
            target_runtime="general",
            reason="Structural file/module discovery request; routing to general runtime.",
            is_safe_readonly=True,
        )

    # Exact source lookup belongs to contract runtime.
    if _has_regex(low, EXACT_LOOKUP_PATTERNS) or _has_any_term(low, EXACT_EVENT_TERMS):
        return RuntimeDecision(
            intent="exact_lookup",
            target_runtime="contract",
            reason="Precise source/event/route lookup request; routing to contract runtime.",
            is_safe_readonly=True,
        )

    # Read-only but broad/non-specific project inspection can go to general.
    if (
        (
            "inspect" in low
            or "trace" in low
            or "show" in low
            or "find" in low
            or "locate" in low
            or "report" in low
            or "list" in low
        )
        and (
            "code" in low
            or "route" in low
            or "frontend" in low
            or "backend" in low
            or "module" in low
            or "project" in low
            or "runtime" in low
            or "worker" in low
        )
    ):
        return RuntimeDecision(
            intent="general_readonly",
            target_runtime="general",
            reason="Broad read-only code inspection request; routing to general runtime.",
            is_safe_readonly=True,
        )

    return RuntimeDecision(
        intent="legacy_exploratory",
        target_runtime="legacy",
        reason="Ambiguous or unsupported prompt; falling back to legacy agent loop.",
        is_safe_readonly=False,
    )


def explain_runtime_route(prompt: str) -> str:
    """Return a stable human-readable runtime routing trace for diagnostics."""
    return route_agent_runtime(prompt).to_trace_text()


__all__ = ["RuntimeDecision", "route_agent_runtime", "explain_runtime_route"]
