
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .general_state import GeneralObservation, GeneralState, GeneralTask
from . import general_tools as tools


WRITE_WORDS = {
    "write", "edit", "create", "delete", "remove", "modify", "patch",
    "apply", "rename", "move", "install", "run", "execute", "fix",
    "change", "update", "send", "schedule"
}

WRITE_PHRASES = {
    "send email",
    "send an email",
    "draft email",
    "draft an email",
    "create calendar",
    "create event",
    "schedule meeting",
    "schedule a meeting",
    "add calendar",
    "update calendar",
}

READ_HINTS = {
    "find",
    "where",
    "locate",
    "search",
    "show",
    "inspect",
    "trace",
    "identify",
    "which file",
    "what file",
    "files related",
    "related files",
    "related to",
    "list files",
    "show files",
}

CODE_HINTS = {
    "frontend",
    "backend",
    "route",
    "routes",
    "handler",
    "event",
    "stream",
    "streaming",
    "chat streaming",
    "streaming files",
    "files",
    "folder",
    "directory",
    "function",
    "call",
    "callsite",
    ".py",
    ".js",
    "/api",
    "chat_routes",
    "agent_loop",
    "tool_output",
    "message_saved",
}


def _contains_any(text: str, words: set[str]) -> bool:
    low = text.lower()
    return any(w in low for w in words)



def _has_write_intent(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in WRITE_PHRASES):
        return True
    return any(re.search(rf"\b{re.escape(word)}\b", low) for word in WRITE_WORDS)


def infer_file_lookup_topic(query: str) -> str:
    low = query.lower()

    topic_checks = [
        ("chat streaming", ["chat streaming", "chat stream"]),
        ("memory", ["memory", "memories"]),
        ("rag", ["rag", "document rag", "retrieval"]),
        ("mcp", ["mcp", "mcp tools", "mcp server", "tool server"]),
        ("email polling", ["email polling", "email poller", "email pollers"]),
        ("email", ["email", "imap", "smtp", "inbox"]),
        ("calendar", ["calendar", "events", "reminder", "rrule"]),
        ("image generation", ["image generation", "generated image", "image gen", "image models"]),
        ("voice", ["voice", "stt", "tts", "speech", "transcription"]),
    ]

    for topic, terms in topic_checks:
        if any(term in low for term in terms):
            return topic

    # Try to extract the phrase after "related to".
    m = re.search(r"related to\s+([A-Za-z0-9_\- /]+)", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(" .?").lower()

    return "the request"


def build_file_lookup_globs(query: str) -> list[str]:
    topic = infer_file_lookup_topic(query)

    topic_globs = {
        "chat streaming": [
            "routes/chat_routes.py",
            "static/js/chat.js",
            "static/js/chatStream.js",
            "static/js/streamingRenderer.js",
            "static/js/streamingSegmenter.js",
            "**/*chat*stream*.py",
            "**/*chat*stream*.js",
            "**/*stream*chat*.py",
            "**/*stream*chat*.js",
            "**/*streaming*.py",
            "**/*streaming*.js",
        ],
        "memory": [
            "services/memory/*.py",
            "src/memory*.py",
            "src/memory_*.py",
            "routes/memory_routes.py",
            "mcp_servers/memory_server.py",
            "static/js/memory.js",
            "tests/test_memory*.py",
            "tests/test_*memory*.py",
        ],
        "rag": [
            "src/rag*.py",
            "src/rag_*.py",
            "mcp_servers/rag_server.py",
            "static/js/rag.js",
            "tests/test_rag*.py",
            "tests/test_*rag*.py",
        ],
        "mcp": [
            "src/*mcp*.py",
            "routes/mcp_routes.py",
            "mcp_servers/*.py",
            "tests/test_mcp*.py",
            "tests/test_*mcp*.py",
            "tests/cli/test_mcp*.py",
        ],
        "email polling": [
            "routes/email_pollers.py",
            "routes/email_helpers.py",
            "routes/email_routes.py",
            "mcp_servers/email_server.py",
            "src/email_thread_parser.py",
            "static/js/email*.js",
            "static/js/emailLibrary/*.js",
            "scripts/demo_email/*.py",
            "tests/test_email*.py",
            "tests/test_*email*.py",
        ],
        "email": [
            "routes/email*.py",
            "mcp_servers/email_server.py",
            "src/email*.py",
            "static/js/email*.js",
            "static/js/emailLibrary/*.js",
            "scripts/demo_email/*.py",
            "tests/test_email*.py",
            "tests/test_*email*.py",
        ],
        "calendar": [
            "routes/calendar_routes.py",
            "src/tools/calendar.py",
            "static/js/calendar.js",
            "static/js/calendar/*.js",
            "tests/test_calendar*.py",
            "tests/test_*calendar*.py",
            "tests/helpers/calendar_routes.py",
            "tests/cli/test_calendar*.py",
        ],
        "image generation": [
            "mcp_servers/image_gen_server.py",
            "src/generated_images.py",
            "src/tools/image.py",
            "services/hwfit/image_models.py",
            "tests/test_*image*.py",
            "tests/test_generated_image*.py",
            "tests/test_gallery_image*.py",
        ],
        "voice": [
            "routes/stt_routes.py",
            "routes/tts_routes.py",
            "services/stt/*.py",
            "services/tts/*.py",
            "static/js/voiceRecorder.js",
            "static/js/tts-ai.js",
            "tests/test_stt*.py",
            "tests/test_tts*.py",
            "tests/test_*stt*.py",
            "tests/test_*tts*.py",
        ],
    }

    if topic in topic_globs:
        return topic_globs[topic]

    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query)
        if w.lower() not in {
            "find", "files", "file", "related", "show", "list",
            "modules", "module", "code", "involved", "inspect", "the", "to",
        }
    ]

    if words:
        base = words[0]
        return [
            f"**/*{base}*.py",
            f"**/*{base}*.js",
            f"tests/test_*{base}*.py",
        ]

    return ["**/*.py", "**/*.js"]

def plan_task(query: str) -> GeneralTask:
    low = query.lower()

    if _has_write_intent(low):
        return GeneralTask(
            raw_query=query,
            kind="unsupported",
            reason="Request appears to involve write/edit/execute behavior.",
            is_readonly=False,
            should_fallback=True,
        )

    asks_for_files = (
        "file" in low
        or "files" in low
        or "list files" in low
        or "show files" in low
        or "related files" in low
        or "files related" in low
    )

    if is_architecture_summary_request(query):
        kind = "architecture_summary"
    elif asks_for_files:
        kind = "file_lookup"
    elif "/api/" in low or "route" in low:
        kind = "route_lookup"
    elif "event" in low or "tool_output" in low or "message_saved" in low or "doc_stream" in low:
        kind = "event_lookup"
    elif _contains_any(low, CODE_HINTS):
        kind = "code_lookup"
    elif _contains_any(low, READ_HINTS):
        kind = "general_readonly"
    else:
        return GeneralTask(
            raw_query=query,
            kind="unsupported",
            reason="No safe read-only tool task recognized.",
            is_readonly=True,
            should_fallback=True,
        )

    target = extract_target(query)

    return GeneralTask(
        raw_query=query,
        kind=kind,
        target=target,
        reason="Safe read-only general runtime task.",
        is_readonly=True,
        should_fallback=False,
    )


def extract_target(query: str) -> str:
    # Prefer exact event/API/code-ish tokens.
    candidates = re.findall(r"[/A-Za-z0-9_.:-]+", query)
    ranked = []
    for c in candidates:
        low = c.lower().strip("`'\".,")
        if not low:
            continue
        score = 0
        if "/" in low or "." in low or "_" in low:
            score += 3
        if low in {"tool_output", "message_saved", "doc_stream_delta", "doc_stream_open"}:
            score += 5
        if low.startswith("/api/"):
            score += 5
        if low in {"frontend", "backend", "handler", "route", "stream"}:
            score -= 2
        if score > 0:
            ranked.append((score, low))
    if ranked:
        ranked.sort(reverse=True)
        return ranked[0][1]
    return ""


def _record(state: GeneralState, tool: str, args: dict, output: str, success: bool = True) -> None:
    state.observations.append(GeneralObservation(tool=tool, args=args, output=output, success=success))



def is_architecture_summary_request(query: str) -> bool:
    low = query.lower()
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
    return any(t in low for t in architecture_terms) and any(t in low for t in topic_terms)


def rank_architecture_files(query: str, files: list[str]) -> list[str]:
    topic = infer_file_lookup_topic(query).lower()

    def score(path: str) -> int:
        low = path.lower().replace("\\", "/")
        score = 0

        if low.startswith("src/"):
            score += 20
        if low.startswith("services/"):
            score += 18
        if low.startswith("routes/"):
            score += 16
        if low.startswith("mcp_servers/"):
            score += 14
        if low.startswith("static/js/"):
            score += 10
        if low.startswith("tests/"):
            score -= 30
        if low.endswith("/__init__.py"):
            score -= 8

        if topic == "memory":
            preferred = [
                "routes/memory_routes.py",
                "services/memory/service.py",
                "services/memory/memory.py",
                "services/memory/memory_extractor.py",
                "src/memory.py",
                "src/memory_provider.py",
                "src/memory_vector.py",
                "mcp_servers/memory_server.py",
                "static/js/memory.js",
            ]
        elif topic == "rag":
            preferred = [
                "src/rag_manager.py",
                "src/rag_vector.py",
                "src/rag_singleton.py",
                "mcp_servers/rag_server.py",
                "static/js/rag.js",
            ]
        elif topic == "mcp":
            preferred = [
                "src/mcp_manager.py",
                "src/builtin_mcp.py",
                "src/mcp_oauth.py",
                "routes/mcp_routes.py",
                "mcp_servers/memory_server.py",
                "mcp_servers/rag_server.py",
                "mcp_servers/email_server.py",
                "mcp_servers/image_gen_server.py",
            ]
        elif topic in {"email", "email polling"}:
            preferred = [
                "routes/email_pollers.py",
                "routes/email_helpers.py",
                "routes/email_routes.py",
                "mcp_servers/email_server.py",
                "src/email_thread_parser.py",
                "static/js/emailInbox.js",
                "static/js/emailLibrary.js",
            ]
        elif topic == "calendar":
            preferred = [
                "routes/calendar_routes.py",
                "src/tools/calendar.py",
                "static/js/calendar.js",
                "static/js/calendar/utils.js",
            ]
        elif topic == "image generation":
            preferred = [
                "mcp_servers/image_gen_server.py",
                "src/generated_images.py",
                "src/tools/image.py",
                "services/hwfit/image_models.py",
            ]
        elif topic == "voice":
            preferred = [
                "routes/stt_routes.py",
                "routes/tts_routes.py",
                "services/stt/stt_service.py",
                "services/tts/tts_service.py",
                "static/js/voiceRecorder.js",
                "static/js/tts-ai.js",
            ]
        else:
            preferred = []

        for index, item in enumerate(preferred):
            if item in low:
                score += 100 - index

        return score

    return sorted(dict.fromkeys(files), key=score, reverse=True)


def summarize_architecture_from_files(state: GeneralState, files: list[str]) -> str:
    topic = infer_file_lookup_topic(state.raw_query)
    ranked = rank_architecture_files(state.raw_query, files)

    if not ranked:
        return f"I could not find reliable files to summarize the {topic} architecture."

    groups = {
        "Routes / API surface": [],
        "Service layer": [],
        "Core backend modules": [],
        "MCP/tool bridge": [],
        "Frontend/UI": [],
        "Tests": [],
        "Other": [],
    }

    for path in ranked[:16]:
        low = path.lower().replace("\\", "/")
        if low.startswith("routes/"):
            groups["Routes / API surface"].append(path)
        elif low.startswith("services/"):
            groups["Service layer"].append(path)
        elif low.startswith("src/"):
            groups["Core backend modules"].append(path)
        elif low.startswith("mcp_servers/"):
            groups["MCP/tool bridge"].append(path)
        elif low.startswith("static/js/"):
            groups["Frontend/UI"].append(path)
        elif low.startswith("tests/"):
            groups["Tests"].append(path)
        else:
            groups["Other"].append(path)

    lines = [f"Read-only architecture map for {topic}:"]

    for heading, paths in groups.items():
        if not paths or heading == "Tests":
            continue
        lines.append("")
        lines.append(f"{heading}:")
        for p in paths[:5]:
            lines.append(f"- `{p}`")

    flow = []
    if groups["Frontend/UI"]:
        flow.append("Frontend/UI")
    if groups["Routes / API surface"]:
        flow.append("Routes/API")
    if groups["Service layer"]:
        flow.append("Service layer")
    if groups["Core backend modules"]:
        flow.append("Core backend modules")
    if groups["MCP/tool bridge"]:
        flow.append("MCP/tool bridge")

    lines.append("")
    lines.append("Likely flow:")
    lines.append("- " + " → ".join(flow) if flow else "- Layer flow is unclear from filenames alone.")

    lines.append("")
    lines.append("Files read for evidence:")
    for p in ranked[:4]:
        lines.append(f"- `{p}`")

    lines.append("")
    lines.append("Evidence: topic-aware glob discovery plus read-only file reads of the top ranked files.")
    return "\n".join(lines)


def execute_readonly_plan(state: GeneralState) -> None:
    assert state.task is not None
    task = state.task
    target = task.target or ""

    # Tool 1: workspace
    out = tools.get_workspace(state.workspace)
    _record(state, "get_workspace", {}, out)

    # File discovery tasks should list likely files, not grep arbitrary content.
    if task.kind == "file_lookup":
        glob_patterns = build_file_lookup_globs(state.raw_query)

        seen = []
        for pat in glob_patterns:
            out = tools.glob_files(state.workspace, pat, max_results=40)
            _record(state, "glob_files", {"pattern": pat, "max_results": 40}, out, bool(out.strip()))
            for line in out.splitlines():
                clean = line.strip().replace("\\", "/")
                if clean and clean not in seen:
                    seen.append(clean)

        state.final_answer = finalize_file_lookup(state, seen)
        return

    if task.kind == "architecture_summary":
        glob_patterns = build_file_lookup_globs(state.raw_query)

        seen = []
        for pat in glob_patterns:
            out = tools.glob_files(state.workspace, pat, max_results=40)
            _record(state, "glob_files", {"pattern": pat, "max_results": 40}, out, bool(out.strip()))
            for line in out.splitlines():
                clean = line.strip().replace("\\", "/")
                if clean and clean not in seen:
                    seen.append(clean)

        ranked = rank_architecture_files(state.raw_query, seen)

        for path in ranked[:4]:
            try:
                out = tools.read_file(state.workspace, path, max_chars=3000)
            except TypeError:
                out = tools.read_file(state.workspace, path)
            _record(state, "read_file", {"path": path, "max_chars": 3000}, out, bool(str(out).strip()))

        state.final_answer = summarize_architecture_from_files(state, ranked)
        return

    # Tool 2: targeted grep
    patterns = []

    if target:
        patterns.append(re.escape(target))

    raw = state.raw_query.lower()

    if "chat_stream" in raw or "/api/chat_stream" in raw:
        patterns.append(r"/api/chat_stream|chat_stream")
    if "tool_output" in raw:
        patterns.append(r"tool_output")
    if "message_saved" in raw:
        patterns.append(r"message_saved")
    if "doc_stream" in raw:
        patterns.append(r"doc_stream")
    if not patterns:
        # fallback keyword extraction
        words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", state.raw_query) if w.lower() not in {"find", "where", "the", "handles", "events", "use", "tools", "needed", "guess"}]
        patterns.extend(re.escape(w) for w in words[:3])

    grep_pattern = "|".join(dict.fromkeys(patterns)) or re.escape(state.raw_query[:40])
    grep_out = tools.grep(state.workspace, grep_pattern, max_results=80)
    _record(state, "grep", {"pattern": grep_pattern, "max_results": 80}, grep_out, bool(grep_out.strip()))

    best = choose_best_match(state, grep_out)
    if best:
        file, line = best
        around = tools.read_file_around(state.workspace, file, line, radius=16)
        _record(state, "read_file_around", {"path": file, "line": line, "radius": 16}, around, bool(around.strip()))

    state.final_answer = finalize(state)


def choose_best_match(state: GeneralState, grep_out: str) -> tuple[str, int] | None:
    rows = []
    target = (state.task.target if state.task else "") or ""
    raw = state.raw_query.lower()

    for line in grep_out.splitlines():
        m = re.match(r"(.+?):(\d+):\s*(.*)", line)
        if not m:
            continue
        file, lineno, content = m.group(1), int(m.group(2)), m.group(3)
        low_file = file.lower().replace("\\", "/")
        low_content = content.lower()
        score = 0

        if target and target.lower() in low_content:
            score += 20
        # Query-directed file authority.
        # Frontend questions should prefer real UI handlers.
        # Backend questions should prefer routes/agent loop emitters.
        wants_frontend = "frontend" in raw or "client" in raw or "ui" in raw
        wants_backend = "backend" in raw or "server" in raw or "emit" in raw or "emits" in raw

        if "static/js/chat.js" in low_file:
            score += 18 if wants_frontend else 10
        if "static/js/chatstream.js" in low_file:
            score += 10 if wants_frontend else 4
        if "routes/chat_routes.py" in low_file:
            score += 14 if wants_backend else 8
        if "src/agent_loop.py" in low_file:
            score += 12 if wants_backend else 3

        if wants_frontend and ("src/agent_loop.py" in low_file or "routes/chat_routes.py" in low_file):
            score -= 10
        if wants_backend and low_file.startswith("static/js/"):
            score -= 8

        if "compare/" in low_file:
            score -= 8
        if "patch_" in low_file or "inspect_" in low_file:
            score -= 20

        if "tool_output" in raw and "tool_output" in low_content:
            score += 10
        if "message_saved" in raw and "message_saved" in low_content:
            score += 10
        if "chat_stream" in raw and ("chat_stream" in low_content or "/api/chat_stream" in low_content):
            score += 10
        if "else if" in low_content or "@router" in low_content or "yield" in low_content:
            score += 3

        rows.append((score, file, lineno, content))

    if not rows:
        return None

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[0][1], rows[0][2]


def finalize_file_lookup(state: GeneralState, files: list[str]) -> str:
    if not files:
        return "I could not find reliable file matches for that request."

    # Prefer real source files over artifacts and tests.
    def score(path: str) -> int:
        low = path.lower().replace("\\", "/")
        s = 0
        if low == "routes/chat_routes.py":
            s += 100
        if low == "static/js/chat.js":
            s += 90
        if low == "static/js/chatstream.js":
            s += 85
        if "streamingrenderer" in low or "streamingsegmenter" in low:
            s += 70
        if low.startswith("static/js/"):
            s += 20
        if low.startswith("routes/"):
            s += 20
        if low.endswith(".py") or low.endswith(".js"):
            s += 10
        if "test" in low or "spec" in low:
            s -= 20
        if "before_" in low or "backup" in low or "inspect" in low:
            s -= 50
        return s

    ranked = sorted(files, key=score, reverse=True)
    top = ranked[:12]

    topic = infer_file_lookup_topic(state.raw_query)
    lines = [f"Likely files related to {topic}:"]
    for f in top:
        lines.append(f"- `{f}`")

    return "\n".join(lines) + "\n\nEvidence: file discovery via read-only glob search."


def build_failure_answer(state: GeneralState) -> str:
    task = state.task
    reason = task.reason if task else "No task plan was created."
    raw_query = state.raw_query or ""

    lines = [
        "I could not handle this with the deterministic general runtime.",
        "",
        f"Reason: {reason}",
        "",
        "This runtime is currently limited to safe read-only Odysseus codebase tasks, such as:",
        "- finding files related to a subsystem",
        "- locating where a frontend/backend event is handled",
        "- summarizing how a known Odysseus subsystem is wired",
        "",
    ]

    low = raw_query.lower()
    if any(word in low for word in ("weather", "current", "latest", "web", "browser", "internet")):
        lines.extend([
            "This looks like a current-world, browser, or web request, so it should go through the legacy/research/browser path instead.",
            "",
            "Safer rewrite for this runtime:",
            "\"Find files related to web search or browser tools in Odysseus.\"",
        ])
    elif task and not task.is_readonly:
        lines.extend([
            "This looks like a write/edit/execute request, so deterministic read-only mode refused it.",
            "",
            "Safer rewrite for this runtime:",
            "\"Find where this behavior is implemented before changing it.\"",
        ])
    else:
        lines.extend([
            "The request was too broad or ambiguous for the deterministic read-only planner.",
            "",
            "Safer rewrite for this runtime:",
            "\"Find files related to memory in Odysseus.\"",
            "or",
            "\"Summarize how memory is wired in Odysseus.\"",
        ])

    lines.extend([
        "",
        "No files were read and no changes were made.",
    ])
    return "\n".join(lines)


def finalize(state: GeneralState) -> str:
    if not state.task:
        return "I could not create a task plan."

    if state.task.should_fallback:
        return build_failure_answer(state)

    around_obs = next((o for o in reversed(state.observations) if o.tool == "read_file_around" and o.success), None)
    grep_obs = next((o for o in reversed(state.observations) if o.tool == "grep"), None)

    if around_obs:
        lines = around_obs.output.splitlines()
        center = ""
        target = state.task.target.lower()
        for l in lines:
            if target and target in l.lower():
                center = l
                break
        if not center and lines:
            center = lines[min(len(lines) // 2, len(lines)-1)]

        m = re.match(r"(.+?):(\d+):\s*(.*)", center)
        if m:
            file, line_no, code = m.group(1), m.group(2), m.group(3)
            target_label = state.task.target or "the requested item"
            return (
                f"The relevant location is `{file}` around line {line_no}, for `{target_label}`.\n\n"
                f"Evidence:\n"
                f"- `>> {line_no}: {code.strip()}`"
            )

    if grep_obs and grep_obs.output.strip():
        first = grep_obs.output.splitlines()[0]
        return f"I found a likely match:\n\n- `{first}`"

    return "I could not find reliable evidence for that request."


def run_general_runtime(query: str, workspace: str) -> GeneralState:
    start = time.time()
    state = GeneralState(raw_query=query, workspace=workspace)
    state.task = plan_task(query)

    if state.task.should_fallback:
        state.final_answer = build_failure_answer(state)
    else:
        execute_readonly_plan(state)

    state.metrics = {
        "runtime": "v0.13-general-langgraph-scaffold",
        "model": "deterministic-general-scaffold",
        "elapsed_s": round(time.time() - start, 3),
        "tool_calls": len(state.observations),
        "fallback_required": bool(state.task.should_fallback if state.task else True),
        "task_kind": state.task.kind if state.task else "unknown",
    }
    return state


def state_to_legacy_frames(state: GeneralState):
    """Convert general runtime state into legacy/UI-compatible SSE frames.

    Important: the existing frontend expects an agent lifecycle shape similar to
    the legacy/contract path. In particular, agent_step frames help initialize
    the visible agent thread state before tool events and before final text.
    """
    task = state.task

    yield {
        "type": "agent_prep",
        "data": {
            "runtime": "v0.13-general-langgraph-scaffold",
            "model": "deterministic-general-scaffold",
            "scope": "general_readonly",
            "workspace": state.workspace,
        },
    }

    if task is not None:
        yield {
            "type": "task_contract",
            "data": {
                "runtime": "v0.13-general-langgraph-scaffold",
                "task": {
                    "kind": task.kind,
                    "target": task.target,
                    "is_readonly": task.is_readonly,
                    "should_fallback": task.should_fallback,
                    "reason": task.reason,
                },
            },
        }

        yield {
            "type": "agent_step",
            "data": {
                "label": "Executing general read-only task",
                "runtime": "v0.13-general-langgraph-scaffold",
                "kind": task.kind,
                "target": task.target,
                "tool_rounds": 0,
            },
        }

    for obs in state.observations:
        yield {
            "type": "tool_start",
            "tool": obs.tool,
            "command": obs.args,
            "call_id": f"general_{len(obs.tool)}_{abs(hash(str(obs.args))) % 100000}",
        }
        yield {
            "type": "tool_output",
            "tool": obs.tool,
            "output": obs.output[:4000],
            "success": obs.success,
        }

    yield {
        "type": "agent_step",
        "data": {
            "label": "Finalizing",
            "runtime": "v0.13-general-langgraph-scaffold",
            "tool_rounds": len(state.observations),
        },
    }

    if state.final_answer:
        yield {"delta": state.final_answer}

    metrics = dict(getattr(state, "metrics", {}) or {})
    metrics.update({
        "runtime": "v0.13-general-langgraph-scaffold",
        "model": "deterministic-general-scaffold",
        "elapsed_s": metrics.get("elapsed_s", 0),
        "tool_calls": len(state.observations),
        "fallback_required": bool(task.should_fallback) if task else True,
        "task_kind": task.kind if task else "unknown",
        "agent_rounds": 1,
        "agent_tool_calls": len(state.observations),
        "langgraph_runtime": True,
        "general_runtime": True,
    })

    yield {
        "type": "metrics",
        "data": metrics,
    }
