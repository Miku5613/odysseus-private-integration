
from __future__ import annotations

import os
import re
from typing import Any

import httpx


DEFAULT_PREFACE_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_NATIVE_PREFACE_URL = "http://localhost:11434/api/chat"

DEFAULT_PREFACE_MODEL_CANDIDATES = [
    "qwen3:8b",
    "batiai/qwen3.5-9b:q6",
    "qwen3.5:9b",
]


def _strip_thinking(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I).strip()
    s = re.sub(r"```.*?```", "", s, flags=re.S).strip()
    s = s.replace("\r\n", "\n").strip()
    return s


def _one_sentence(text: str, *, max_chars: int = 240) -> str:
    s = _strip_thinking(text)
    s = re.sub(r"\s+", " ", s).strip(" \n\r\t\"'")
    if not s:
        return ""

    m = re.search(r"(.+?[.!?])(?:\s|$)", s)
    if m:
        s = m.group(1).strip()

    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-") + "."

    return s


def _preface_is_acceptable(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return False

    # Block prefaces that answer before retrieval or falsely claim missing access/context.
    banned_fragments = [
        "you didn't write",
        "you did not write",
        "as this is the start",
        "start of our conversation",
        "i don't have access",
        "i do not have access",
        "i can't access",
        "i cannot access",
        "i'm afraid",
        "there are no notes",
        "no notes",
        "nothing about",
        "i found",
        "based on your notes",
    ]

    return not any(fragment in s for fragment in banned_fragments)


def _model_candidates() -> list[str]:
    configured = [
        os.getenv("KNOWLEDGE_PREFACE_MODEL", "").strip(),
        os.getenv("ODYSSEUS_PREFACE_MODEL", "").strip(),
        os.getenv("LLM_MODEL", "").strip(),
    ]

    out: list[str] = []
    for model in configured + DEFAULT_PREFACE_MODEL_CANDIDATES:
        if model and model not in out:
            out.append(model)

    return out


def build_preface_prompt(
    *,
    user_request: str,
    intent: str,
    resource: str,
    next_action: str,
    safety: str,
) -> list[dict[str, str]]:
    system = (
        "You are the conversational judgment layer for Odysseus, a local AI workstation. "
        "Your job is to write one short natural acknowledgement before runtime/tools execute. "
        "Do not answer the user's actual question yet. Do not claim whether notes exist. "
        "Do not choose tools. Do not mention internal node names. Do not reveal hidden reasoning. "
        "Do not use markdown. Output exactly one sentence."
    )

    user = f"""
User request:
{user_request}

Structured system tags:
- intent: {intent}
- target_resource: {resource}
- next_action: {next_action}
- safety_mode: {safety}

Write one brief, natural sentence that tells the user how you are interpreting the job and what you are about to check or do.
Do not answer the user's question. Do not say the user did or did not write something. Do not apologize for missing context.
Avoid repetitive template wording like "This looks like..." unless it genuinely fits.
Keep it under 35 words.
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _try_native_ollama_preface(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float,
) -> tuple[str, str]:
    """Try Ollama native /api/chat with thinking disabled."""

    url = os.getenv("KNOWLEDGE_PREFACE_NATIVE_URL", DEFAULT_NATIVE_PREFACE_URL).strip() or DEFAULT_NATIVE_PREFACE_URL

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": float(os.getenv("KNOWLEDGE_PREFACE_TEMPERATURE", "0.55")),
            "num_predict": int(os.getenv("KNOWLEDGE_PREFACE_MAX_TOKENS", "180")),
        },
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(connect=1.5, read=timeout_s, write=2.0, pool=1.0)) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message") or {}
        content = msg.get("content") or ""
        text = _one_sentence(content)

        if text and _preface_is_acceptable(text):
            return text, ""

        return "", "native empty or unacceptable preface"
    except Exception as e:
        return "", f"native {type(e).__name__}: {e}"


def _try_openai_compat_preface(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float,
) -> tuple[str, str]:
    """Fallback path for OpenAI-compatible Ollama endpoint."""

    url = os.getenv("KNOWLEDGE_PREFACE_URL", DEFAULT_PREFACE_URL).strip() or DEFAULT_PREFACE_URL

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("KNOWLEDGE_PREFACE_TEMPERATURE", "0.55")),
        "max_tokens": int(os.getenv("KNOWLEDGE_PREFACE_MAX_TOKENS", "180")),
        "stream": False,
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(connect=1.5, read=timeout_s, write=2.0, pool=1.0)) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = ""
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content") or choices[0].get("text") or ""

        text = _one_sentence(content)

        if text and _preface_is_acceptable(text):
            return text, ""

        return "", "openai-compatible empty or unacceptable preface"
    except Exception as e:
        return "", f"openai-compatible {type(e).__name__}: {e}"


def generate_intent_preface(
    *,
    user_request: str,
    intent: str,
    resource: str,
    next_action: str,
    safety: str = "read-only",
    timeout_s: float = 8.0,
) -> dict[str, Any]:
    """Return a short LLM-authored preface, with honest technical fallback.

    The LLM only writes the acknowledgement. It does not choose tools or execute anything.
    """

    if os.getenv("ODYSSEUS_ENABLE_LLM_PREFACE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return {
            "text": "The LLM preface step is disabled, so I’m skipping the preface and starting the job.",
            "source": "technical_fallback_disabled",
            "model": "",
            "ok": False,
        }

    messages = build_preface_prompt(
        user_request=user_request,
        intent=intent,
        resource=resource,
        next_action=next_action,
        safety=safety,
    )

    prefer_native = os.getenv("KNOWLEDGE_PREFACE_USE_NATIVE_OLLAMA", "1").strip().lower() not in {"0", "false", "no", "off"}
    last_error = ""

    for model in _model_candidates():
        if prefer_native:
            text, err = _try_native_ollama_preface(
                model=model,
                messages=messages,
                timeout_s=timeout_s,
            )
            if text:
                return {
                    "text": text,
                    "source": "llm_native_ollama",
                    "model": model,
                    "ok": True,
                }
            last_error = err

        text, err = _try_openai_compat_preface(
            model=model,
            messages=messages,
            timeout_s=timeout_s,
        )
        if text:
            return {
                "text": text,
                "source": "llm_openai_compat",
                "model": model,
                "ok": True,
            }
        last_error = err

    return {
        "text": "The LLM preface step failed or took too long, so I’m skipping the preface and starting the job.",
        "source": "technical_fallback_error",
        "model": "",
        "ok": False,
        "error": last_error,
    }


def is_simple_conversation_fast_path(message: str) -> bool:
    text = str(message or "").strip().lower()
    text = re.sub(r"[.!?\s]+$", "", text)

    if not text:
        return True

    simple = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
        "nice",
        "great",
        "good morning",
        "good evening",
        "good night",
    }

    return text in simple
