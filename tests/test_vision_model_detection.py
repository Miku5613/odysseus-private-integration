"""Tests for is_vision_model (issue #124).

Local vision models served through Ollama/llama.cpp show up under many
names. If one isn't recognized as vision-capable, the image attachment is
stripped from the request before it reaches the model, so it silently never
sees the picture.
"""
from src import chat_helpers
from src.chat_helpers import is_vision_model, model_supports_vision


def test_recognizes_local_and_hosted_vision_models():
    for name in [
        # the ones #124 missed
        "moondream", "moondream:latest",
        "llama3.2-vision:11b", "granite3.2-vision",
        "qwen2.5-vl:7b", "qwen2.5vl", "internvl2.5", "cogvlm",
        # already worked, keep them working
        "llava", "llava:7b", "bakllava", "minicpm-v",
        "gpt-4o", "claude-sonnet-4", "gemini-2.0-flash", "pixtral-12b",
    ]:
        assert is_vision_model(name), f"{name!r} should be detected as vision-capable"


def test_text_only_models_not_flagged():
    for name in ["qwen2.5:3b", "mistral", "llama3.1:8b", "deepseek-r1", "phi3", "vicuna", ""]:
        assert not is_vision_model(name), f"{name!r} should not be flagged as vision"


def test_none_is_safe():
    assert is_vision_model(None) is False


def test_recognizes_multimodal_families_without_vision_in_name():
    # issue #1274: these are vision-capable but their names don't contain
    # "vision"/"vl", so they were dropped and the model never saw the image.
    for name in [
        "gemma3:4b", "gemma3", "gemma-3-27b-it",
        "llama4:scout", "llama4", "llama-4-maverick",
        "mistral-small3.1", "mistral-small-3.2",
        "phi-4-multimodal", "phi4-multimodal",
    ]:
        assert is_vision_model(name), f"{name!r} should be detected as vision-capable"


def test_new_keywords_do_not_overmatch_text_models():
    # The added families must not flag their text-only siblings.
    for name in ["gemma2:9b", "gemma:7b", "llama3.3", "mistral-small", "phi-3-mini"]:
        assert not is_vision_model(name), f"{name!r} should not be flagged as vision"


class _FakeResponse:
    is_success = True

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_ollama_advertised_vision_capability_overrides_model_name(monkeypatch):
    chat_helpers._ollama_vision_cache.clear()

    def fake_post(url, *, json, timeout):
        assert url == "http://localhost:11434/api/show"
        assert json == {"model": "qwen3.5:9b"}
        return _FakeResponse({
            "capabilities": ["completion", "vision", "tools", "thinking"],
        })

    monkeypatch.setattr(chat_helpers.httpx, "post", fake_post)

    assert model_supports_vision(
        "qwen3.5:9b",
        "http://localhost:11434/v1",
    )


def test_ollama_advertised_text_only_capability_overrides_name_hint(monkeypatch):
    chat_helpers._ollama_vision_cache.clear()

    def fake_post(url, *, json, timeout):
        return _FakeResponse({"capabilities": ["completion"]})

    monkeypatch.setattr(chat_helpers.httpx, "post", fake_post)

    assert not model_supports_vision(
        "custom-vision-name",
        "http://localhost:11434/v1",
    )
