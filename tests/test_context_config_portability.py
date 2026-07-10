
from __future__ import annotations

import json
from pathlib import Path


EXAMPLE_CONFIG = Path("config/local_roots.example.json")
LOCAL_CONFIG = Path("config/local_roots.json")
GITIGNORE = Path(".gitignore")


READ_ONLY_CAPABILITIES = {
    "context.status",
    "context.list",
    "context.read",
    "context.search",
}

WRITE_LIKE_WORDS = {
    "write",
    "create",
    "update",
    "delete",
    "remove",
    "move",
    "rename",
    "mkdir",
    "touch",
    "save",
    "apply",
}


def _load_example() -> dict:
    return json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))


def test_local_roots_example_exists_and_is_json():
    assert EXAMPLE_CONFIG.exists()

    data = _load_example()

    assert isinstance(data, dict)
    assert "roots" in data
    assert isinstance(data["roots"], list)


def test_local_roots_json_is_ignored_not_template_tracked():
    assert GITIGNORE.exists()

    gitignore_text = GITIGNORE.read_text(encoding="utf-8").replace("\\", "/")

    assert "config/local_roots.json" in gitignore_text
    assert "!/config/local_roots.example.json" in gitignore_text


def test_local_roots_example_does_not_contain_dev_machine_paths():
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8").lower().replace("\\", "/")

    forbidden_fragments = [
        "d:/ai/",
        "d:\\ai\\",
        "c:/users/miku",
        "c:\\users\\miku",
        "c:/users/devjit",
        "c:\\users\\devjit",
        "odysseus-upstream-test",
        "ideaverse_test",
    ]

    assert not any(fragment.lower() in text for fragment in forbidden_fragments)


def test_local_roots_example_uses_template_style_paths_or_env_vars():
    data = _load_example()
    roots = data["roots"]

    assert roots

    for root in roots:
        local_path = str(root.get("local_path", ""))
        path_env = str(root.get("path_env", ""))

        assert local_path or path_env

        if path_env:
            assert path_env.upper() == path_env
            assert " " not in path_env

        if local_path:
            normalized = local_path.replace("\\", "/").lower()

            assert (
                normalized.startswith("./")
                or normalized.startswith("~/")
                or "${" in normalized
                or "%userprofile%" in normalized
                or "<" in normalized
                or "example" in normalized
            ), local_path


def test_local_roots_example_capabilities_are_read_only():
    data = _load_example()

    for root in data["roots"]:
        capabilities = set(root.get("allowed_capabilities", []))

        assert capabilities
        assert capabilities <= READ_ONLY_CAPABILITIES

        for capability in capabilities:
            lowered = capability.lower()
            assert not any(word in lowered for word in WRITE_LIKE_WORDS), capability


def test_local_roots_example_safety_scope_is_read_only_sandbox():
    data = _load_example()

    for root in data["roots"]:
        assert root.get("safety_scope") == "read_only_sandbox"


def test_local_roots_example_roots_have_stable_public_fields():
    data = _load_example()

    required = {
        "id",
        "provider_type",
        "display_name",
        "allowed_capabilities",
        "safety_scope",
        "enabled",
    }

    for root in data["roots"]:
        assert required <= set(root.keys())
        assert isinstance(root["id"], str)
        assert isinstance(root["provider_type"], str)
        assert isinstance(root["display_name"], str)
        assert isinstance(root["allowed_capabilities"], list)
        assert isinstance(root["enabled"], bool)
        assert root.get("local_path") or root.get("path_env")


def test_local_roots_json_is_allowed_to_exist_locally_but_not_required():
    # This test documents the intended split:
    # - local_roots.example.json is the portable tracked template
    # - local_roots.json is optional machine-local config
    #
    # The test intentionally does not require LOCAL_CONFIG to exist.
    assert LOCAL_CONFIG.name == "local_roots.json"
    assert EXAMPLE_CONFIG.name == "local_roots.example.json"
