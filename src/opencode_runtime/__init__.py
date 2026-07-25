"""Standalone OpenCode runtime bridge for Odysseus.

OpenCode remains an external runtime. This package exposes transport,
normalization, and lifecycle primitives only; it does not import Legacy,
LangGraph, MultiGraph, or Odysseus capability tools.
"""

from .config import OpenCodeRuntimeConfig
from .normalizer import OpenCodeEventNormalizer, extract_final_assistant_message
from .process_manager import (
    OpenCodeServerLifecycleError,
    OpenCodeServerManager,
)
from .service import OpenCodeRuntimeService

__all__ = [
    "OpenCodeEventNormalizer",
    "OpenCodeRuntimeConfig",
    "OpenCodeRuntimeService",
    "OpenCodeServerLifecycleError",
    "OpenCodeServerManager",
    "extract_final_assistant_message",
]
