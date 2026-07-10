"""Context provider foundation.

This package is intentionally isolated from the active runtime for now.

It defines portable root metadata, provider capability contracts, and a small
registry skeleton without wiring them into LangGraph, chat routes, or tool
execution.
"""

from .base import BaseContextAdapter, CapabilityNotSupported
from .registry import (
    ContextProviderRegistry,
    ProviderNotRegistered,
    ProviderRegistryError,
    build_default_registry,
)
from .runtime_tool import (
    RuntimeContextToolError,
    RuntimeContextToolLimits,
    context_tool_call,
)
from .roots import (
    AuthorizedRoot,
    RootConfigError,
    load_authorized_roots,
    verify_authorized_root,
)

__all__ = [
    "AuthorizedRoot",
    "BaseContextAdapter",
    "CapabilityNotSupported",
    "ContextProviderRegistry",
    "ProviderNotRegistered",
    "ProviderRegistryError",
    "RootConfigError",
    "RuntimeContextToolError",
    "RuntimeContextToolLimits",
    "build_default_registry",
    "context_tool_call",
    "load_authorized_roots",
    "verify_authorized_root",
]
