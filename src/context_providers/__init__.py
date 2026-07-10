"""Context provider foundation.

This package is intentionally isolated from the active runtime for now.

It defines portable root metadata and provider capability contracts without
wiring them into LangGraph, chat routes, or tool execution.
"""

from .base import BaseContextAdapter, CapabilityNotSupported
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
    "RootConfigError",
    "load_authorized_roots",
    "verify_authorized_root",
]
