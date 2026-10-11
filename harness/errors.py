"""Exception hierarchy. Everything the harness raises is one of these."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for every error this package raises."""


class ProviderError(HarnessError):
    """An LLM endpoint refused or failed the request."""

    retryable = False


class RateLimitError(ProviderError):
    """429 / quota exhausted. Safe to retry with backoff."""

    retryable = True


class TransientProviderError(ProviderError):
    """5xx / connection reset. Safe to retry with backoff."""

    retryable = True


class AuthenticationError(ProviderError):
    """401 / 403. Never retried -- retrying cannot fix a bad key."""


class ToolError(HarnessError):
    """A tool refused or could not complete its work."""


class PolicyDenied(HarnessError):
    """The execution policy blocked the action."""


class ContextOverflow(HarnessError):
    """The conversation cannot fit the model's context window."""
