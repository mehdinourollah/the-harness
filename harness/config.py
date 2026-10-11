"""Top-level configuration. Overridable from the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


class HarnessConfig(BaseModel):
    model: str = "gpt-4o-mini"
    provider: Literal["openai", "anthropic"] = "openai"
    base_url: str | None = None
    api_key: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int = 4096
    max_turns: int = 24
    max_retries: int = 4
    request_timeout_s: float = 120.0
    max_output_chars: int = 40_000
    workspace: Path = Field(default_factory=Path.cwd)
    #: Enable the Langfuse/OTLP trace export.
    trace: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    @model_validator(mode="after")
    def _sane(self) -> Self:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        return self

    @classmethod
    def with_cli_overrides(cls, base: HarnessConfig, **overrides: Any) -> HarnessConfig:
        """Apply only non-None overrides (CLI flags beat config file)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return base.model_copy(update=clean)
