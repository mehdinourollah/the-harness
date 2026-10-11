"""Langfuse tracing over OpenTelemetry.

PydanticAI emits OpenTelemetry spans for every model and tool call. Langfuse
ingests OTLP at `/api/public/otel`, authenticated with Basic auth over the
project's public/secret key pair. Nothing here is Langfuse-specific beyond the
endpoint and that header, so the same wiring works against any OTLP backend.
"""

from __future__ import annotations

import base64
import os

from harness.config import HarnessConfig
from harness.errors import ConfigError


def setup_tracing(config: HarnessConfig) -> bool:
    """Wire PydanticAI spans to Langfuse. Returns True when active.

    A no-op (returns False) when tracing is off, so the harness runs fine
    without an observability backend.
    """
    if not config.trace:
        return False

    public = config.langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = config.langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        raise ConfigError(
            "tracing enabled but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are unset"
        )

    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
        f"{config.langfuse_host.rstrip('/')}/api/public/otel"
    )
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"Authorization=Basic {auth},x-langfuse-ingestion-version=4"
    )

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import set_tracer_provider
    from pydantic_ai import Agent

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(provider)
    Agent.instrument_all()
    return True
