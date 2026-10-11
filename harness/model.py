"""Build a PydanticAI model from config.

One place knows about vendor selection; everything else takes a `Model`.
Point `base_url` at any OpenAI- or Anthropic-compatible endpoint
(OpenAI, vLLM, Ollama, Groq, DeepSeek, local proxies).
"""

from __future__ import annotations

import os

from pydantic_ai.models import Model

from harness.config import HarnessConfig

KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


def build_model(config: HarnessConfig) -> Model:
    api_key = config.api_key or os.environ.get(KEY_ENV[config.provider])
    base_url = config.base_url or DEFAULT_URLS[config.provider]

    if config.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        # A missing key still constructs: real endpoints fail with a clear
        # 401, and keyless local proxies keep working.
        return AnthropicModel(
            config.model,
            provider=AnthropicProvider(
                base_url=base_url, api_key=api_key or "not-needed"
            ),
        )

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        config.model,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key or "not-needed",
        ),
    )
