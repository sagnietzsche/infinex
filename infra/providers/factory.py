from core.config import Settings
from infra.providers.base import BaseProvider
from infra.providers.echo import EchoProvider
from infra.providers.litellm_provider import (
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenAIProvider,
)


def build_provider(settings: Settings) -> BaseProvider:
    provider = settings.provider.lower()

    if provider == "echo":
        return EchoProvider()
    if provider == "openai":
        _require_api_key("OPENAI_API_KEY", settings.openai_api_key)
        return OpenAIProvider(api_key=settings.openai_api_key)
    if provider == "anthropic":
        _require_api_key("ANTHROPIC_API_KEY", settings.anthropic_api_key)
        return AnthropicProvider(api_key=settings.anthropic_api_key)
    if provider in {"gemini", "google"}:
        _require_api_key("GOOGLE_API_KEY or GEMINI_API_KEY", settings.google_api_key)
        return GeminiProvider(api_key=settings.google_api_key)
    if provider == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url)

    raise ValueError(f"Unsupported provider: {settings.provider}")


def build_streaming_provider(settings: Settings) -> BaseProvider:
    return build_provider(settings)


def _require_api_key(name: str, value: str | None) -> None:
    if not value:
        raise ValueError(f"{name} must be set when that provider is selected")
