from collections.abc import Mapping
from dataclasses import dataclass, field
import os


PROVIDER_MODEL_MAPPINGS: dict[str, str] = {
    "openai/gpt-4o": "anthropic/claude-opus-4-5",
}


MODEL_PRICING_USD_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gateway-echo": {"input": 0.0, "output": 0.0},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-opus-4-5": {"input": 0.015, "output": 0.075},
    "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
}


@dataclass(frozen=True)
class Settings:
    app_name: str = "llm-gateway"
    batch_max_size: int = 16
    batch_max_wait_ms: int = 20
    batch_queue_max_size: int = 1024
    provider: str = "echo"
    provider_mode: str | None = None
    provider_fallback_chain: tuple[str, ...] = ()
    provider_model_mapping: Mapping[str, str] = field(
        default_factory=lambda: dict(PROVIDER_MODEL_MAPPINGS)
    )
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True
    semantic_cache_enabled: bool = True
    semantic_cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.95
    semantic_cache_embedding_model: str = "text-embedding-3-small"
    semantic_cache_embedding_dimension: int = 1536
    allowed_api_keys: tuple[str, ...] = ()
    rate_limit_capacity: int = 60
    rate_limit_refill_per_second: float = 1.0
    max_retries: int = 3
    retry_base_delay_ms: int = 200
    retry_max_delay_ms: int = 5000
    cb_error_threshold: float = 0.5
    cb_window_seconds: int = 60
    cb_cooldown_seconds: int = 30

    def __post_init__(self) -> None:
        provider = (self.provider_mode or self.provider).lower()
        provider_fallback_chain = tuple(
            item.lower() for item in self.provider_fallback_chain if item
        )
        if not provider_fallback_chain:
            provider_fallback_chain = (provider,)
        provider = provider_fallback_chain[0]
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "provider_mode", provider)
        object.__setattr__(
            self, "provider_fallback_chain", provider_fallback_chain
        )


def _read_positive_int(
    name: str, default: int, *, aliases: tuple[str, ...] = ()
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        for alias in aliases:
            raw_value = os.getenv(alias)
            if raw_value is not None:
                break

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")

    return value


def _read_non_negative_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")

    return value


def _read_positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")

    return value


def _read_threshold(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc

    if value > 1:
        value = value / 100

    if value <= 0 or value > 1:
        raise ValueError(f"{name} must be greater than zero and at most 100")

    return value


def _read_csv(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, "")
    return tuple(
        value.strip() for value in raw_value.split(",") if value.strip()
    )


def load_settings() -> Settings:
    provider_fallback_chain = tuple(
        provider.lower() for provider in _read_csv("PROVIDER_FALLBACK_CHAIN")
    )
    semantic_cache_enabled = (
        os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() != "false"
    )
    provider = (
        provider_fallback_chain[0]
        if provider_fallback_chain
        else os.getenv(
            "PROVIDER", os.getenv("PROVIDER_MODE", Settings.provider)
        ).lower()
    )
    configured_providers = set(provider_fallback_chain or (provider,))

    return Settings(
        app_name=os.getenv("APP_NAME", Settings.app_name),
        batch_max_size=_read_positive_int(
            "BATCH_MAX_SIZE",
            Settings.batch_max_size,
            aliases=("BATCH_SIZE",),
        ),
        batch_max_wait_ms=_read_positive_int(
            "BATCH_MAX_WAIT_MS",
            Settings.batch_max_wait_ms,
            aliases=("MAX_WAIT_MS",),
        ),
        batch_queue_max_size=_read_positive_int(
            "BATCH_QUEUE_MAX_SIZE", Settings.batch_queue_max_size
        ),
        provider=provider,
        provider_fallback_chain=provider_fallback_chain,
        openai_api_key=(
            os.getenv("OPENAI_API_KEY")
            if "openai" in configured_providers or semantic_cache_enabled
            else None
        ),
        anthropic_api_key=(
            os.getenv("ANTHROPIC_API_KEY")
            if "anthropic" in configured_providers
            else None
        ),
        google_api_key=(
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if configured_providers & {"gemini", "google"}
            else None
        ),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", Settings.ollama_base_url),
        redis_url=os.getenv("REDIS_URL", Settings.redis_url),
        cache_ttl_seconds=_read_positive_int(
            "CACHE_TTL_SECONDS", Settings.cache_ttl_seconds
        ),
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() != "false",
        semantic_cache_enabled=semantic_cache_enabled,
        semantic_cache_ttl_seconds=_read_positive_int(
            "SEMANTIC_CACHE_TTL_SECONDS",
            Settings.semantic_cache_ttl_seconds,
        ),
        semantic_cache_threshold=_read_threshold(
            "SEMANTIC_CACHE_THRESHOLD",
            Settings.semantic_cache_threshold,
        ),
        semantic_cache_embedding_model=os.getenv(
            "SEMANTIC_CACHE_EMBEDDING_MODEL",
            Settings.semantic_cache_embedding_model,
        ),
        semantic_cache_embedding_dimension=_read_positive_int(
            "SEMANTIC_CACHE_EMBEDDING_DIMENSION",
            Settings.semantic_cache_embedding_dimension,
        ),
        allowed_api_keys=_read_csv("API_KEYS"),
        rate_limit_capacity=_read_positive_int(
            "RATE_LIMIT_CAPACITY", Settings.rate_limit_capacity
        ),
        rate_limit_refill_per_second=_read_positive_float(
            "RATE_LIMIT_REFILL_PER_SECOND",
            Settings.rate_limit_refill_per_second,
        ),
        max_retries=_read_non_negative_int("MAX_RETRIES", Settings.max_retries),
        retry_base_delay_ms=_read_positive_int(
            "RETRY_BASE_DELAY_MS", Settings.retry_base_delay_ms
        ),
        retry_max_delay_ms=_read_positive_int(
            "RETRY_MAX_DELAY_MS", Settings.retry_max_delay_ms
        ),
        cb_error_threshold=_read_threshold(
            "CB_ERROR_THRESHOLD", Settings.cb_error_threshold
        ),
        cb_window_seconds=_read_positive_int(
            "CB_WINDOW_SECONDS", Settings.cb_window_seconds
        ),
        cb_cooldown_seconds=_read_positive_int(
            "CB_COOLDOWN_SECONDS", Settings.cb_cooldown_seconds
        ),
    )
