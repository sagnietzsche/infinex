from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "llm-gateway"
    batch_max_size: int = 16
    batch_max_wait_ms: int = 20
    batch_queue_max_size: int = 1024
    provider_mode: str = "echo"
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = True
    allowed_api_keys: tuple[str, ...] = ()
    rate_limit_capacity: int = 60
    rate_limit_refill_per_second: float = 1.0


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


def _read_csv(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, "")
    return tuple(
        value.strip() for value in raw_value.split(",") if value.strip()
    )


def load_settings() -> Settings:
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
        provider_mode=os.getenv("PROVIDER_MODE", Settings.provider_mode),
        redis_url=os.getenv("REDIS_URL", Settings.redis_url),
        cache_ttl_seconds=_read_positive_int(
            "CACHE_TTL_SECONDS", Settings.cache_ttl_seconds
        ),
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() != "false",
        allowed_api_keys=_read_csv("API_KEYS"),
        rate_limit_capacity=_read_positive_int(
            "RATE_LIMIT_CAPACITY", Settings.rate_limit_capacity
        ),
        rate_limit_refill_per_second=_read_positive_float(
            "RATE_LIMIT_REFILL_PER_SECOND",
            Settings.rate_limit_refill_per_second,
        ),
    )
