from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "llm-gateway"
    batch_max_size: int = 8
    batch_max_wait_ms: int = 25
    provider_mode: str = "echo"


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")

    return value


def load_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", Settings.app_name),
        batch_max_size=_read_positive_int(
            "BATCH_MAX_SIZE", Settings.batch_max_size
        ),
        batch_max_wait_ms=_read_positive_int(
            "BATCH_MAX_WAIT_MS", Settings.batch_max_wait_ms
        ),
        provider_mode=os.getenv("PROVIDER_MODE", Settings.provider_mode),
    )
