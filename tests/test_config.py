from unittest.mock import patch

from core.config import Settings, load_settings


def test_batch_tuning_aliases_are_supported() -> None:
    with patch.dict(
        "os.environ",
        {
            "BATCH_SIZE": "24",
            "MAX_WAIT_MS": "15",
            "QUEUE_MAX_DEPTH": "512",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.batch_max_size == 24
    assert settings.batch_max_wait_ms == 15
    assert settings.batch_queue_max_size == 512


def test_provider_env_selects_active_provider_and_key() -> None:
    with patch.dict(
        "os.environ",
        {
            "PROVIDER": "anthropic",
            "OPENAI_API_KEY": "ignored-openai-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "SEMANTIC_CACHE_ENABLED": "false",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.provider == "anthropic"
    assert settings.provider_mode == "anthropic"
    assert settings.anthropic_api_key == "anthropic-key"
    assert settings.openai_api_key is None


def test_provider_mode_alias_is_still_supported() -> None:
    with patch.dict(
        "os.environ",
        {
            "PROVIDER_MODE": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11435",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11435"


def test_settings_constructor_accepts_provider_mode_alias() -> None:
    settings = Settings(provider_mode="anthropic")

    assert settings.provider == "anthropic"
    assert settings.provider_mode == "anthropic"


def test_retry_settings_are_loaded_from_environment() -> None:
    with patch.dict(
        "os.environ",
        {
            "MAX_RETRIES": "4",
            "RETRY_BASE_DELAY_MS": "150",
            "RETRY_MAX_DELAY_MS": "3000",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.max_retries == 4
    assert settings.retry_base_delay_ms == 150
    assert settings.retry_max_delay_ms == 3000


def test_circuit_breaker_settings_are_loaded_from_environment() -> None:
    with patch.dict(
        "os.environ",
        {
            "CB_ERROR_THRESHOLD": "75",
            "CB_WINDOW_SECONDS": "45",
            "CB_COOLDOWN_SECONDS": "10",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.cb_error_threshold == 0.75
    assert settings.cb_window_seconds == 45
    assert settings.cb_cooldown_seconds == 10


def test_shutdown_drain_timeout_is_loaded_from_environment() -> None:
    with patch.dict(
        "os.environ",
        {"SHUTDOWN_DRAIN_TIMEOUT_SECONDS": "12.5"},
        clear=True,
    ):
        settings = load_settings()

    assert settings.shutdown_drain_timeout_seconds == 12.5


def test_semantic_cache_settings_are_loaded_from_environment() -> None:
    with patch.dict(
        "os.environ",
        {
            "PROVIDER": "anthropic",
            "OPENAI_API_KEY": "openai-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "SEMANTIC_CACHE_ENABLED": "true",
            "SEMANTIC_CACHE_TTL_SECONDS": "120",
            "SEMANTIC_CACHE_THRESHOLD": "97",
            "SEMANTIC_CACHE_EMBEDDING_MODEL": "text-embedding-3-small",
            "SEMANTIC_CACHE_EMBEDDING_DIMENSION": "1536",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.provider == "anthropic"
    assert settings.openai_api_key == "openai-key"
    assert settings.semantic_cache_enabled is True
    assert settings.semantic_cache_ttl_seconds == 120
    assert settings.semantic_cache_threshold == 0.97
    assert settings.semantic_cache_embedding_model == "text-embedding-3-small"
    assert settings.semantic_cache_embedding_dimension == 1536


def test_semantic_cache_can_be_disabled_independently() -> None:
    with patch.dict(
        "os.environ",
        {
            "PROVIDER": "echo",
            "OPENAI_API_KEY": "ignored",
            "SEMANTIC_CACHE_ENABLED": "false",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.cache_enabled is True
    assert settings.semantic_cache_enabled is False
    assert settings.openai_api_key is None


def test_provider_fallback_chain_selects_primary_and_loads_chain_keys() -> None:
    with patch.dict(
        "os.environ",
        {
            "PROVIDER_FALLBACK_CHAIN": "openai,anthropic",
            "OPENAI_API_KEY": "openai-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.provider == "openai"
    assert settings.provider_mode == "openai"
    assert settings.provider_fallback_chain == ("openai", "anthropic")
    assert settings.openai_api_key == "openai-key"
    assert settings.anthropic_api_key == "anthropic-key"


def test_api_key_metadata_sets_premium_key_priority() -> None:
    with patch.dict(
        "os.environ",
        {
            "API_KEYS": "dev-key,premium-key:tier=premium,slow-key:priority=low",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.allowed_api_keys == ("dev-key", "premium-key", "slow-key")
    assert settings.priority_for_api_key("premium-key") == "high"
    assert settings.priority_for_api_key("slow-key") == "low"
    assert settings.priority_for_api_key("dev-key") is None
