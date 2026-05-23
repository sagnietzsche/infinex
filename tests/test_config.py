from unittest.mock import patch

from core.config import load_settings


def test_batch_tuning_aliases_are_supported() -> None:
    with patch.dict(
        "os.environ",
        {
            "BATCH_SIZE": "24",
            "MAX_WAIT_MS": "15",
            "BATCH_QUEUE_MAX_SIZE": "512",
        },
        clear=True,
    ):
        settings = load_settings()

    assert settings.batch_max_size == 24
    assert settings.batch_max_wait_ms == 15
    assert settings.batch_queue_max_size == 512
