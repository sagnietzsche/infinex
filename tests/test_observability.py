import logging

from fastapi.testclient import TestClient

from core.config import Settings
from main import create_app


def test_metrics_endpoint_exports_gateway_metrics() -> None:
    app = create_app(Settings(cache_enabled=False, batch_max_wait_ms=5))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "metrics"}]},
        )
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert metrics.status_code == 200
    body = metrics.text
    assert "request_count_total" in body
    assert "latency_histogram_bucket" in body
    assert "cache_hit_rate" in body
    assert "current_queue_depth" in body


def test_structured_logs_include_trace_id_and_batcher_id(caplog) -> None:
    app = create_app(Settings(cache_enabled=False, batch_max_wait_ms=5))

    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"x-trace-id": "trace-test-123"},
                json={"messages": [{"role": "user", "content": "logs"}]},
            )

    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "trace-test-123"
    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert '"trace_id": "trace-test-123"' in log_output
    assert '"batcher_id": "async-' in log_output
