"""Phase 8: benchmarking.

Load generator submitting M concurrent requests with configurable
prompt/output lengths, measuring TTFT, inter-token latency, throughput,
and latency percentiles. The payoff plots for the README:
  - throughput vs concurrency: static vs continuous batching
  - memory utilization: contiguous vs paged cache
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchConfig:
    num_requests: int = 32
    concurrency: int = 8
    prompt_len: int = 64
    output_len: int = 64
    temperature: float = 1.0
    seed: int = 0


@dataclass
class RequestMetrics:
    request_id: int
    submit_time: float
    first_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        raise NotImplementedError

    @property
    def e2e_latency(self) -> float:
        raise NotImplementedError

    def inter_token_latencies(self) -> list[float]:
        raise NotImplementedError


@dataclass
class BenchResult:
    config: BenchConfig
    requests: list[RequestMetrics]

    def throughput_tokens_per_sec(self) -> float:
        """Total output tokens / wall time across the whole run."""
        raise NotImplementedError

    def percentiles(self, metric: str, ps: tuple[float, ...] = (50, 90, 99)) -> dict[float, float]:
        """Percentiles for 'ttft' or 'e2e'."""
        raise NotImplementedError

    def summary(self) -> str:
        """Human-readable table of the numbers above."""
        raise NotImplementedError


async def run_benchmark(server, config: BenchConfig) -> BenchResult:
    """Submit config.num_requests with at most config.concurrency in
    flight, recording per-token timestamps via the streaming API."""
    raise NotImplementedError


def sweep_concurrency(concurrencies: list[int]) -> None:
    """Run the static-vs-continuous throughput sweep and emit the
    README plots (matplotlib, saved to bench_out/)."""
    raise NotImplementedError


if __name__ == "__main__":
    sweep_concurrency([1, 2, 4, 8, 16, 32])
