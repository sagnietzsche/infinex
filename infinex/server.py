"""Phase 7: serving API.

A driver loop (asyncio task) repeatedly calls engine.step() and routes
emitted tokens to per-request asyncio.Queue objects. Streaming is an
async generator per request. Optionally wrapped in FastAPI with SSE for
an OpenAI-completions-style endpoint.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from .scheduler import Engine
from .sequence import SamplingParams
from .tokenizer import Tokenizer


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stream: bool = False
    seed: int | None = None

    def sampling_params(self) -> SamplingParams:
        raise NotImplementedError


@dataclass
class CompletionResponse:
    request_id: int
    text: str
    finish_reason: str  # "stop" | "length"
    prompt_tokens: int
    completion_tokens: int


class LLMServer:
    """Async front-end over the Engine."""

    def __init__(self, engine: Engine, tokenizer: Tokenizer) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self._queues: dict[int, asyncio.Queue] = {}
        self._driver: asyncio.Task | None = None
        self._next_request_id = 0

    async def start(self) -> None:
        """Spawn the driver loop task."""
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def _drive(self) -> None:
        """Loop: engine.step(), push StepOutput tokens onto per-request
        queues, sleep briefly when idle."""
        raise NotImplementedError

    def _submit(self, request: CompletionRequest) -> int:
        """Tokenize, build a Sequence, add to the engine.
        Returns the request id."""
        raise NotImplementedError

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Non-streaming: run to completion, return the full response."""
        raise NotImplementedError

    async def stream(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Streaming: yield decoded text chunks as tokens are produced."""
        raise NotImplementedError
        yield  # makes this an async generator


def create_app(server: LLMServer):
    """Optional FastAPI wrapper: POST /v1/completions with SSE streaming
    when stream=true. Import fastapi lazily so the core has no hard dep."""
    raise NotImplementedError
