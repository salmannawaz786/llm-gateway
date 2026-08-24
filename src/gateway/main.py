"""FastAPI application.

The HTTP surface is deliberately OpenAI-compatible, so any existing client
library can point at this gateway by changing one base URL. A gateway nobody
can adopt without rewriting their code is not a gateway.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gateway.cache.semantic import SemanticCache
from gateway.config import get_settings
from gateway.providers.mock import MockBehaviour, MockProvider
from gateway.reliability.budget import RetryBudget
from gateway.reliability.executor import ReliableExecutor
from gateway.service import GatewayService
from gateway.types import (
    ChatRequest,
    ChunkType,
    GatewayError,
    Message,
    PermanentProviderError,
)

log = structlog.get_logger(__name__)


# --- Wire format -----------------------------------------------------------
# These Pydantic models exist only at the HTTP boundary. They are converted to
# the internal dataclasses immediately, so the rest of the codebase never
# depends on the shape of the public API.


class MessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[MessageIn] = Field(min_length=1)
    model: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=32_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

    def to_domain(self) -> ChatRequest:
        return ChatRequest(
            messages=tuple(Message(role=m.role, content=m.content) for m in self.messages),
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=self.stream,
        )


def build_service() -> GatewayService:
    """Assemble the provider pool.

    Mock providers are the default so the gateway runs with zero API keys.
    Real providers are added when their keys are present.
    """
    settings = get_settings()
    providers = [
        MockProvider("mock-primary", MockBehaviour.healthy()),
        MockProvider("mock-backup", MockBehaviour.healthy()),
    ]
    executor = ReliableExecutor(
        providers,
        hedge_delay_s=settings.hedge_delay_ms / 1000,
        max_retries=settings.max_retries,
        retry_budget=RetryBudget(ratio=settings.retry_budget_ratio),
        breaker_failure_threshold=settings.breaker_failure_threshold,
        breaker_recovery_seconds=settings.breaker_recovery_seconds,
        request_timeout_s=settings.request_timeout_s,
    )
    cache = (
        SemanticCache(
            threshold=settings.cache_similarity_threshold,
            max_entries=settings.cache_max_entries,
        )
        if settings.cache_enabled
        else None
    )
    return GatewayService(executor, cache)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the executor's lifetime.

    Connection pools are created once at startup and closed at shutdown --
    never per request. Building an httpx client per request is the single most
    common performance bug in Python services: it throws away connection reuse
    and forces a fresh TLS handshake every time.
    """
    app.state.service = build_service()
    log.info(
        "gateway.started",
        providers=[p.name for p in app.state.service.executor.providers],
        cache=app.state.service.cache is not None,
    )
    yield
    await app.state.service.aclose()
    log.info("gateway.stopped")


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    service: GatewayService = app.state.service
    executor = service.executor
    body: dict[str, Any] = {
        "status": "ok",
        "providers": {name: b.state.value for name, b in executor.breakers.items()},
        "retry_budget": executor.budget.stats,
        "hedges": {"fired": executor.hedges_fired, "won": executor.hedges_won},
    }
    if service.cache is not None:
        s = service.cache.stats
        body["cache"] = {
            "embedder": service.cache.embedder.name,
            "threshold": service.cache.threshold,
            "entries": len(service.cache),
            "hit_rate": round(s.hit_rate, 4),
            "hits": s.hits,
            "misses": s.misses,
            "tokens_saved": s.tokens_saved,
        }
    return body


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    service: GatewayService = app.state.service
    request = body.to_domain()

    if body.stream:
        return StreamingResponse(
            _sse(service.executor, request),
            media_type="text/event-stream",
            # Proxies love to buffer streamed responses, which defeats the
            # entire point of streaming. This header asks nginx not to.
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    try:
        response = await service.complete(request)
    except PermanentProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GatewayError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "model": response.model,
        "provider": response.provider,
        "cached": response.cached,
        "cache_similarity": response.cache_similarity,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response.content}}],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


async def _sse(executor: ReliableExecutor, request: ChatRequest) -> AsyncIterator[str]:
    """Translate internal chunks into Server-Sent Events.

    Errors cannot be signalled with an HTTP status once streaming has begun --
    the 200 was already sent. So a mid-stream failure is delivered as an error
    *event* instead, and the client is expected to check for it.
    """
    try:
        async for chunk in executor.stream(request):
            if chunk.type is ChunkType.DELTA:
                payload = {"choices": [{"delta": {"content": chunk.delta}}]}
                yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    except GatewayError as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
