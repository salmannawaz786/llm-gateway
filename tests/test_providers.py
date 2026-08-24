"""Adapter tests against recorded wire formats.

These run with no network and no API keys: `httpx.MockTransport` intercepts
requests inside httpx itself, so the adapter's real request-building,
status-mapping and SSE-parsing code all execute. That is the difference between
testing the adapter and testing a mock of the adapter.

The status-mapping tests matter most. That mapping is the contract between a
provider and the reliability layer, and getting it wrong is silent: classify a
400 as retryable and the gateway retries a malformed request until the budget
drains.
"""

from __future__ import annotations

import json

import httpx
import pytest

from gateway.providers.gemini import GeminiProvider
from gateway.providers.openai_compat import OpenAICompatProvider
from gateway.types import (
    ChatRequest,
    ChunkType,
    Message,
    PermanentProviderError,
    RateLimitError,
    TransientProviderError,
)


def req(text: str = "hello", *, system: str | None = None) -> ChatRequest:
    messages = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=text))
    return ChatRequest(messages=tuple(messages))


def openai_provider(handler: object) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "groq",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def gemini_provider(handler: object) -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


# --- OpenAI-compatible: happy path ----------------------------------------

OPENAI_BODY = {
    "model": "llama-3.1-8b-instant",
    "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 3},
}


async def test_openai_compat_parses_response() -> None:
    provider = openai_provider(lambda request: httpx.Response(200, json=OPENAI_BODY))

    response = await provider.complete(req())

    assert response.content == "hi there"
    assert response.provider == "groq"
    assert response.usage.total_tokens == 12


async def test_openai_compat_sends_expected_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=OPENAI_BODY)

    provider = openai_provider(handler)
    await provider.complete(ChatRequest(messages=(Message(role="user", content="hi"),)))

    assert captured["model"] == "test-model"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["stream"] is False
    assert captured["auth"] == "Bearer test-key"


# --- OpenAI-compatible: error mapping -------------------------------------


@pytest.mark.parametrize("status", [500, 502, 503, 408])
async def test_server_errors_are_retryable(status: int) -> None:
    provider = openai_provider(lambda request: httpx.Response(status, text="upstream boom"))

    with pytest.raises(TransientProviderError) as exc:
        await provider.complete(req())
    assert exc.value.retryable is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_client_errors_are_permanent(status: int) -> None:
    """A 401 must never be retried -- the key will not fix itself."""
    provider = openai_provider(lambda request: httpx.Response(status, text="bad request"))

    with pytest.raises(PermanentProviderError) as exc:
        await provider.complete(req())
    assert exc.value.retryable is False


async def test_rate_limit_honours_retry_after() -> None:
    """The provider knows its own capacity better than our backoff formula."""
    provider = openai_provider(
        lambda request: httpx.Response(429, text="slow down", headers={"retry-after": "2.5"})
    )

    with pytest.raises(RateLimitError) as exc:
        await provider.complete(req())
    assert exc.value.retry_after == 2.5


async def test_timeout_is_mapped_to_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(TransientProviderError):
        await openai_provider(handler).complete(req())


async def test_connection_error_is_mapped_to_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(TransientProviderError):
        await openai_provider(handler).complete(req())


# --- OpenAI-compatible: streaming -----------------------------------------


def sse(*events: str) -> bytes:
    return "".join(f"data: {e}\n\n" for e in events).encode()


async def test_stream_parses_sse_deltas() -> None:
    stream = sse(
        json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
        json.dumps({"choices": [{"delta": {"content": "lo"}}]}),
        json.dumps({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 4,
                                                          "completion_tokens": 2}}),
        "[DONE]",
    )
    provider = openai_provider(lambda request: httpx.Response(200, content=stream))

    chunks = [c async for c in provider.stream(req())]

    text = "".join(c.delta for c in chunks if c.type is ChunkType.DELTA)
    assert text == "Hello"
    assert chunks[-1].type is ChunkType.DONE
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 2


async def test_stream_skips_keepalives_and_malformed_chunks() -> None:
    """Keepalives and comment lines are normal SSE, not errors."""
    raw = (
        b": keepalive\n\n"
        b"\n"
        b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
        b"data: {not json}\n\n"
        b"data: [DONE]\n\n"
    )
    provider = openai_provider(lambda request: httpx.Response(200, content=raw))

    chunks = [c async for c in provider.stream(req())]

    assert "".join(c.delta for c in chunks if c.type is ChunkType.DELTA) == "ok"


async def test_stream_error_status_is_classified() -> None:
    """A streaming request that fails before any data must still map correctly."""
    provider = openai_provider(lambda request: httpx.Response(503, text="unavailable"))

    with pytest.raises(TransientProviderError):
        [c async for c in provider.stream(req())]


# --- Gemini ----------------------------------------------------------------

GEMINI_BODY = {
    "candidates": [{"content": {"parts": [{"text": "bonjour"}]}}],
    "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2},
}


async def test_gemini_parses_response() -> None:
    provider = gemini_provider(lambda request: httpx.Response(200, json=GEMINI_BODY))

    response = await provider.complete(req())

    assert response.content == "bonjour"
    assert response.provider == "gemini"
    assert response.usage.prompt_tokens == 7


async def test_gemini_maps_roles_and_system_prompt() -> None:
    """Gemini has no system role and calls the assistant "model"."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=GEMINI_BODY)

    provider = gemini_provider(handler)
    await provider.complete(req("hi", system="be terse"))

    assert captured["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert captured["systemInstruction"] == {"parts": [{"text": "be terse"}]}
    assert captured["key"] == "test-key"


async def test_gemini_handles_safety_filtered_response() -> None:
    """A filtered candidate has no `parts`; naive indexing would raise KeyError.

    That would surface as an unexpected exception type and be misclassified by
    the reliability layer rather than returning cleanly.
    """
    provider = gemini_provider(
        lambda request: httpx.Response(200, json={"candidates": [{"finishReason": "SAFETY"}]})
    )

    response = await provider.complete(req())
    assert response.content == ""


async def test_gemini_error_mapping() -> None:
    provider = gemini_provider(lambda request: httpx.Response(429, text="quota"))
    with pytest.raises(RateLimitError):
        await provider.complete(req())

    provider = gemini_provider(lambda request: httpx.Response(400, text="bad"))
    with pytest.raises(PermanentProviderError):
        await provider.complete(req())


async def test_gemini_streams() -> None:
    stream = sse(
        json.dumps({"candidates": [{"content": {"parts": [{"text": "bon"}]}}]}),
        json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": "jour"}]}}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
            }
        ),
    )
    provider = gemini_provider(lambda request: httpx.Response(200, content=stream))

    chunks = [c async for c in provider.stream(req())]

    assert "".join(c.delta for c in chunks if c.type is ChunkType.DELTA) == "bonjour"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 2


# --- Interchangeability ----------------------------------------------------


async def test_both_adapters_satisfy_the_same_interface() -> None:
    """The abstraction's whole justification: two very different wire formats
    produce the same internal type, so every downstream feature works with both.
    """
    a = openai_provider(lambda request: httpx.Response(200, json=OPENAI_BODY))
    b = gemini_provider(lambda request: httpx.Response(200, json=GEMINI_BODY))

    for provider in (a, b):
        response = await provider.complete(req())
        assert isinstance(response.content, str)
        assert response.provider == provider.name
        assert response.usage.total_tokens > 0


# --- Metrics ---------------------------------------------------------------


def test_hedge_counters_only_ever_increment() -> None:
    """Counters must not move backwards, or `rate()` breaks silently.

    The executor exposes cumulative integers and the metrics layer samples them
    at scrape time, so the conversion to Counter increments has to be delta
    based rather than a direct set.
    """
    from gateway.observability import metrics

    def value(label: str) -> float:
        return float(metrics.hedges_total.labels(result=label)._value.get())  # noqa: SLF001

    metrics.record_hedges(5, 3)
    after_first = value("fired")

    # Sampling the same totals again must be a no-op, not a double count.
    metrics.record_hedges(5, 3)
    assert value("fired") == after_first

    metrics.record_hedges(8, 3)
    assert value("fired") == after_first + 3


async def test_metrics_endpoint_exposes_gateway_series() -> None:
    from gateway.main import app

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            await client.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
            )
            response = await client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "gateway_requests_total" in body
    assert "gateway_circuit_state" in body
    assert "gateway_request_duration_seconds" in body
