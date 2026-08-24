"""Adapter for any provider speaking the OpenAI chat-completions wire format.

Groq, Together, Fireworks, OpenRouter, vLLM and Ollama all expose this shape,
so one adapter covers them by changing a base URL. That is why it is written
generically rather than as a `GroqProvider`.

The two things worth reading here are the error mapping and the SSE parser.
Both are where a naive HTTP client quietly gets things wrong.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from gateway.providers.base import Provider
from gateway.types import (
    ChatRequest,
    ChatResponse,
    ChunkType,
    PermanentProviderError,
    RateLimitError,
    StreamChunk,
    TransientProviderError,
    Usage,
)

log = structlog.get_logger(__name__)


class OpenAICompatProvider(Provider):
    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        # One client for the provider's lifetime. Creating an AsyncClient per
        # request is the most common performance bug in Python HTTP code: it
        # discards the connection pool and pays a fresh TLS handshake every
        # time.
        #
        # Tests inject a `transport`, not a whole client. An earlier version
        # accepted a client, which meant an injected one silently carried no
        # Authorization header -- so the tests exercised a code path that could
        # never authenticate, and would have passed even if auth were broken.
        # Owning the client here keeps auth non-optional.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    # -- wire format --------------------------------------------------------

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        return {
            "model": request.model or self.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": stream,
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status onto the gateway's error taxonomy.

        This mapping is the entire contract between a provider and the
        reliability layer. Classify a 400 as retryable and the gateway will
        cheerfully retry a malformed request forever; classify a 503 as
        permanent and it will give up on a provider that was about to recover.
        """
        status = response.status_code
        if status < 400:
            return

        detail = response.text[:200]

        if status == 429:
            # Honour the provider's own backoff instruction when it gives one --
            # it knows more about its capacity than our exponential backoff does.
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                f"{self.name} rate limited: {detail}",
                provider=self.name,
                retry_after=float(retry_after) if retry_after else None,
            )

        if status in (408, 409, 425) or status >= 500:
            raise TransientProviderError(
                f"{self.name} returned {status}: {detail}",
                provider=self.name,
                status_code=status,
            )

        # Everything else in 4xx is our fault: bad auth, bad model name,
        # malformed body. Retrying produces the same failure more slowly.
        raise PermanentProviderError(
            f"{self.name} returned {status}: {detail}",
            provider=self.name,
            status_code=status,
        )

    # -- Provider API -------------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            response = await self._client.post(
                "/chat/completions", json=self._payload(request, stream=False)
            )
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"{self.name} timed out", provider=self.name) from exc
        except httpx.HTTPError as exc:
            # Connection reset, DNS failure, refused connection: all transient
            # from the caller's point of view, and all worth failing over.
            raise TransientProviderError(
                f"{self.name} transport error: {exc}", provider=self.name
            ) from exc

        self._raise_for_status(response)
        body = response.json()

        choice = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return ChatResponse(
            content=choice,
            model=body.get("model", self.model),
            provider=self.name,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        payload = self._payload(request, stream=True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    # The body has not been read yet on a streaming response,
                    # so it must be pulled in before the text is available --
                    # otherwise the error message is empty.
                    await response.aread()
                    self._raise_for_status(response)

                prompt_tokens = completion_tokens = 0

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        # Blank lines are SSE keepalives and comment lines start
                        # with ':'. Both are normal and must be skipped rather
                        # than parsed as JSON.
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        log.warning("stream.bad_chunk", provider=self.name, data=data[:120])
                        continue

                    if usage := event.get("usage"):
                        # Most providers only report token counts on the final
                        # event, so capture it whenever it appears.
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)

                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            yield StreamChunk(type=ChunkType.DELTA, delta=delta)

                yield StreamChunk(
                    type=ChunkType.DONE,
                    usage=Usage(
                        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
                    ),
                )
        except httpx.TimeoutException as exc:
            raise TransientProviderError(f"{self.name} timed out", provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise TransientProviderError(
                f"{self.name} transport error: {exc}", provider=self.name
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def groq_provider(api_key: str, model: str, **kwargs: Any) -> OpenAICompatProvider:
    """Groq's free tier. OpenAI-compatible, so no bespoke adapter is needed."""
    return OpenAICompatProvider(
        "groq",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model=model,
        **kwargs,
    )
