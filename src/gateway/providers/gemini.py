"""Adapter for Google's Gemini API.

Unlike Groq, Gemini does not speak the OpenAI wire format, so this adapter
exists to prove the abstraction actually holds. Three concrete differences:

  - messages are `contents` with `parts`, and the assistant role is "model"
  - there is no system role; the system prompt is a separate top-level field
  - the API key goes in a header, not a bearer token

None of that leaks past this file. The executor, cache, and metrics layers
never learn that Gemini is shaped differently -- which is the entire point of
having a Provider interface.
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

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(Provider):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str = "gemini",
    ) -> None:
        self.name = name
        self.model = model
        # Transport, not client, is injectable -- see OpenAICompatProvider for
        # why: an injected client would bypass the auth header entirely.
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers={"x-goog-api-key": api_key},
            timeout=timeout_s,
            transport=transport,
        )

    # -- wire format --------------------------------------------------------

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        """Translate the internal request into Gemini's schema."""
        contents = []
        system_parts = []

        for message in request.messages:
            if message.role == "system":
                # Gemini has no system role; it takes a separate instruction
                # field. Folding system text into the first user turn would
                # change how the model weights it.
                system_parts.append({"text": message.content})
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        return payload

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return

        detail = response.text[:200]
        if status == 429:
            raise RateLimitError(
                f"gemini rate limited: {detail}", provider=self.name, retry_after=None
            )
        if status >= 500 or status == 408:
            raise TransientProviderError(
                f"gemini returned {status}: {detail}", provider=self.name, status_code=status
            )
        raise PermanentProviderError(
            f"gemini returned {status}: {detail}", provider=self.name, status_code=status
        )

    @staticmethod
    def _extract_text(body: dict[str, Any]) -> str:
        """Pull the response text out of Gemini's nested candidate structure.

        Defensive by design: a safety-filtered response comes back with a
        candidate that has no `parts` at all, and naive indexing raises
        KeyError, which the reliability layer would then misclassify as a
        transport failure and retry pointlessly.
        """
        candidates = body.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts)

    @staticmethod
    def _extract_usage(body: dict[str, Any]) -> Usage:
        meta = body.get("usageMetadata") or {}
        return Usage(
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
        )

    # -- Provider API -------------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        model = request.model or self.model
        try:
            response = await self._client.post(
                f"/models/{model}:generateContent", json=self._payload(request)
            )
        except httpx.TimeoutException as exc:
            raise TransientProviderError("gemini timed out", provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise TransientProviderError(
                f"gemini transport error: {exc}", provider=self.name
            ) from exc

        self._raise_for_status(response)
        body = response.json()

        return ChatResponse(
            content=self._extract_text(body),
            model=model,
            provider=self.name,
            usage=self._extract_usage(body),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        model = request.model or self.model
        params = {"alt": "sse"}
        try:
            async with self._client.stream(
                "POST",
                f"/models/{model}:streamGenerateContent",
                json=self._payload(request),
                params=params,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_status(response)

                usage = Usage()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        log.warning("stream.bad_chunk", provider=self.name, data=data[:120])
                        continue

                    text = self._extract_text(event)
                    if text:
                        yield StreamChunk(type=ChunkType.DELTA, delta=text)
                    if "usageMetadata" in event:
                        usage = self._extract_usage(event)

                yield StreamChunk(type=ChunkType.DONE, usage=usage)
        except httpx.TimeoutException as exc:
            raise TransientProviderError("gemini timed out", provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise TransientProviderError(
                f"gemini transport error: {exc}", provider=self.name
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
