"""
VoxCore — brain/llm_client.py
Groq async streaming LLM wrapper — single interface to all models.
"""

import asyncio
import logging
import os
import time
from typing import AsyncGenerator, Optional, Any

from groq import AsyncGroq, RateLimitError, APIError

from core.event_bus import EventBus, EventType

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "llama-3.1-8b-instant": {
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "max_tokens": 2048,
        "temperature": 0.5,
        "top_p": 0.9,
    },
    "moonshotai/kimi-k2-instruct": {
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.95,
    },
    "qwen/qwen3-32b": {
        "max_tokens": 4096,
        "temperature": 0.4,
        "top_p": 0.9,
    },
    "openai/gpt-oss-120b": {
        "max_tokens": 2048,
        "temperature": 0.5,
        "top_p": 0.9,
    },
    "meta-llama/llama-guard-4-12b": {
        "max_tokens": 256,
        "temperature": 0.0,
    },
}

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0


class LLMClient:
    """Unified async streaming LLM client for all Groq models."""

    def __init__(self, event_bus: EventBus, config: dict) -> None:
        self.event_bus = event_bus
        self.config = config
        # max_retries=0: disable Groq SDK's internal retry-on-429 loop.
        # Without this, a rate-limit response causes the SDK to block for
        # 20–30 s per retry before surfacing RateLimitError to our handler.
        # With max_retries=0 the error surfaces immediately so our
        # pick_available() / fallback logic in TurnManager can cut over
        # to a non-rate-limited model within the same turn.
        self._groq_client = AsyncGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            max_retries=0,
        )

        # Usage tracking
        self._session_token_count: int = 0
        self._call_count: int = 0
        self._model_usage: dict[str, int] = {}

        # Rate limit state: model → earliest-retry timestamp
        self._rate_limited: dict[str, float] = {}

        self._logger = logging.getLogger("LLMClient")

    # ── streaming interface ────────────────────────────────────────────────

    async def stream_response(
        self, messages: list[dict], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        params = MODEL_CONFIGS.get(model, {}).copy()
        params.update(kwargs)
        # Remove stream key from params if present; we control it explicitly
        params.pop("stream", None)

        # Rate limit guard
        if self.is_rate_limited(model):
            wait = self._rate_limited[model] - time.time()
            self._logger.warning("Model %s is rate limited for %.1fs", model, wait)
            await self.event_bus.publish(
                EventType.LLM_ERROR,
                {"model": model, "error": "rate_limited", "retry_after": wait},
            )
            return

        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                stream = await self._groq_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **params,
                )

                token_count = 0
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        token_count += 1
                        yield delta.content

                # Track usage
                self._session_token_count += token_count
                self._call_count += 1
                self._model_usage[model] = self._model_usage.get(model, 0) + token_count
                return

            except asyncio.CancelledError:
                self._logger.debug("LLM stream cancelled (interrupt)")
                raise

            except RateLimitError as exc:
                retry_after = float(getattr(exc, "retry_after", 60) or 60)
                self._rate_limited[model] = time.time() + retry_after
                self._logger.warning(
                    "Rate limited on %s for %.0fs", model, retry_after
                )
                await self.event_bus.publish(
                    EventType.LLM_ERROR,
                    {"model": model, "error": "rate_limited", "retry_after": retry_after},
                )
                return

            except APIError as exc:
                last_err = exc
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self._logger.warning(
                    "Groq API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)

        # Exhausted retries
        self._logger.error("Groq API failed after %d retries: %s", MAX_RETRIES, last_err)
        await self.event_bus.publish(
            EventType.LLM_ERROR,
            {"model": model, "error": str(last_err)},
        )

    # ── non-streaming interface ────────────────────────────────────────────

    async def complete(self, messages: list[dict], model: str, **kwargs: Any) -> str:
        params = MODEL_CONFIGS.get(model, {}).copy()
        params.update(kwargs)
        params.pop("stream", None)

        last_err: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self._groq_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    **params,
                )

                text = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                tokens = getattr(usage, "total_tokens", len(text) // 4) if usage else len(text) // 4

                self._session_token_count += tokens
                self._call_count += 1
                self._model_usage[model] = self._model_usage.get(model, 0) + tokens
                return text

            except asyncio.CancelledError:
                raise

            except RateLimitError as exc:
                retry_after = float(getattr(exc, "retry_after", 60) or 60)
                self._rate_limited[model] = time.time() + retry_after
                self._logger.warning("Rate limited on %s for %.0fs", model, retry_after)
                await self.event_bus.publish(
                    EventType.LLM_ERROR,
                    {"model": model, "error": "rate_limited", "retry_after": retry_after},
                )
                return ""

            except APIError as exc:
                last_err = exc
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self._logger.warning(
                    "Groq API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)

        self._logger.error("Groq API failed after %d retries: %s", MAX_RETRIES, last_err)
        await self.event_bus.publish(
            EventType.LLM_ERROR,
            {"model": model, "error": str(last_err)},
        )
        return ""

    # ── usage & rate-limit queries ─────────────────────────────────────────

    def is_rate_limited(self, model: str) -> bool:
        return model in self._rate_limited and time.time() < self._rate_limited[model]

    def pick_available(self, *models: str) -> str:
        """Return the first non-rate-limited model from the list.

        If every model in the list is currently rate-limited, returns
        the first (primary) model so the call still goes through and
        our RateLimitError handler records the retry-after time.
        Empty/None entries in the list are silently skipped.
        """
        candidates = [m for m in models if m]
        for m in candidates:
            if not self.is_rate_limited(m):
                return m
        return candidates[0] if candidates else (models[0] if models else "")