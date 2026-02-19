"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        VOXCORE — brain/llm_client.py                            ║
║          GROQ ASYNC STREAMING LLM WRAPPER — SINGLE INTERFACE TO ALL MODELS     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Async wrapper around the Groq Python SDK for streaming LLM completions.
    Provides a unified interface to call any Groq-hosted model. Handles:
    - Streaming responses (async generator yielding tokens)
    - Retry logic with exponential backoff
    - Rate limit handling and backoff
    - Token usage tracking per call and per session
    - Model-specific parameter tuning

    This is the ONLY module that talks to Groq's LLM API. All other modules
    get LLM responses through this client.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import os                               # Environment variables
import time                             # Rate limit tracking

from typing import AsyncGenerator, Optional, Any
from groq import AsyncGroq              # Groq async SDK

from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS — dict mapping model names to their optimal parameters:

    MODEL_CONFIGS = {
        "llama-3.1-8b-instant": {
            "max_tokens": 512,          # Keep responses short for voice
            "temperature": 0.7,         # Slightly creative
            "top_p": 0.9,
            "stream": True,
        },
        "meta-llama/llama-4-scout-17b-16e-instruct": {
            "max_tokens": 2048,
            "temperature": 0.5,         # More focused for reasoning
            "top_p": 0.9,
            "stream": True,
        },
        "moonshotai/kimi-k2-instruct": {
            "max_tokens": 4096,         # Longer for tool use
            "temperature": 0.3,         # Precise for tool calling
            "top_p": 0.95,
            "stream": True,
        },
        "qwen/qwen3-32b": {
            "max_tokens": 4096,
            "temperature": 0.4,
            "top_p": 0.9,
            "stream": True,
        },
        "openai/gpt-oss-120b": {
            "max_tokens": 2048,
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": True,
        },
        "meta-llama/llama-guard-4-12b": {
            "max_tokens": 256,          # Safety check — short response
            "temperature": 0.0,         # Deterministic
            "stream": False,            # No need to stream safety checks
        },
    }

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0      # seconds — doubles each retry (1s, 2s, 4s)

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: LLMClient
──────────────────────────────────────────────────────────────────────────────────
    Unified async streaming LLM client for all Groq models.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For publishing LLM_ERROR events
            - config: dict — The "models" section from config.yaml
        
        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            self.config: dict                    = config
            self._groq_client: AsyncGroq         = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
            
            # Token usage tracking
            self._session_token_count: int       = 0       # Total tokens used this session
            self._call_count: int                = 0       # Total API calls this session
            self._model_usage: dict[str, int]    = {}      # Tokens per model: {"llama-3.1-8b": 5000, ...}
            
            # Rate limit state per model
            self._rate_limited: dict[str, float] = {}      # model → retry_after timestamp
            
            self._logger: logging.Logger         = logging.getLogger("LLMClient")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def stream_response(self, messages: list[dict], model: str,
                               **kwargs) -> AsyncGenerator[str, None]
        INPUTS:
            - messages: list[dict] — OpenAI-format messages array:
                [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
            - model: str — Model identifier (e.g. "llama-3.1-8b-instant")
            - **kwargs — Override any model config parameter (temperature, max_tokens, etc.)
        OUTPUT:
            - AsyncGenerator[str, None] — Yields individual tokens (strings) as they arrive
        WHAT IT DOES:
            1. Gets model-specific config: params = MODEL_CONFIGS.get(model, {}).copy()
            2. Overrides with any kwargs: params.update(kwargs)
            3. Checks rate limit:
               If model in self._rate_limited and time.time() < self._rate_limited[model]:
                   self._logger.warning(f"Model {model} is rate limited")
                   Publish LLM_ERROR event
                   return
            4. Tries up to MAX_RETRIES:
               stream = await self._groq_client.chat.completions.create(
                   model=model,
                   messages=messages,
                   stream=True,
                   **params
               )
               async for chunk in stream:
                   if chunk.choices[0].delta.content:
                       token = chunk.choices[0].delta.content
                       yield token
            5. Tracks usage: self._session_token_count += total_tokens
            6. Increments self._call_count
        
        ERROR HANDLING:
            On groq.RateLimitError (429):
                - Extract retry_after from error headers
                - Set self._rate_limited[model] = time.time() + retry_after
                - Log warning: "Rate limited on {model} for {retry_after}s"
                - Publish LLM_ERROR with error details
            On groq.APIError:
                - Retry with exponential backoff
                - After MAX_RETRIES, publish LLM_ERROR and give up
            On asyncio.CancelledError:
                - Clean up — this is normal during interruptions
                - Re-raise to propagate cancellation

    async def complete(self, messages: list[dict], model: str, **kwargs) -> str
        INPUTS:
            - messages: list[dict] — Messages array
            - model: str — Model identifier
            - **kwargs — Parameter overrides
        OUTPUT:
            - str — Complete response text (non-streaming)
        WHAT IT DOES:
            1. Same as stream_response but with stream=False
            2. Returns the complete text at once
            3. Used for safety guard checks and other non-streaming needs
            4. response = await self._groq_client.chat.completions.create(
                   model=model, messages=messages, stream=False, **params
               )
            5. Returns response.choices[0].message.content

    def get_usage_stats(self) -> dict
        INPUTS: None
        OUTPUT: dict — Token usage statistics:
            {
                "total_tokens": int,
                "total_calls": int,
                "per_model": dict[str, int],
                "rate_limited_models": list[str]
            }
        WHAT IT DOES:
            Returns a snapshot of current session usage for monitoring/debugging.

    def is_rate_limited(self, model: str) -> bool
        INPUTS:
            - model: str — Model identifier to check
        OUTPUT: bool — Whether this model is currently rate limited
        WHAT IT DOES:
            Returns model in self._rate_limited and time.time() < self._rate_limited[model]

    def reset_usage(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            Resets all token counters and rate limit states.
            Called on session reset.

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - LLMClient        (class)
    - MODEL_CONFIGS    (dict constant)
═══════════════════════════════════════════════════════════════════════════════════

GROQ RATE LIMITS (Free Tier — as of Feb 2026):
    llama-3.1-8b-instant:       30 req/min, 6,000 tokens/min, 500K tokens/day
    llama-4-scout-17b:          15 req/min, 30,000 tokens/min*
    kimi-k2-instruct:           30 req/min, 10,000 req/day, 300K tokens/day
    qwen3-32b:                  30 req/min, 6,000 tokens/min, 500K tokens/day
    gpt-oss-120b:               ~8K tokens/day (very limited)
    llama-guard-4-12b:          30 req/min, 15,000 tokens/min

    The fast brain (llama-3.1-8b) handles ~30 conversational turns per minute,
    which is more than enough for natural speech (humans speak ~5-10 turns/min).
"""
