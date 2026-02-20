"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — safety/guard.py                                   ║
║           SAFETY GUARD — ASYNC INPUT/OUTPUT FILTERING VIA LLAMA GUARD           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Provides async content safety filtering for both user inputs and AI outputs.
    Uses Groq's Llama-Guard-4-12b model to classify text against Meta's safety
    taxonomy. Runs asynchronously and publishes SAFETY_FLAG events when unsafe
    content is detected — does NOT block the pipeline.

    Two modes:
    - Input filtering: Checks user speech transcripts in real-time
    - Output filtering: Checks AI responses before/during TTS

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from groq import AsyncGroq                  # Groq async client

from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

GUARD_MODEL = "meta-llama/llama-guard-4-12b"    # Llama Guard 4 on Groq
MAX_RETRIES = 2                                  # Retry on API failures
TIMEOUT_SECONDS = 3.0                            # Max wait for safety check
MIN_TEXT_LENGTH = 5                               # Skip very short texts

# Categories from Meta's safety taxonomy
SAFETY_CATEGORIES = [
    "S1: Violent Crimes",
    "S2: Non-Violent Crimes",
    "S3: Sex-Related Crimes",
    "S4: Child Sexual Exploitation",
    "S5: Defamation",
    "S6: Specialized Advice",
    "S7: Privacy",
    "S8: Intellectual Property",
    "S9: Indiscriminate Weapons",
    "S10: Hate",
    "S11: Suicide & Self-Harm",
    "S12: Sexual Content",
    "S13: Elections",
    "S14: Code Interpreter Abuse",
]

═══════════════════════════════════════════════════════════════════════════════════
ENUMS:
═══════════════════════════════════════════════════════════════════════════════════

class SafetyVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"       # API failure — fail-open
    SKIPPED = "skipped"   # Text too short

═══════════════════════════════════════════════════════════════════════════════════
DATACLASSES:
═══════════════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyResult:
    verdict: SafetyVerdict          # safe / unsafe / error / skipped
    categories: list[str] = field(default_factory=list)  # Violated categories
    raw_output: str = ""            # Raw model output
    direction: str = "input"        # "input" or "output"
    original_text: str = ""         # The text that was checked
    latency_ms: float = 0.0        # Time taken for the check

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: SafetyGuard
──────────────────────────────────────────────────────────────────────────────────
    Async safety filtering using Llama Guard via Groq.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, api_key: str,
                          enabled: bool = True)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For publishing SAFETY_FLAG events
            - api_key: str — Groq API key
            - enabled: bool — Master on/off switch (default True)
        INITIALIZES:
            self._bus = event_bus
            self._enabled = enabled
            self._client = AsyncGroq(api_key=api_key) if enabled else None
            self._stats = {"checked": 0, "flagged": 0, "errors": 0, "skipped": 0}
            self._logger = logging.getLogger("SafetyGuard")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. If not self._enabled:
               Log "SafetyGuard disabled"; return
            2. Subscribe to EventType.TRANSCRIPT → self._check_input
            3. Subscribe to EventType.LLM_SENTENCE → self._check_output
            4. Log "SafetyGuard active with Llama Guard 4"
            5. Block with asyncio.Event().wait()

    async def _check_input(self, event) -> None
        INPUTS:
            - event: Event — data["text"] contains user transcript
        OUTPUT: None
        WHAT IT DOES:
            1. text = event.data.get("text", "")
            2. result = await self.check(text, direction="input")
            3. If result.verdict == SafetyVerdict.UNSAFE:
               Publish EventType.SAFETY_FLAG with data={
                   "direction": "input",
                   "text": text,
                   "categories": result.categories,
                   "verdict": result.verdict.value
               }

    async def _check_output(self, event) -> None
        INPUTS:
            - event: Event — data["text"] contains AI-generated sentence
        OUTPUT: None
        WHAT IT DOES:
            1. text = event.data.get("text", "")
            2. result = await self.check(text, direction="output")
            3. If result.verdict == SafetyVerdict.UNSAFE:
               Publish EventType.SAFETY_FLAG with data={
                   "direction": "output",
                   "text": text,
                   "categories": result.categories,
                   "verdict": result.verdict.value
               }

    async def check(self, text: str, direction: str = "input") -> SafetyResult
        INPUTS:
            - text: str — Text to evaluate
            - direction: str — "input" (user text) or "output" (AI text)
        OUTPUT:
            - SafetyResult — Verdict with details
        WHAT IT DOES:
            1. If not self._enabled:
               return SafetyResult(verdict=SafetyVerdict.SKIPPED)
            2. If len(text.strip()) < MIN_TEXT_LENGTH:
               self._stats["skipped"] += 1
               return SafetyResult(verdict=SafetyVerdict.SKIPPED, direction=direction)
            3. self._stats["checked"] += 1
            4. start_time = time.monotonic()
            5. Build messages for Llama Guard:
               If direction == "input":
                   messages = [
                       {"role": "user", "content": text}
                   ]
               Else:
                   messages = [
                       {"role": "user", "content": "Previous user message"},
                       {"role": "assistant", "content": text}
                   ]
            6. For attempt in range(MAX_RETRIES):
               Try:
                   response = await asyncio.wait_for(
                       self._client.chat.completions.create(
                           model=GUARD_MODEL,
                           messages=messages,
                           temperature=0.0,
                           max_tokens=100
                       ),
                       timeout=TIMEOUT_SECONDS
                   )
                   raw = response.choices[0].message.content.strip()
                   result = self._parse_verdict(raw, direction, text)
                   result.latency_ms = (time.monotonic() - start_time) * 1000
                   If result.verdict == SafetyVerdict.UNSAFE:
                       self._stats["flagged"] += 1
                   return result
               Except asyncio.TimeoutError:
                   Log warning "Safety check timed out"
               Except Exception as e:
                   Log error f"Safety check failed: {e}"
            7. self._stats["errors"] += 1
            8. return SafetyResult(verdict=SafetyVerdict.ERROR, direction=direction,
                                   original_text=text)
               # Fail-open: errors don't block content

    def _parse_verdict(self, raw_output: str, direction: str,
                       original_text: str) -> SafetyResult
        INPUTS:
            - raw_output: str — Raw Llama Guard output (e.g., "safe" or "unsafe\nS1")
            - direction: str — "input" or "output"
            - original_text: str — The checked text
        OUTPUT:
            - SafetyResult — Parsed result
        WHAT IT DOES:
            1. lines = raw_output.strip().lower().split("\n")
            2. verdict_str = lines[0].strip()
            3. If verdict_str == "safe":
               return SafetyResult(
                   verdict=SafetyVerdict.SAFE,
                   raw_output=raw_output,
                   direction=direction,
                   original_text=original_text
               )
            4. # Unsafe — extract categories
               categories = []
               For line in lines[1:]:
                   stripped = line.strip()
                   For cat in SAFETY_CATEGORIES:
                       If cat.lower().startswith(stripped):
                           categories.append(cat)
                           break
            5. return SafetyResult(
                   verdict=SafetyVerdict.UNSAFE,
                   categories=categories,
                   raw_output=raw_output,
                   direction=direction,
                   original_text=original_text
               )

    def get_stats(self) -> dict
        INPUTS: None
        OUTPUT: dict — Copy of internal stats counter
        WHAT IT DOES:
            return dict(self._stats)

    @property
    def is_enabled(self) -> bool
        OUTPUT: bool
        WHAT IT DOES:
            return self._enabled

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - SafetyGuard    (class)
    - SafetyVerdict  (enum)
    - SafetyResult   (dataclass)
═══════════════════════════════════════════════════════════════════════════════════

NOTES:
    - Fail-open design: API errors or timeouts do NOT block content
    - Async and non-blocking — safety checks run in parallel with pipeline
    - Llama Guard 4 is free on Groq with generous rate limits
    - SAFETY_FLAG events are consumed by TurnManager to optionally suppress
      AI responses or warn the user
    - Input checks run on user transcripts; output checks run before TTS
    - Stats tracking for monitoring and debugging
"""



import logging
import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from groq import AsyncGroq

from core.event_bus import EventBus, EventType

# ─── Constants ────────────────────────────────────────────────────────────────

GUARD_MODEL = "meta-llama/llama-guard-4-12b"
MAX_RETRIES = 2
TIMEOUT_SECONDS = 3.0
MIN_TEXT_LENGTH = 5

SAFETY_CATEGORIES = [
    "S1: Violent Crimes",
    "S2: Non-Violent Crimes",
    "S3: Sex-Related Crimes",
    "S4: Child Sexual Exploitation",
    "S5: Defamation",
    "S6: Specialized Advice",
    "S7: Privacy",
    "S8: Intellectual Property",
    "S9: Indiscriminate Weapons",
    "S10: Hate",
    "S11: Suicide & Self-Harm",
    "S12: Sexual Content",
    "S13: Elections",
    "S14: Code Interpreter Abuse",
]


# ─── Enum ─────────────────────────────────────────────────────────────────────

class SafetyVerdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    ERROR = "error"
    SKIPPED = "skipped"


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    verdict: SafetyVerdict
    categories: list[str] = field(default_factory=list)
    raw_output: str = ""
    direction: str = "input"
    original_text: str = ""
    latency_ms: float = 0.0


# ─── SafetyGuard ──────────────────────────────────────────────────────────────

class SafetyGuard:
    """Async safety filtering using Llama Guard via Groq."""

    def __init__(self, event_bus: EventBus, api_key: str, enabled: bool = True) -> None:
        self._bus: EventBus = event_bus
        self._enabled: bool = enabled
        self._client: Optional[AsyncGroq] = AsyncGroq(api_key=api_key) if enabled else None
        self._stats: dict[str, int] = {"checked": 0, "flagged": 0, "errors": 0, "skipped": 0}
        self._logger: logging.Logger = logging.getLogger("SafetyGuard")

    async def run(self) -> None:
        """Subscribe to transcript and LLM sentence events, block forever."""
        if not self._enabled:
            self._logger.info("SafetyGuard disabled")
            return

        transcript_queue: asyncio.Queue = self._bus.subscribe(EventType.TRANSCRIPT_READY)
        llm_sentence_queue: asyncio.Queue = self._bus.subscribe(EventType.LLM_STREAM_DONE)

        self._logger.info("SafetyGuard active with Llama Guard 4")

        async def _input_loop() -> None:
            while True:
                event = await transcript_queue.get()
                await self._check_input(event)

        async def _output_loop() -> None:
            while True:
                event = await llm_sentence_queue.get()
                await self._check_output(event)

        await asyncio.gather(_input_loop(), _output_loop())

    async def _check_input(self, event) -> None:
        """Check user transcript for safety violations."""
        text: str = event.data.get("text", "")
        result = await self.check(text, direction="input")
        if result.verdict == SafetyVerdict.UNSAFE:
            await self._bus.publish(
                EventType.SAFETY_FLAGGED,
                {
                    "direction": "input",
                    "text": text,
                    "categories": result.categories,
                    "verdict": result.verdict.value,
                },
                source="SafetyGuard",
            )

    async def _check_output(self, event) -> None:
        """Check AI-generated sentence for safety violations."""
        text: str = event.data.get("text", "")
        result = await self.check(text, direction="output")
        if result.verdict == SafetyVerdict.UNSAFE:
            await self._bus.publish(
                EventType.SAFETY_FLAGGED,
                {
                    "direction": "output",
                    "text": text,
                    "categories": result.categories,
                    "verdict": result.verdict.value,
                },
                source="SafetyGuard",
            )

    async def check(self, text: str, direction: str = "input") -> SafetyResult:
        """Evaluate text against Llama Guard safety taxonomy."""
        if not self._enabled:
            return SafetyResult(verdict=SafetyVerdict.SKIPPED)

        if len(text.strip()) < MIN_TEXT_LENGTH:
            self._stats["skipped"] += 1
            return SafetyResult(verdict=SafetyVerdict.SKIPPED, direction=direction)

        self._stats["checked"] += 1
        start_time = time.monotonic()

        # Build messages for Llama Guard
        if direction == "input":
            messages = [{"role": "user", "content": text}]
        else:
            messages = [
                {"role": "user", "content": "Previous user message"},
                {"role": "assistant", "content": text},
            ]

        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=GUARD_MODEL,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=100,
                    ),
                    timeout=TIMEOUT_SECONDS,
                )
                raw: str = response.choices[0].message.content.strip()
                result = self._parse_verdict(raw, direction, text)
                result.latency_ms = (time.monotonic() - start_time) * 1000

                if result.verdict == SafetyVerdict.UNSAFE:
                    self._stats["flagged"] += 1

                return result

            except asyncio.TimeoutError:
                self._logger.warning(
                    "Safety check timed out (attempt %d/%d)", attempt + 1, MAX_RETRIES
                )
            except Exception as e:
                self._logger.error(
                    "Safety check failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES, e
                )

        # All retries exhausted — fail open
        self._stats["errors"] += 1
        return SafetyResult(
            verdict=SafetyVerdict.ERROR,
            direction=direction,
            original_text=text,
        )

    def _parse_verdict(
        self, raw_output: str, direction: str, original_text: str
    ) -> SafetyResult:
        """Parse raw Llama Guard output into a SafetyResult."""
        lines = raw_output.strip().lower().split("\n")
        verdict_str = lines[0].strip()

        if verdict_str == "safe":
            return SafetyResult(
                verdict=SafetyVerdict.SAFE,
                raw_output=raw_output,
                direction=direction,
                original_text=original_text,
            )

        # Unsafe — extract violated categories
        categories: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            for cat in SAFETY_CATEGORIES:
                if cat.lower().startswith(stripped):
                    categories.append(cat)
                    break

        return SafetyResult(
            verdict=SafetyVerdict.UNSAFE,
            categories=categories,
            raw_output=raw_output,
            direction=direction,
            original_text=original_text,
        )

    def get_stats(self) -> dict:
        """Return a copy of internal stats counters."""
        return dict(self._stats)

    @property
    def is_enabled(self) -> bool:
        """Whether the safety guard is active."""
        return self._enabled
