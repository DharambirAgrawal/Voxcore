"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/compressor.py                              ║
║           MEMORY COMPRESSOR — SUMMARIZE OVERFLOW TURNS FOR STORAGE              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    When the short-term memory window overflows, evicted turns are batched
    and sent to the compressor. This module uses a deep-thinking LLM
    (qwen3-32b via Groq) to summarize batches of old turns into concise
    summaries, which are then stored in long-term ChromaDB memory.

    This keeps context efficient: instead of storing raw turns, compressed
    summaries capture key information at a fraction of the token cost.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging
import asyncio
from typing import Optional

from core.event_bus import EventBus, EventType
from core.session import Turn
from brain.llm_client import LLMClient

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

COMPRESS_MODEL = "qwen3-32b"                # Deep thinking model for summarization
MAX_SUMMARY_TOKENS = 300                    # Max tokens for the summary output
BATCH_SIZE = 5                              # Minimum turns before compressing

COMPRESS_SYSTEM_PROMPT = '''You are a conversation summarizer. Given a batch of
conversation turns, produce a concise summary capturing:
1. Key facts mentioned by the user (names, preferences, dates)
2. Decisions made or action items
3. Emotional context or tone shifts
4. Any explicit requests to "remember" something

Output ONLY the summary, no preamble. Keep it under 150 words.'''

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: MemoryCompressor
──────────────────────────────────────────────────────────────────────────────────
    Summarizes overflow conversation turns using a deep-thinking LLM.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, llm_client: LLMClient)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For subscribing to MEMORY_COMPRESS events
              and publishing results back with summary attached
            - llm_client: LLMClient — Shared LLM client for Groq API access
        INITIALIZES:
            self._bus = event_bus
            self._llm = llm_client
            self._pending: list[Turn] = []
            self._is_compressing: bool = False
            self._logger = logging.getLogger("MemoryCompressor")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Subscribe to EventType.MEMORY_COMPRESS → self._on_compress_request
            2. Log "MemoryCompressor ready"
            3. Block forever with asyncio.Event().wait()

    async def _on_compress_request(self, event) -> None
        INPUTS:
            - event: Event — data["turns"] is a list of Turn objects
        OUTPUT: None
        WHAT IT DOES:
            1. turns = event.data.get("turns", [])
            2. self._pending.extend(turns)
            3. If len(self._pending) < BATCH_SIZE: return  # Wait for more
            4. If self._is_compressing: return  # Don't stack compressions
            5. batch = self._pending[:BATCH_SIZE * 2]  # Take up to 2x batch
            6. self._pending = self._pending[len(batch):]
            7. asyncio.create_task(self._compress_batch(batch))

    async def _compress_batch(self, turns: list[Turn]) -> None
        INPUTS:
            - turns: list[Turn] — Batch of conversation turns to summarize
        OUTPUT: None
        WHAT IT DOES:
            1. self._is_compressing = True
            2. formatted = self._format_turns(turns)
            3. Try:
               a. summary = await self._llm.complete(
                      model=COMPRESS_MODEL,
                      system_prompt=COMPRESS_SYSTEM_PROMPT,
                      user_prompt=formatted,
                      max_tokens=MAX_SUMMARY_TOKENS,
                      temperature=0.3
                  )
               b. If summary and summary.strip():
                  Publish EventType.MEMORY_STORE with data={
                      "summary": summary.strip(),
                      "turns": turns,
                      "turn_count": len(turns)
                  }
                  Log f"Compressed {len(turns)} turns into {len(summary)} chars"
               c. Else:
                  Log warning "Empty summary from compressor"
            4. Except Exception as e:
               Log error f"Compression failed: {e}"
               # Put turns back into pending for retry
               self._pending = turns + self._pending
            5. Finally:
               self._is_compressing = False

    def _format_turns(self, turns: list[Turn]) -> str
        INPUTS:
            - turns: list[Turn] — Turns to format for the LLM prompt
        OUTPUT:
            - str — Formatted conversation text
        WHAT IT DOES:
            1. lines = []
            2. For each turn in turns:
               lines.append(f"[{turn.role}]: {turn.text}")
            3. Return "\n".join(lines)

    def get_pending_count(self) -> int
        INPUTS: None
        OUTPUT: int — Number of turns waiting to be compressed
        WHAT IT DOES:
            return len(self._pending)

    @property
    def is_busy(self) -> bool
        OUTPUT: bool — Whether a compression is in progress
        WHAT IT DOES:
            return self._is_compressing

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - MemoryCompressor   (class)
═══════════════════════════════════════════════════════════════════════════════════

NOTES:
    - Uses qwen3-32b for high-quality summarization (deep thinking)
    - Temperature 0.3 for consistent, factual summaries
    - Compression runs as background task — doesn't block conversation
    - Failed compressions re-queue turns for retry
    - The MEMORY_STORE event is consumed by LongTermMemory.store()
"""
