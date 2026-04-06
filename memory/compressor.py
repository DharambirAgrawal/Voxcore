"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/compressor.py                              ║
║           MEMORY COMPRESSOR — SUMMARIZE OVERFLOW TURNS FOR STORAGE              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    When the short-term memory window overflows (compress_after_turns reached),
    this module summarises the conversation turns via llm_fast and stores
    the summary into ChromaDB long-term memory with proper type/category tags.

    Critical fix (v5): after compression, only the last 3 turns are retained
    in working memory. Without this, compression fires on EVERY subsequent
    turn and ChromaDB fills with duplicate summaries.

    Also filters noise: turns under 5 words or filler-only are excluded
    from the summary input.

═══════════════════════════════════════════════════════════════════════════════════
NOTES:
    - Uses models.llm_fast from config for summarization (cheap, runs every 20 turns)
    - Temperature 0.3 for consistent, factual summaries
    - Compression runs as background task — doesn't block conversation
    - CRITICAL FIX (v5): After compressing, only last 3 turns are retained
      in working memory. Without this, compression fires on every subsequent
      turn and ChromaDB fills with duplicate summaries.
    - Noise filter: turns under 5 words and filler-only responses are excluded
"""


import logging
import asyncio
from datetime import datetime
from typing import Optional

from groq import AsyncGroq

from core.event_bus import EventBus, EventType
from core.session import Turn

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_SUMMARY_TOKENS = 300

# Filler responses that should never be stored in long-term memory
FILLER_RESPONSES = frozenset({
    "okay", "yes", "no", "thanks", "got it", "sure",
    "right", "yeah", "alright", "fine", "understood",
    "yep", "nope", "cool", "nice", "great", "good",
    "hmm", "uh huh", "mm hmm", "oh", "ah",
})

COMPRESS_SYSTEM_PROMPT = """Summarize this conversation excerpt in 2-3 sentences.
Third person. Facts only. No filler. No AI speech.
Do not include greetings, filler words, or one-word responses.

CRITICAL — preserve these EXACTLY as they appear (do NOT paraphrase):
- Full URLs (https://...)
- Names of people, places, companies
- Numbers, dates, code snippets
- Any explicit user preference or instruction
If a URL or name was mentioned, it MUST appear verbatim in the summary."""


class MemoryCompressor:
    """Summarizes overflow conversation turns and stores to long-term memory.

    v5: Works directly with short_term and long_term stores.
    Critical fix: clears working memory after compression (keeps last 3 turns).
    """

    def __init__(
        self,
        event_bus: EventBus,
        short_term,
        long_term,
        config: dict,
    ) -> None:
        self._bus: EventBus = event_bus
        self._short_term = short_term
        self._long_term = long_term
        self._config: dict = config
        self._model: str = config.get("models", {}).get("llm_fast", "llama-3.1-8b-instant")
        self._compress_after: int = config.get("memory", {}).get("compress_after_turns", 20)
        self._is_compressing: bool = False
        self._groq = AsyncGroq(max_retries=0)
        self._logger: logging.Logger = logging.getLogger("MemoryCompressor")

    async def run(self) -> None:
        """Subscribe to TURN_COMPLETE and check if compression is needed."""
        self._bus.subscribe(EventType.TURN_COMPLETE, self._on_turn_complete)
        self._logger.info("MemoryCompressor ready (threshold=%d turns)", self._compress_after)
        await asyncio.Event().wait()

    async def _on_turn_complete(self, event) -> None:
        """Check if compression should trigger after each turn."""
        if self._is_compressing:
            return

        current_count = len(self._short_term)
        if current_count >= self._compress_after:
            self._logger.info(
                "Compression triggered: %d turns >= %d threshold",
                current_count, self._compress_after,
            )
            asyncio.create_task(self._compress())

    async def _compress(self) -> None:
        """Compress current working memory, store summary, keep last 3 turns."""
        self._is_compressing = True
        try:
            turns = self._short_term.get_recent()
            if not turns:
                return

            # Filter noise — only include turns with real content
            meaningful = [
                t for t in turns
                if len(t.content.split()) > 5
                and t.content.lower().strip() not in FILLER_RESPONSES
            ]

            if meaningful:
                summary = await self._build_summary_llm(meaningful)
                if summary:
                    await self._long_term.store(
                        text=summary,
                        memory_type="episodic",
                        category="event",
                        importance=0.4,
                        metadata={
                            "turn_count": len(turns),
                            "compressed_at": datetime.now().isoformat(),
                        },
                    )

            # ✅ THE FIX — keep last 6 turns, clear the rest
            # (was 3, but that's too aggressive — URLs/names shared 4 turns
            #  ago were being dropped before the LLM could reference them)
            keep_count = 6
            all_turns = list(self._short_term._window)
            self._short_term._window.clear()
            self._short_term._overflow_buffer.clear()
            for turn in all_turns[-keep_count:]:
                self._short_term._window.append(turn)

            self._logger.info(
                "Compressed. Kept last %d of %d turns.", keep_count, len(turns)
            )

        except Exception as e:
            self._logger.error("Compression failed: %s", e)
        finally:
            self._is_compressing = False

    async def _build_summary_llm(self, turns: list[Turn]) -> Optional[str]:
        """Summarise turns using llm_fast via Groq."""
        formatted = self._format_turns(turns)
        try:
            response = await self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": COMPRESS_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Conversation:\n{formatted}"},
                ],
                max_tokens=MAX_SUMMARY_TOKENS,
                temperature=0.3,
            )
            text = response.choices[0].message.content
            return text.strip() if text else None
        except Exception as e:
            self._logger.error("Summary LLM call failed: %s", e)
            return None

    async def compress_session_end(self) -> None:
        """Called from session shutdown — compress remaining turns."""
        remaining = self._short_term.get_recent()
        if len(remaining) > 2:
            await self._compress()

    @staticmethod
    def _format_turns(turns: list[Turn]) -> str:
        """Format turns into a readable conversation transcript."""
        lines: list[str] = []
        for turn in turns:
            lines.append(f"[{turn.role}]: {turn.content}")
        return "\n".join(lines)

    @property
    def is_busy(self) -> bool:
        """Whether a compression is currently in progress."""
        return self._is_compressing