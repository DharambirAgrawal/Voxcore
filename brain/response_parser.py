"""
VOXCORE — brain/response_parser.py
Splits streaming LLM output into speech tokens + agent JSON in real-time.
"""

import logging
import json
import re

from typing import AsyncGenerator
from core.event_bus import EventBus, EventType

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
AGENT_OPEN_TAG = "<agent>"
AGENT_CLOSE_TAG = "</agent>"
EMOTION_TAG_PATTERN = re.compile(r'\[(\w+)\]')

# ── Special paralinguistic tags ──────────────────────────────────────
# Tags that map to pre-recorded audio clips instead of TTS synthesis
CLIP_TAGS = frozenset({"laughs", "chuckles", "light_laugh", "sighs"})
PAUSE_TAG = "..."  # [...] → 300ms silence insertion

# ── Session cache reference tags ────────────────────────────────────
# [ref:sc_XXXX] tags are kept in working memory but stripped from TTS output
REF_TAG_PATTERN = re.compile(r'\s*\[ref:sc_[a-f0-9_]+\]')


class ResponseParser:
    """Real-time LLM output parser that splits speech and agent output."""

    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus: EventBus = event_bus

        # Speech buffer state
        self._speech_buffer: str = ""
        self._sentence_index: int = 0
        self._current_emotion: str = ""

        # Agent buffer state
        self._agent_buffer: str = ""
        self._in_agent_tag: bool = False

        # Full response tracking
        self._full_response: str = ""
        self._is_first_token: bool = True

        self._logger: logging.Logger = logging.getLogger("ResponseParser")

    # ───────────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────────

    async def parse_stream(
        self,
        token_stream: AsyncGenerator[str, None],
        suppress_stream_done_if_empty: bool = False,
    ) -> str:
        """Consume streaming tokens, fire speech/agent events, return full response.

        Args:
            suppress_stream_done_if_empty: When True, skip publishing LLM_STREAM_DONE
                if no tokens were received (zero-length response).  Use this when the
                caller wants to retry with a fallback model before committing to done.
        """
        self._reset()

        async for token in token_stream:
            self._full_response += token

            if self._is_first_token:
                self._is_first_token = False

            await self._process_token(token)

        # Flush any remaining speech
        await self._flush_speech_buffer()

        # Signal stream completion — suppressed on empty if caller asks for it
        if self._full_response or not suppress_stream_done_if_empty:
            await self.event_bus.publish(EventType.LLM_STREAM_DONE, {})

        return self._full_response

    # ───────────────────────────────────────────────────────────────────────
    # Internal processing
    # ───────────────────────────────────────────────────────────────────────

    async def _process_token(self, token: str) -> None:
        # CASE 1: Currently inside <agent> block
        if self._in_agent_tag:
            self._agent_buffer += token

            if AGENT_CLOSE_TAG in self._agent_buffer:
                parts = self._agent_buffer.split(AGENT_CLOSE_TAG, 1)
                raw = parts[0]
                after_tag = parts[1] if len(parts) > 1 else ""

                try:
                    parsed = json.loads(raw)
                    await self.event_bus.publish(EventType.LLM_AGENT_TAG, {
                        "action": parsed.get("action"),
                        "params": parsed.get("params", {}),
                        "raw": raw,
                    })
                    self._logger.info("Agent tag parsed: action=%s", parsed.get("action"))
                except json.JSONDecodeError:
                    self._logger.error("Failed to parse agent JSON: %s", raw)

                self._in_agent_tag = False
                self._agent_buffer = ""

                # Process any text after </agent> as speech
                if after_tag.strip():
                    self._speech_buffer += after_tag
                    await self._check_sentence_boundaries()

            return

        # CASE 2 & 3: Not in agent block — accumulate into speech buffer
        self._speech_buffer += token

        # Check for <agent> tag start
        if AGENT_OPEN_TAG in self._speech_buffer:
            parts = self._speech_buffer.split(AGENT_OPEN_TAG, 1)
            text_before = parts[0]
            text_after = parts[1] if len(parts) > 1 else ""

            # Flush speech before the agent tag
            if text_before.strip():
                old_buffer = self._speech_buffer
                self._speech_buffer = text_before
                await self._flush_speech_buffer()

            self._in_agent_tag = True
            self._agent_buffer = text_after
            self._speech_buffer = ""
            return

        # Normal speech — check for sentence boundaries
        await self._check_sentence_boundaries()

    async def _check_sentence_boundaries(self) -> None:
        """Split buffer on sentence endings and emit complete sentences."""
        parts = SENTENCE_ENDINGS.split(self._speech_buffer)

        if len(parts) > 1:
            # All parts except last are complete sentences
            for sentence in parts[:-1]:
                await self._emit_sentence(sentence)
            # Keep the remainder (incomplete sentence) in buffer
            self._speech_buffer = parts[-1]

    async def _emit_sentence(self, sentence: str) -> None:
        """Fire an LLM_SPEECH_TOKEN event for a complete sentence.

        V2: Also detects [laughs], [chuckles], [sighs], [light_laugh] tags
        and fires PLAY_CLIP events instead of sending them to TTS.
        Also detects [...] and fires PAUSE_MARKER events.
        """
        sentence = sentence.strip()
        if not sentence:
            return

        # ── Guard: drop hallucinated conversation-turn lines ─────────────
        # The LLM sometimes copies the few-shot example format from the
        # system prompt and generates spurious "User: X" or "You: Y" lines.
        # These must NEVER reach TTS.
        _lower = sentence.lower()
        if (_lower.startswith(("user:", "you:", "human:", "assistant:"))
                or _lower.startswith(("[user]", "[you]", "[human]", "[assistant]"))):
            self._logger.warning(
                "Dropped hallucinated turn line: '%s'", sentence[:120]
            )
            return

        # ── Check for pause markers [...] ────────────────────────────
        # Split sentence around [...] and process each part
        parts = sentence.split("[...]")
        if len(parts) > 1:
            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    await self._emit_sentence(part)  # recurse for the text part
                if i < len(parts) - 1:
                    # Insert a pause marker between parts
                    await self.event_bus.publish(EventType.PAUSE_MARKER, {
                        "duration_ms": 300,
                        "sentence_index": self._sentence_index,
                    })
                    self._logger.debug("Pause marker [...] at sentence %d", self._sentence_index)
            return

        # ── Check for clip tags [laughs], [chuckles], etc. ───────────
        # Pattern: sentence might be just "[laughs]" or contain it inline
        clip_pattern = re.compile(r'\[(laughs|chuckles|light_laugh|sighs)\]')
        clip_match = clip_pattern.search(sentence)
        if clip_match:
            clip_tag = clip_match.group(1)
            # Text before the clip tag
            text_before = sentence[:clip_match.start()].strip()
            # Text after the clip tag
            text_after = sentence[clip_match.end():].strip()

            # Emit text before the clip (if any)
            if text_before:
                await self._emit_speech_token(text_before)

            # Fire PLAY_CLIP event for the clip
            await self.event_bus.publish(EventType.PLAY_CLIP, {
                "clip_name": clip_tag,
                "sentence_index": self._sentence_index,
            })
            self._logger.debug("Clip tag [%s] at sentence %d", clip_tag, self._sentence_index)

            # Emit text after the clip (if any)
            if text_after:
                await self._emit_speech_token(text_after)
            return

        # ── Normal sentence (no special tags) ────────────────────────────
        await self._emit_speech_token(sentence)

    async def _emit_speech_token(self, sentence: str) -> None:
        """Fire an LLM_SPEECH_TOKEN event for a normal speech sentence."""
        sentence = sentence.strip()
        if not sentence:
            return

        # ── Strip [ref:sc_...] tags from TTS output ─────────────────
        # These tags are kept in self._full_response (stored in working memory)
        # but must NOT be spoken aloud.
        tts_text = REF_TAG_PATTERN.sub("", sentence).strip()
        if not tts_text:
            return

        # Extract emotion tag if present
        match = EMOTION_TAG_PATTERN.search(tts_text)
        if match:
            tag = match.group(1)
            # Only set emotion if it's an actual emotion, not a clip tag
            if tag not in CLIP_TAGS and tag != PAUSE_TAG:
                self._current_emotion = tag

        await self.event_bus.publish(EventType.LLM_SPEECH_TOKEN, {
            "text": tts_text,
            "sentence_index": self._sentence_index,
            "emotion": self._current_emotion,
        })

        self._logger.debug("Sentence %d: '%s'",
                           self._sentence_index,
                           sentence[:60] + "..." if len(sentence) > 60 else sentence)
        self._sentence_index += 1

    async def _flush_speech_buffer(self) -> None:
        """Emit whatever remains in the speech buffer as a final sentence."""
        if self._speech_buffer.strip():
            await self._emit_sentence(self._speech_buffer)
            self._speech_buffer = ""

    def _reset(self) -> None:
        """Reset all state for a new response."""
        self._speech_buffer = ""
        self._agent_buffer = ""
        self._in_agent_tag = False
        self._sentence_index = 0
        self._current_emotion = ""
        self._full_response = ""
        self._is_first_token = True