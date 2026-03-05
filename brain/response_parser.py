"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — brain/response_parser.py                          ║
║     SPLITS STREAMING LLM OUTPUT INTO SPEECH TOKENS + AGENT JSON IN REAL-TIME   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Consumes the streaming token output from LLMClient and maintains TWO output
    buffers simultaneously:
    
    1. Speech buffer — accumulates tokens for TTS. Fires LLM_SPEECH_TOKEN events
       sentence-by-sentence (splits on . ? ! \n). Each event triggers TTS for
       that sentence immediately — TTS starts BEFORE the LLM finishes.
    
    2. Agent buffer — watches for <agent> opening tag, captures everything until
       </agent>, parses the content as JSON, and fires LLM_AGENT_TAG event.
    
    Both buffers flush in real-time. This is what enables the "speak while
    still generating" behavior that makes VoxCore feel responsive.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import json                             # Parse agent JSON
import re                               # Regex for emotion tag extraction and sentence splitting

from typing import AsyncGenerator, Optional
from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')   # Split on sentence boundaries
AGENT_OPEN_TAG = "<agent>"
AGENT_CLOSE_TAG = "</agent>"
EMOTION_TAG_PATTERN = re.compile(r'\[(\w+)\]')     # Matches [cheerful], [calm], etc.

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: ResponseParser
──────────────────────────────────────────────────────────────────────────────────
    Real-time LLM output parser that splits speech and agent output.

    CONSTRUCTOR: __init__(self, event_bus: EventBus)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For publishing LLM_SPEECH_TOKEN and LLM_AGENT_TAG events
        
        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            
            # Speech buffer state
            self._speech_buffer: str             = ""       # Accumulates tokens until sentence boundary
            self._sentence_index: int            = 0        # Counter for sentence ordering
            self._current_emotion: str           = ""       # Last detected emotion tag
            
            # Agent buffer state
            self._agent_buffer: str              = ""       # Accumulates tokens between <agent>...</agent>
            self._in_agent_tag: bool             = False    # Currently inside <agent> block?
            
            # Full response tracking
            self._full_response: str             = ""       # Complete response for session history
            self._is_first_token: bool           = True     # Track first token for state transition
            
            self._logger: logging.Logger         = logging.getLogger("ResponseParser")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def parse_stream(self, token_stream: AsyncGenerator[str, None]) -> str
        INPUTS:
            - token_stream: AsyncGenerator[str, None] — From LLMClient.stream_response()
        OUTPUT:
            - str — The complete response text (after stream finishes)
        WHAT IT DOES:
            1. Reset all buffers: self._reset()
            2. Async for token in token_stream:
               a. self._full_response += token
               b. If self._is_first_token:
                  self._is_first_token = False
                  (TurnManager detects this to transition to SPEAKING)
               c. await self._process_token(token)
            3. Flush remaining speech buffer: await self._flush_speech_buffer()
            4. Publish LLM_STREAM_DONE event
            5. Return self._full_response

    async def _process_token(self, token: str) -> None
        INPUTS:
            - token: str — A single token from the LLM stream
        OUTPUT: None (fires events as needed)
        WHAT IT DOES:
            CASE 1: Currently inside <agent> block (self._in_agent_tag == True):
                1. self._agent_buffer += token
                2. If AGENT_CLOSE_TAG in self._agent_buffer:
                   a. Extract JSON: raw = self._agent_buffer.split(AGENT_CLOSE_TAG)[0]
                   b. Try: parsed = json.loads(raw)
                   c. Publish LLM_AGENT_TAG event with data:
                      {"action": parsed.get("action"), "params": parsed.get("params", {}), "raw": raw}
                   d. Log: "Agent tag parsed: action={parsed['action']}"
                   e. Reset: self._in_agent_tag = False, self._agent_buffer = ""
                   f. If there's text after </agent>, process it as speech
            
            CASE 2: Not in agent block, but detection of <agent> start:
                1. self._speech_buffer += token
                2. If AGENT_OPEN_TAG in self._speech_buffer:
                   a. Split: text_before = speech_buffer up to <agent>
                   b. Flush text_before as speech (if non-empty)
                   c. Set self._in_agent_tag = True
                   d. self._agent_buffer = everything after <agent>
                   e. self._speech_buffer = ""
            
            CASE 3: Normal speech token:
                1. self._speech_buffer += token
                2. Check for sentence boundaries
                3. If sentence boundary found:
                   a. Split buffer at boundary
                   b. For each complete sentence: await self._emit_sentence(sentence)
                   c. Keep remainder in buffer

    async def _emit_sentence(self, sentence: str) -> None
        INPUTS:
            - sentence: str — A complete sentence to send to TTS
        OUTPUT: None (fires LLM_SPEECH_TOKEN event)
        WHAT IT DOES:
            1. Strip whitespace from sentence
            2. If empty, return
            3. Extract emotion tag: match = EMOTION_TAG_PATTERN.search(sentence)
               If match: self._current_emotion = match.group(1)
            4. Publish LLM_SPEECH_TOKEN event with data:
               {"text": sentence, "sentence_index": self._sentence_index, "emotion": self._current_emotion}
            5. Increment self._sentence_index
            6. Log: "Sentence {index}: '{sentence[:60]}...'"

    async def _flush_speech_buffer(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. If self._speech_buffer has content:
               await self._emit_sentence(self._speech_buffer)
               self._speech_buffer = ""

    def _reset(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            Resets all state for a new response:
            self._speech_buffer = ""
            self._agent_buffer = ""
            self._in_agent_tag = False
            self._sentence_index = 0
            self._current_emotion = ""
            self._full_response = ""
            self._is_first_token = True

    def get_partial_response(self) -> str
        INPUTS: None
        OUTPUT: str — Whatever response has been generated so far
        WHAT IT DOES:
            Returns self._full_response
            Used by TurnManager when an interrupt happens mid-stream
            to save the partial response to session history.

    @property
    def is_first_token_received(self) -> bool
        OUTPUT: bool — Whether at least one token has been received
        WHAT IT DOES:
            Returns not self._is_first_token

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - ResponseParser       (class)
    - SENTENCE_ENDINGS     (compiled regex)
    - AGENT_OPEN_TAG       (str constant)
    - AGENT_CLOSE_TAG      (str constant)
    - EMOTION_TAG_PATTERN  (compiled regex)
═══════════════════════════════════════════════════════════════════════════════════

EXAMPLE LLM OUTPUT AND PARSING:

    Raw LLM output:
        "[concerned] I see what you mean, that does sound frustrating. [calm] Let me check that for you right now. <agent>{\"action\": \"web_search\", \"params\": {\"query\": \"London weather\"}}</agent>"
    
    Events fired:
        1. LLM_SPEECH_TOKEN: {"text": "[concerned] I see what you mean, that does sound frustrating.", "sentence_index": 0, "emotion": "concerned"}
        2. LLM_SPEECH_TOKEN: {"text": "[calm] Let me check that for you right now.", "sentence_index": 1, "emotion": "calm"}
        3. LLM_AGENT_TAG: {"action": "web_search", "params": {"query": "London weather"}, "raw": "..."}
    
    TTS starts playing sentence 0 while the LLM is still generating sentence 1+.
"""



"""
VOXCORE — brain/response_parser.py
Splits streaming LLM output into speech tokens + agent JSON in real-time.
"""

import asyncio
import logging
import json
import re

from typing import AsyncGenerator, Optional
from core.event_bus import EventBus, EventType

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
AGENT_OPEN_TAG = "<agent>"
AGENT_CLOSE_TAG = "</agent>"
EMOTION_TAG_PATTERN = re.compile(r'\[(\w+)\]')

# ── V2: Special paralinguistic tags ──────────────────────────────────────
# Tags that map to pre-recorded audio clips instead of TTS synthesis
CLIP_TAGS = frozenset({"laughs", "chuckles", "light_laugh", "sighs"})
PAUSE_TAG = "..."  # [...] → 300ms silence insertion


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

    async def parse_stream(self, token_stream: AsyncGenerator[str, None]) -> str:
        """Consume streaming tokens, fire speech/agent events, return full response."""
        self._reset()

        async for token in token_stream:
            self._full_response += token

            if self._is_first_token:
                self._is_first_token = False

            await self._process_token(token)

        # Flush any remaining speech
        await self._flush_speech_buffer()

        # Signal stream completion
        await self.event_bus.publish(EventType.LLM_STREAM_DONE, {})

        return self._full_response

    def get_partial_response(self) -> str:
        """Return whatever response has been generated so far (used on interrupt)."""
        return self._full_response

    @property
    def is_first_token_received(self) -> bool:
        """Whether at least one token has been received."""
        return not self._is_first_token

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

        # ── V2: Check for pause markers [...] ────────────────────────────
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

        # ── V2: Check for clip tags [laughs], [chuckles], etc. ───────────
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

        # Extract emotion tag if present
        match = EMOTION_TAG_PATTERN.search(sentence)
        if match:
            tag = match.group(1)
            # Only set emotion if it's an actual emotion, not a clip tag
            if tag not in CLIP_TAGS and tag != PAUSE_TAG:
                self._current_emotion = tag

        await self.event_bus.publish(EventType.LLM_SPEECH_TOKEN, {
            "text": sentence,
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