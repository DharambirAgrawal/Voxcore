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
