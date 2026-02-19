"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/short_term.py                              ║
║            SHORT-TERM MEMORY — ROLLING WINDOW OF RECENT CONVERSATION            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Maintains a fixed-size rolling window of the most recent conversation turns.
    This is the primary context source for the fast LLM path.
    Uses a double-ended queue (deque) with a configurable max size (default: 20).
    When the window fills, the oldest turn is popped and optionally forwarded
    to the compressor for long-term storage.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging
from collections import deque
from typing import Optional
from copy import deepcopy

from core.event_bus import EventBus, EventType
from core.session import Turn

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_TURNS = 20          # Maximum turns in rolling window
COMPRESS_THRESHOLD = 15         # When window hits this, start compressing oldest

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: ShortTermMemory
──────────────────────────────────────────────────────────────────────────────────
    Rolling window of recent conversation turns.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, max_turns: int = DEFAULT_MAX_TURNS)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For subscribing to TURN_COMPLETE and publishing
              MEMORY_COMPRESS events
            - max_turns: int — Maximum number of turns to keep (default 20)
        INITIALIZES:
            self._bus = event_bus
            self._max = max_turns
            self._window: deque[Turn] = deque(maxlen=max_turns)
            self._overflow_buffer: list[Turn] = []  # Turns evicted from window
            self._logger = logging.getLogger("ShortTermMemory")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Subscribe to EventType.TURN_COMPLETE → self._on_turn_complete
            2. Subscribe to EventType.SESSION_RESET → self.clear
            3. Log "ShortTermMemory listening for turns"
            4. Block forever (await asyncio.Event().wait()) — actual work
               happens via event callbacks

    async def _on_turn_complete(self, event) -> None
        INPUTS:
            - event: Event — Payload contains Turn object at event.data["turn"]
        OUTPUT: None
        WHAT IT DOES:
            1. Extract turn from event.data["turn"]
            2. If len(self._window) == self._max:
               - evicted = self._window[0]  (oldest, about to be popped by deque)
               - self._overflow_buffer.append(evicted)
            3. self._window.append(turn)
            4. If len(self._overflow_buffer) >= 5:
               - Publish EventType.MEMORY_COMPRESS with
                 data={"turns": self._overflow_buffer.copy()}
               - self._overflow_buffer.clear()
            5. Log turn count

    def get_recent(self, n: int = None) -> list[Turn]
        INPUTS:
            - n: int | None — Number of most recent turns (default: all in window)
        OUTPUT:
            - list[Turn] — Deep copies of the N most recent turns
        WHAT IT DOES:
            1. If n is None or n >= len(self._window):
               return list(deepcopy(self._window))
            2. return list(deepcopy(self._window))[-n:]

    def get_context_string(self, n: int = None) -> str
        INPUTS:
            - n: int | None — Number of turns
        OUTPUT:
            - str — Formatted conversation history string
        WHAT IT DOES:
            1. turns = self.get_recent(n)
            2. For each turn, format as "{role}: {text}"
            3. Join with "\n"
            4. Return formatted string

    def clear(self, event=None) -> None
        INPUTS:
            - event: optional (ignored) — Allows use as event callback
        OUTPUT: None
        WHAT IT DOES:
            1. self._window.clear()
            2. self._overflow_buffer.clear()
            3. Log "Short-term memory cleared"

    @property
    def size(self) -> int
        OUTPUT: int — Current number of turns in window
        WHAT IT DOES:
            return len(self._window)

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - ShortTermMemory   (class)
═══════════════════════════════════════════════════════════════════════════════════
"""
