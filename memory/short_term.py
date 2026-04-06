"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/short_term.py                              ║
║            SHORT-TERM MEMORY — ROLLING WINDOW OF RECENT CONVERSATION            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
from collections import deque
from copy import deepcopy

from core.event_bus import EventBus, EventType
from core.session import Turn

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_TURNS = 20
COMPRESS_THRESHOLD = 15


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: ShortTermMemory
# ═══════════════════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """Rolling window of recent conversation turns."""

    def __init__(self, event_bus: EventBus, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._bus = event_bus
        self._max = max_turns
        self._window: deque[Turn] = deque(maxlen=max_turns)
        self._overflow_buffer: list[Turn] = []
        self._logger = logging.getLogger("ShortTermMemory")

    async def run(self) -> None:
        """Subscribe to events and block forever; work happens via callbacks."""
        self._bus.subscribe(EventType.TURN_COMPLETE, self._on_turn_complete)
        self._bus.subscribe(EventType.SESSION_RESET, self.clear)
        self._logger.info("ShortTermMemory listening for turns")
        await asyncio.Event().wait()

    async def _on_turn_complete(self, event) -> None:
        """Handle a completed turn: append to window, manage overflow."""
        turn = event.data["turn"]

        if len(self._window) == self._max:
            evicted = self._window[0]
            self._overflow_buffer.append(evicted)

        self._window.append(turn)

        if len(self._overflow_buffer) >= 5:
            await self._bus.publish(
                EventType.MEMORY_COMPRESS,
                data={"turns": self._overflow_buffer.copy()},
            )
            self._overflow_buffer.clear()

        self._logger.debug(f"Turn count: {len(self._window)}")

    def get_recent(self, n: int = None) -> list[Turn]:
        """Return deep copies of the N most recent turns."""
        if n is None or n >= len(self._window):
            return list(deepcopy(self._window))
        return list(deepcopy(self._window))[-n:]

    def clear(self, event=None) -> None:
        """Clear the rolling window and overflow buffer."""
        self._window.clear()
        self._overflow_buffer.clear()
        self._logger.info("Short-term memory cleared")

    def __len__(self) -> int:
        """Support len() for the ShortTermMemory."""
        return len(self._window)

    @property
    def size(self) -> int:
        """Current number of turns in the window."""
        return len(self._window)