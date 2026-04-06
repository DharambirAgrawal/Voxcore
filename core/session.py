"""
VoxCore — core/session.py
Master state — single source of truth for the conversation.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class TurnState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    SOFT_INJECT = "soft_inject"
    PENDING = "pending"         # TTS paused, Gate 3 classifying
    PAUSED = "paused"           # TTS frozen in RAM, resumable


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    role: str
    content: str
    timestamp: float
    emotion: Optional[str]
    turn_id: int
    was_interrupted: bool = False
    agent_output: Optional[dict] = None


@dataclass
class TextInEntry:
    content: str
    priority: str = "normal"
    source: str = "unknown"
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION
# ═══════════════════════════════════════════════════════════════════════════════

class Session:
    """Master state container for the entire VoxCore session."""

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.start_time: float = time.time()
        self.turn_count: int = 0
        self.state: TurnState = TurnState.LISTENING

        self.persona_name: str = config["name"]
        self.system_prompt: str = config["system_prompt"]
        self.voice: str = config["voice"]
        self.language: str = config["language"]

        self.history: deque[Turn] = deque(maxlen=50)
        self.text_in_queue: list[TextInEntry] = []
        self.compressed_summary: str = ""

        self.event_bus: EventBus = event_bus
        self._lock: asyncio.Lock = asyncio.Lock()

        # Subscribe to memory compression results to update context
        self.event_bus.subscribe(EventType.MEMORY_COMPRESSED, self._on_memory_compressed)

    async def _on_memory_compressed(self, event) -> None:
        """Update compressed_summary when memory is compressed."""
        summary = event.data.get("summary", "")
        if summary:
            if self.compressed_summary:
                self.compressed_summary += "\n" + summary
            else:
                self.compressed_summary = summary
            logger.info("Session memory updated (%d chars)", len(self.compressed_summary))

    # ── state transitions ──────────────────────────────────────────────────

    async def set_state(self, new_state: TurnState) -> None:
        async with self._lock:
            old = self.state
            self.state = new_state
        await self.event_bus.publish(EventType.STATE_CHANGED, {
            "state": new_state.value,
            "old_state": old.value,
            "new_state": new_state.value,
        })
        logger.info("State: %s → %s", old.value, new_state.value)

    # ── history management ─────────────────────────────────────────────────

    async def add_turn(
        self,
        role: str,
        content: str,
        emotion: Optional[str] = None,
        was_interrupted: bool = False,
        agent_output: Optional[dict] = None,
    ) -> Turn:
        self.turn_count += 1
        turn = Turn(
            role=role,
            content=content,
            timestamp=time.time(),
            emotion=emotion,
            turn_id=self.turn_count,
            was_interrupted=was_interrupted,
            agent_output=agent_output,
        )
        self.history.append(turn)

        # Notify memory subsystem of the completed turn
        await self.event_bus.publish(
            EventType.TURN_COMPLETE,
            {"turn": turn},
            source="Session",
        )
        return turn

    # ── text-in injection ──────────────────────────────────────────────────

    async def inject_text(
        self, content: str, priority: str = "normal", source: str = "unknown"
    ) -> None:
        entry = TextInEntry(
            content=content,
            priority=priority,
            source=source,
            timestamp=time.time(),
        )
        self.text_in_queue.append(entry)
        await self.event_bus.publish(EventType.TEXT_INJECTED, {"source": source})

    def flush_text_in(self) -> list[TextInEntry]:
        entries = list(self.text_in_queue)
        self.text_in_queue.clear()

        high = sorted(
            (e for e in entries if e.priority == "high"), key=lambda e: e.timestamp
        )
        normal = sorted(
            (e for e in entries if e.priority != "high"), key=lambda e: e.timestamp
        )
        return high + normal

    # ── queries ────────────────────────────────────────────────────────────

    def get_recent_history(self, n: int = 20) -> list[Turn]:
        return list(self.history)[-n:]

    def get_full_context(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "state": self.state.value,
            "persona": self.persona_name,
            "compressed_summary": self.compressed_summary,
            "recent_history": [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp,
                    "emotion": t.emotion,
                    "turn_id": t.turn_id,
                    "was_interrupted": t.was_interrupted,
                    "agent_output": t.agent_output,
                }
                for t in self.get_recent_history()
            ],
            "pending_text_in": len(self.text_in_queue),
        }

    # ── persistence hook ───────────────────────────────────────────────────

    async def save(self) -> None:
        logger.info("Saving session %s...", self.session_id)