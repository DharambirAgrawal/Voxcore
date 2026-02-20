"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         VOXCORE — core/session.py                               ║
║              MASTER STATE — SINGLE SOURCE OF TRUTH FOR THE CONVERSATION         ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Holds ALL state for the current conversation session. Every module reads from
    or writes to the Session object. It contains the conversation history, the
    text-in injection queue, persona config, turn state, and session metadata.

    This is the GOD OBJECT of VoxCore — deliberately so, to avoid scattered state.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # For asyncio.Queue used in text_in_queue
import time                             # For session start time
import uuid                             # For generating unique session_id
from collections import deque           # For bounded conversation history
from dataclasses import dataclass, field  # For Turn dataclass
from enum import Enum                   # For TurnState enum
from typing import Optional, Any        # Type hints

from core.event_bus import EventBus     # To publish state change events

═══════════════════════════════════════════════════════════════════════════════════
ENUMS:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
ENUM: TurnState(Enum)
──────────────────────────────────────────────────────────────────────────────────
    Members:
        LISTENING    = "listening"      # VAD active, STT accumulating, waiting for user to finish
        THINKING     = "thinking"       # STT finalized, LLM streaming started, waiting for first token
        SPEAKING     = "speaking"       # TTS playing audio, LLM may still be streaming
        INTERRUPTED  = "interrupted"    # User spoke during SPEAKING, TTS killed, re-entering THINKING
    
    Used by: TurnManager, InterruptionDetector, AudioPlayer, BackchannelSelector

═══════════════════════════════════════════════════════════════════════════════════
DATACLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
DATACLASS: Turn
──────────────────────────────────────────────────────────────────────────────────
    Fields:
        role: str                       # "user" or "assistant"
        content: str                    # The text content of this turn
        timestamp: float                # time.time() when turn was created
        emotion: Optional[str]          # Emotion tag if present (e.g. "cheerful", "calm")
        turn_id: int                    # Sequential turn number in this session
        was_interrupted: bool = False   # True if this assistant turn was interrupted
        agent_output: Optional[dict] = None  # Parsed agent JSON if this turn contained <agent> tags
    
    Used in: session.history deque, passed to prompt_builder, stored in memory

──────────────────────────────────────────────────────────────────────────────────
DATACLASS: TextInEntry
──────────────────────────────────────────────────────────────────────────────────
    Fields:
        content: str                    # The injected text content
        priority: str = "normal"        # "normal" or "high" — high goes first in prompt
        source: str = "unknown"         # Where this injection came from (e.g. "tool_result", "rag", "notification")
        timestamp: float                # time.time() when injected
    
    Used by: TextInjector writes these, PromptBuilder reads/flushes them

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: Session
──────────────────────────────────────────────────────────────────────────────────
    The master state container for the entire VoxCore session.

    CONSTRUCTOR: __init__(self, config: dict, event_bus: EventBus)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - config: dict — The "persona" section from config.yaml containing:
                - name: str (persona name)
                - system_prompt: str (full system prompt)
                - voice: str (Kokoro voice ID, e.g. "af_heart")
                - language: str ("en" or "ar")
            - event_bus: EventBus — The shared event bus instance
        
        INITIALIZES:
            self.session_id: str          = str(uuid.uuid4())
            self.start_time: float        = time.time()
            self.turn_count: int          = 0
            self.state: TurnState         = TurnState.LISTENING
            self.persona_name: str        = config["name"]
            self.system_prompt: str       = config["system_prompt"]
            self.voice: str               = config["voice"]
            self.language: str            = config["language"]
            self.history: deque[Turn]     = deque(maxlen=50)  # Keeps last 50 turns raw, prompt_builder uses last 20
            self.text_in_queue: list[TextInEntry] = []        # Pending text-in injections
            self.compressed_summary: str  = ""                # Summary of old turns from compressor
            self.event_bus: EventBus      = event_bus
            self._lock: asyncio.Lock      = asyncio.Lock()    # Protects concurrent writes

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def set_state(self, new_state: TurnState) -> None
        INPUTS:
            - new_state: TurnState — The new state to transition to
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Acquires self._lock
            2. Sets self.state = new_state
            3. Publishes a STATE_CHANGED event via event_bus with the new state value
            4. Logs the state transition: "State: {old} → {new}"
    
    async def add_turn(self, role: str, content: str, emotion: Optional[str] = None,
                       was_interrupted: bool = False, agent_output: Optional[dict] = None) -> Turn
        INPUTS:
            - role: str — "user" or "assistant"
            - content: str — The text content
            - emotion: Optional[str] — Emotion tag if present
            - was_interrupted: bool — Whether this turn was interrupted
            - agent_output: Optional[dict] — Parsed agent JSON if any
        OUTPUT:
            - Turn — The created Turn dataclass instance
        WHAT IT DOES:
            1. Increments self.turn_count
            2. Creates Turn(role=role, content=content, timestamp=time.time(),
                           emotion=emotion, turn_id=self.turn_count,
                           was_interrupted=was_interrupted, agent_output=agent_output)
            3. Appends to self.history
            4. Returns the Turn

    async def inject_text(self, content: str, priority: str = "normal", source: str = "unknown") -> None
        INPUTS:
            - content: str — Text to inject into the context
            - priority: str — "normal" or "high"
            - source: str — Source identifier
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Creates TextInEntry(content=content, priority=priority, source=source, timestamp=time.time())
            2. Appends to self.text_in_queue
            3. Publishes TEXT_INJECTED event via event_bus

    def flush_text_in(self) -> list[TextInEntry]
        INPUTS:
            - None
        OUTPUT:
            - list[TextInEntry] — All pending text-in entries, sorted: high priority first, then by timestamp
        WHAT IT DOES:
            1. Copies self.text_in_queue
            2. Clears self.text_in_queue
            3. Sorts: high priority entries first, then normal, each group sorted by timestamp ascending
            4. Returns the sorted list

    def get_recent_history(self, n: int = 20) -> list[Turn]
        INPUTS:
            - n: int — Number of recent turns to return (default 20)
        OUTPUT:
            - list[Turn] — The last n turns from self.history
        WHAT IT DOES:
            1. Returns list(self.history)[-n:] — slices the deque to get last n items

    def get_full_context(self) -> dict
        INPUTS:
            - None
        OUTPUT:
            - dict with keys:
                - session_id: str
                - turn_count: int
                - state: str (TurnState.value)
                - persona: str
                - compressed_summary: str
                - recent_history: list[dict] (each Turn as dict)
                - pending_text_in: int (count of pending injections)
        WHAT IT DOES:
            Assembles a snapshot of the entire session state for debugging/API responses.

    async def save(self) -> None
        INPUTS:
            - None
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Logs "Saving session {session_id}..."
            2. This is a hook for persistence — can be extended to save to disk/DB
            3. Currently a no-op placeholder for shutdown cleanup

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - TurnState     (enum)
    - Turn          (dataclass)
    - TextInEntry   (dataclass)
    - Session       (class)
═══════════════════════════════════════════════════════════════════════════════════
"""


"""
VoxCore — core/session.py
Master state — single source of truth for the conversation.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

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