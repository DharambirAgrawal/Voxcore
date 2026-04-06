"""
╔══════��═══════════════════════════════════════════════════════════════════════════╗
║                        VOXCORE — core/event_bus.py                              ║
║              ASYNC PUB/SUB — ZERO-COUPLING INTER-MODULE COMMUNICATION           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List


class EventType(str, Enum):
    """All event types in the VoxCore system."""

    # --- Input Events ---
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TRANSCRIPT_READY = "transcript_ready"
    TEXT_INJECTED = "text_injected"
    AUDIO_CHUNK = "audio_chunk"

    # --- Brain Events ---
    LLM_SPEECH_TOKEN = "llm_speech_token"
    LLM_AGENT_TAG = "llm_agent_tag"
    LLM_STREAM_DONE = "llm_stream_done"
    LLM_ERROR = "llm_error"

    # --- Output Events ---
    TTS_CHUNK_READY = "tts_chunk_ready"
    TTS_SENTENCE_DONE = "tts_sentence_done"
    TTS_ALL_DONE = "tts_all_done"
    PLAYBACK_DONE = "playback_done"

    # --- State Events ---
    INTERRUPT_DETECTED = "interrupt_detected"
    STATE_CHANGED = "state_changed"

    # --- Backchannel Events ---
    BACKCHANNEL_OPPORTUNITY = "backchannel_opportunity"
    BACKCHANNEL_FIRE = "backchannel_fire"

    # --- Speaking Monitor Events (v2) ---
    GATE1_PASSED = "gate1_passed"
    PAUSE_MARKER = "pause_marker"
    PLAY_CLIP = "play_clip"
    POSITIVE_REACTION = "positive_reaction"
    MONITOR_CLASSIFY = "monitor_classify"
    VOLUME_DUCK = "volume_duck"        # Duck audio volume (user speaking)
    VOLUME_RESTORE = "volume_restore"  # Restore audio volume (false alarm / IGNORE)
    PLAYBACK_PAUSE = "playback_pause"    # Freeze playback in place (user speaking)
    PLAYBACK_RESUME = "playback_resume"  # Resume playback from where it froze

    # --- V3: Interrupt Pipeline Events ---
    PENDING = "pending"                    # User speech detected during SPEAKING, TTS paused
    CLASSIFIED = "classified"              # Gate 3 returned DECISION + TYPE
    FILLER_REACTION = "filler_reaction"    # Gate 2 detected filler word, AI continues
    FILTER_ACTIVE_CHANGE = "filter_active_change"  # AriaVoiceFilter / Gate 0 toggled
    WARMUP_COMPLETE = "warmup_complete"    # Startup warmup finished, echo protection ready

    # --- Agent Events ---
    TOOL_RESULT_READY = "tool_result_ready"
    AGENT_JSON_OUT = "agent_json_out"
    SPOKEN_TOOL_OUTPUT = "spoken_tool_output"

    # --- Safety Events ---
    SAFETY_FLAGGED = "safety_flagged"

    # --- Turn Events ---
    TURN_COMPLETE = "turn_complete"
    SESSION_RESET = "session_reset"

    # --- Memory Events ---
    MEMORY_COMPRESS = "memory_compress"
    MEMORY_COMPRESSED = "memory_compressed"


@dataclass
class Event:
    """A single event flowing through the bus."""

    event_type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class EventBus:
    """Central pub/sub hub. All modules hold a reference to the same instance."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[asyncio.Queue]] = {}
        self._callbacks: Dict[
            EventType, List[Callable[[Event], Awaitable[None]]]
        ] = {}
        self._logger: logging.Logger = logging.getLogger("EventBus")

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], Awaitable[None]] | None = None,
    ) -> asyncio.Queue | None:
        """
        Create and return a new queue that will receive all events of the
        given type. Each subscriber gets its own queue (fan-out).

        If *callback* is provided, register it via ``on()`` instead of
        creating a queue-based subscription (returns ``None``).
        """
        if callback is not None:
            self.on(event_type, callback)
            return None
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.setdefault(event_type, []).append(queue)
        return queue

    def on(
        self,
        event_type: EventType,
        callback: Callable[[Event], Awaitable[None]],
    ) -> None:
        """Register an async callback to be invoked when *event_type* fires."""
        self._callbacks.setdefault(event_type, []).append(callback)

    async def publish(
        self,
        event_type: EventType,
        data: Any = None,
        source: str = "",
    ) -> None:
        """
        Publish an event to all subscribers and callbacks.

        Queue delivery is non-blocking (``put_nowait``); if a subscriber's
        queue is full the event is dropped with a warning.  Callbacks are
        scheduled as fire-and-forget tasks so ``publish`` never awaits.
        """
        event = Event(event_type=event_type, data=data, source=source)

        # Fan-out to queue-based subscribers
        for queue in self._subscribers.get(event_type, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._logger.warning(
                    "Subscriber queue full for %s — dropping event",
                    event_type.value,
                )

        # Fire-and-forget async callbacks
        for callback in self._callbacks.get(event_type, []):
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(callback(event))
            except RuntimeError:
                # No running event loop — should not happen in normal operation
                self._logger.warning(
                    "No running loop when publishing %s via callback",
                    event_type.value,
                )

        self._logger.debug(
            "Published %s from %s", event_type.value, source or "unknown"
        )

    def unsubscribe(
        self, event_type: EventType, queue: asyncio.Queue
    ) -> None:
        """Remove a specific subscriber queue."""
        queues = self._subscribers.get(event_type)
        if queues and queue in queues:
            queues.remove(queue)
            if not queues:
                del self._subscribers[event_type]

    def remove_callback(
        self,
        event_type: EventType,
        callback: Callable[[Event], Awaitable[None]],
    ) -> None:
        """Remove a specific callback."""
        callbacks = self._callbacks.get(event_type)
        if callbacks and callback in callbacks:
            callbacks.remove(callback)
            if not callbacks:
                del self._callbacks[event_type]

    def get_subscriber_count(self, event_type: EventType) -> int:
        """Return total number of subscribers (queues + callbacks) for an event type."""
        return len(self._subscribers.get(event_type, [])) + len(
            self._callbacks.get(event_type, [])
        )