"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        VOXCORE — core/event_bus.py                              ║
║              ASYNC PUB/SUB — ZERO-COUPLING INTER-MODULE COMMUNICATION           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    A lightweight asynchronous publish/subscribe event system built on asyncio.
    ALL inter-module communication flows through this bus. No module directly
    imports or calls another module — they publish events and subscribe to events.
    
    This is what makes VoxCore modular. Any module can be swapped out without
    touching any other module, as long as it publishes/subscribes to the same events.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # asyncio.Queue for each subscriber
import logging                          # Logger for event bus activity
from typing import Any, Callable, Awaitable, Optional
from dataclasses import dataclass, field
from enum import Enum
import time                             # Timestamp on events

═══════════════════════════════════════════════════════════════════════════════════
ENUMS:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
ENUM: EventType(str, Enum)
──────────────────────────────────────────────────────────────────────────────────
    All event types in the VoxCore system. Every publish/subscribe uses one of these.

    Members:
        # --- Input Events ---
        SPEECH_START         = "speech_start"          # VAD detected speech onset
        SPEECH_END           = "speech_end"            # VAD detected end of speech (silence threshold met)
        TRANSCRIPT_READY     = "transcript_ready"      # STT completed, final transcript available
        TEXT_INJECTED        = "text_injected"         # External text injected into context buffer

        # --- Brain Events ---
        LLM_SPEECH_TOKEN     = "llm_speech_token"      # One sentence of speech text from LLM (for TTS)
        LLM_AGENT_TAG        = "llm_agent_tag"         # Parsed <agent> JSON from LLM output
        LLM_STREAM_DONE      = "llm_stream_done"       # LLM finished streaming all tokens
        LLM_ERROR            = "llm_error"             # LLM call failed

        # --- Output Events ---
        TTS_CHUNK_READY      = "tts_chunk_ready"       # Audio chunk ready for playback
        TTS_SENTENCE_DONE    = "tts_sentence_done"     # One sentence fully synthesized and queued
        PLAYBACK_DONE        = "playback_done"         # Audio player finished playing all queued audio

        # --- State Events ---
        INTERRUPT_DETECTED   = "interrupt_detected"    # User spoke during SPEAKING state
        STATE_CHANGED        = "state_changed"         # TurnState changed (carries new state)
        
        # --- Backchannel Events ---
        BACKCHANNEL_OPPORTUNITY = "backchannel_opportunity"  # Phrase boundary detected, backchannel possible
        BACKCHANNEL_FIRE     = "backchannel_fire"      # Backchannel clip selected and should play

        # --- Agent Events ---
        TOOL_RESULT_READY    = "tool_result_ready"     # Tool execution completed, result available
        AGENT_JSON_OUT       = "agent_json_out"        # Validated agent JSON ready for external consumers

        # --- Safety Events ---
        SAFETY_FLAGGED       = "safety_flagged"        # Content flagged by Llama Guard

        # --- Memory Events ---
        MEMORY_COMPRESSED    = "memory_compressed"     # Old turns compressed into summary

═══════════════════════════════════════════════════════════════════════════════════
DATACLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
DATACLASS: Event
──────────────────────────────────────────────────────────────────────────────────
    Fields:
        event_type: EventType           # Which event this is
        data: Any = None                # Payload — varies by event type (see payload docs below)
        timestamp: float = field(default_factory=time.time)  # When the event was published
        source: str = ""                # Which module published this event (for debugging)
    
    PAYLOAD DOCUMENTATION BY EVENT TYPE:
        SPEECH_START:           data = None
        SPEECH_END:             data = {"audio_buffer": bytes, "duration_s": float}
        TRANSCRIPT_READY:       data = {"text": str, "confidence": float, "language": str}
        TEXT_INJECTED:          data = {"content": str, "priority": str, "source": str}
        LLM_SPEECH_TOKEN:       data = {"text": str, "sentence_index": int, "emotion": str}
        LLM_AGENT_TAG:          data = {"action": str, "params": dict, "raw": str}
        LLM_STREAM_DONE:        data = {"total_tokens": int, "model": str}
        LLM_ERROR:              data = {"error": str, "model": str}
        TTS_CHUNK_READY:        data = {"audio": bytes, "sentence_index": int}
        TTS_SENTENCE_DONE:      data = {"sentence_index": int}
        PLAYBACK_DONE:          data = None
        INTERRUPT_DETECTED:     data = {"speech_prob": float, "during_sentence": int}
        STATE_CHANGED:          data = {"old_state": str, "new_state": str}
        BACKCHANNEL_OPPORTUNITY: data = {"is_question": bool, "energy_level": float, "speech_duration_s": float}
        BACKCHANNEL_FIRE:       data = {"clip_name": str, "clip_path": str}
        TOOL_RESULT_READY:      data = {"action": str, "result": Any, "success": bool}
        AGENT_JSON_OUT:         data = {"action": str, "params": dict, "session_id": str, "turn_id": int, "timestamp": float}
        SAFETY_FLAGGED:         data = {"direction": str, "category": str, "content": str}  # direction = "input" or "output"
        MEMORY_COMPRESSED:      data = {"turns_compressed": int, "summary": str}

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: EventBus
──────────────────────────────────────────────────────────────────────────────────
    The central pub/sub hub. All modules hold a reference to the same EventBus instance.

    CONSTRUCTOR: __init__(self)
    ─────────────────────────────────────────────────────────────
        INPUTS: None
        
        INITIALIZES:
            self._subscribers: dict[EventType, list[asyncio.Queue]]
                — Maps each EventType to a list of subscriber queues.
                — Each subscriber gets its own queue so events fan out (1:N).
            
            self._callbacks: dict[EventType, list[Callable[[Event], Awaitable[None]]]]
                — Maps each EventType to a list of async callback functions.
                — Alternative to queue-based subscription for simpler handlers.
            
            self._logger: logging.Logger = logging.getLogger("EventBus")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def subscribe(self, event_type: EventType) -> asyncio.Queue
        INPUTS:
            - event_type: EventType — The event type to subscribe to
        OUTPUT:
            - asyncio.Queue — A new queue that will receive Event objects of this type
        WHAT IT DOES:
            1. Creates a new asyncio.Queue(maxsize=100)
            2. Appends to self._subscribers[event_type]
            3. Returns the queue
        USAGE:
            queue = event_bus.subscribe(EventType.TRANSCRIPT_READY)
            while True:
                event = await queue.get()
                # process event.data

    def on(self, event_type: EventType, callback: Callable[[Event], Awaitable[None]]) -> None
        INPUTS:
            - event_type: EventType — The event type to listen for
            - callback: async function(Event) -> None — Called when event fires
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Appends callback to self._callbacks[event_type]
        USAGE:
            async def handle_transcript(event: Event):
                print(event.data["text"])
            event_bus.on(EventType.TRANSCRIPT_READY, handle_transcript)

    async def publish(self, event_type: EventType, data: Any = None, source: str = "") -> None
        INPUTS:
            - event_type: EventType — The event to publish
            - data: Any — Payload (see payload docs above)
            - source: str — Name of the publishing module (for debug logging)
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Creates Event(event_type=event_type, data=data, source=source)
            2. For each queue in self._subscribers[event_type]:
               - If queue is full, logs warning and drops event (non-blocking)
               - Otherwise, puts Event into queue (non-blocking via put_nowait)
            3. For each callback in self._callbacks[event_type]:
               - Fires asyncio.create_task(callback(event))
               - Does NOT await — callbacks run concurrently
            4. Logs at DEBUG level: "Published {event_type} from {source}"

    def unsubscribe(self, event_type: EventType, queue: asyncio.Queue) -> None
        INPUTS:
            - event_type: EventType — The event type to unsubscribe from
            - queue: asyncio.Queue — The specific queue to remove
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Removes queue from self._subscribers[event_type]
            2. If event_type has no more subscribers, removes the key

    def remove_callback(self, event_type: EventType, callback: Callable) -> None
        INPUTS:
            - event_type: EventType — The event type
            - callback: Callable — The specific callback to remove
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Removes callback from self._callbacks[event_type]

    def get_subscriber_count(self, event_type: EventType) -> int
        INPUTS:
            - event_type: EventType
        OUTPUT:
            - int — Number of subscribers (queues + callbacks) for this event type
        WHAT IT DOES:
            Returns len(self._subscribers.get(event_type, [])) + len(self._callbacks.get(event_type, []))

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - EventType     (enum)
    - Event         (dataclass)
    - EventBus      (class)
═══════════════════════════════════════════════════════════════════════════════════
"""

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
    PLAYBACK_DONE = "playback_done"

    # --- State Events ---
    INTERRUPT_DETECTED = "interrupt_detected"
    STATE_CHANGED = "state_changed"

    # --- Backchannel Events ---
    BACKCHANNEL_OPPORTUNITY = "backchannel_opportunity"
    BACKCHANNEL_FIRE = "backchannel_fire"

    # --- Agent Events ---
    TOOL_RESULT_READY = "tool_result_ready"
    AGENT_JSON_OUT = "agent_json_out"

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
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
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