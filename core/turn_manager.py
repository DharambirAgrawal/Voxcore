"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                       VOXCORE — core/turn_manager.py                            ║
║          THE 4-STATE MACHINE — ORCHESTRATES ALL TURN-TAKING LOGIC              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Implements the VoxCore Turn State Machine with 4 states:
        LISTENING → THINKING → SPEAKING → (INTERRUPTED → THINKING)
    
    Subscribes to events from VAD, STT, LLM, and AudioPlayer.
    Drives state transitions on the Session object.
    Coordinates the handoff between input pipeline, brain, and output pipeline.

    This is the CONDUCTOR of VoxCore — it doesn't do processing itself, it just
    tells other modules when to start/stop via state transitions and events.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Main event loop, task management
import logging                          # Module logger

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType, Event

from brain.llm_client import LLMClient
from brain.prompt_builder import PromptBuilder
from brain.response_parser import ResponseParser
from brain.router import BrainRouter

from safety.guard import SafetyGuard    # Optional — only if safety is enabled

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: TurnManager
──────────────────────────────────────────────────────────────────────────────────
    The state machine that drives the entire conversational flow.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus,
                          llm_client: LLMClient, prompt_builder: PromptBuilder,
                          response_parser: ResponseParser, brain_router: BrainRouter,
                          safety_guard: Optional[SafetyGuard] = None,
                          config: dict = None)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — The master session state
            - event_bus: EventBus — The shared event bus
            - llm_client: LLMClient — For initiating LLM streaming calls
            - prompt_builder: PromptBuilder — For building the message array
            - response_parser: ResponseParser — For parsing LLM output stream
            - brain_router: BrainRouter — For deciding which LLM model to use
            - safety_guard: Optional[SafetyGuard] — For input safety checking (None if disabled)
            - config: dict — Full config dict for any turn_manager-specific settings
        
        INITIALIZES:
            self.session: Session                = session
            self.event_bus: EventBus             = event_bus
            self.llm_client: LLMClient           = llm_client
            self.prompt_builder: PromptBuilder   = prompt_builder
            self.response_parser: ResponseParser = response_parser
            self.brain_router: BrainRouter       = brain_router
            self.safety_guard: Optional[SafetyGuard] = safety_guard
            self._logger: logging.Logger         = logging.getLogger("TurnManager")
            self._current_llm_task: Optional[asyncio.Task] = None   # The running LLM stream task
            self._interrupted_content: str       = ""               # Partial LLM output saved on interrupt
            
            # Subscribe to events:
            self._transcript_queue = event_bus.subscribe(EventType.TRANSCRIPT_READY)
            self._interrupt_queue = event_bus.subscribe(EventType.INTERRUPT_DETECTED)
            self._playback_done_queue = event_bus.subscribe(EventType.PLAYBACK_DONE)
            self._llm_done_queue = event_bus.subscribe(EventType.LLM_STREAM_DONE)
            self._safety_queue = event_bus.subscribe(EventType.SAFETY_FLAGGED)

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            The main event loop of the turn manager. Runs two concurrent tasks:
            1. _handle_transcripts() — Listens for TRANSCRIPT_READY events
            2. _handle_interrupts() — Listens for INTERRUPT_DETECTED events
            3. _handle_playback_done() — Listens for PLAYBACK_DONE events
            
            Uses asyncio.gather() to run all three concurrently.
            Never returns (runs until cancelled).

    async def _handle_transcripts(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever in loop)
        WHAT IT DOES:
            Infinite loop that:
            1. Awaits self._transcript_queue.get()
            2. Extracts transcript text from event.data["text"]
            3. If text is empty or whitespace, ignores it
            4. Logs: "User said: {text}"
            5. Adds user turn to session: await session.add_turn("user", text)
            6. If safety_guard enabled, fires safety check asynchronously:
               asyncio.create_task(safety_guard.check_input(text))
            7. Calls await self._start_llm_response(text)

    async def _start_llm_response(self, user_text: str) -> None
        INPUTS:
            - user_text: str — The user's transcribed speech
        OUTPUT: None
        WHAT IT DOES:
            1. Transitions state: await session.set_state(TurnState.THINKING)
            2. Builds prompt: messages = prompt_builder.build(session)
            3. Selects model: model = brain_router.route(user_text, session)
            4. Starts LLM streaming as a task:
               self._current_llm_task = asyncio.create_task(
                   self._stream_and_parse(messages, model)
               )
            5. Does NOT await the task — it runs in the background while
               the turn manager continues listening for interrupts

    async def _stream_and_parse(self, messages: list[dict], model: str) -> None
        INPUTS:
            - messages: list[dict] — The full message array for the LLM
            - model: str — The model identifier to use
        OUTPUT: None
        WHAT IT DOES:
            1. Starts the LLM stream: stream = llm_client.stream_response(messages, model)
            2. Passes the stream to response_parser.parse_stream(stream)
               — response_parser handles firing LLM_SPEECH_TOKEN and LLM_AGENT_TAG events
            3. On first LLM_SPEECH_TOKEN event detection, transitions:
               await session.set_state(TurnState.SPEAKING)
            4. When stream completes, collects the full assistant response text
            5. Adds assistant turn: await session.add_turn("assistant", full_text, emotion=...)
            6. If safety_guard enabled, fires output safety check asynchronously
            7. Publishes LLM_STREAM_DONE event

    async def _handle_interrupts(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever in loop)
        WHAT IT DOES:
            Infinite loop that:
            1. Awaits self._interrupt_queue.get()
            2. Checks if current state is SPEAKING (ignore interrupts in other states)
            3. Logs: "INTERRUPT detected — killing TTS"
            4. Transitions: await session.set_state(TurnState.INTERRUPTED)
            5. Cancels the current LLM task: self._current_llm_task.cancel()
            6. Saves partial LLM output to session:
               — The response_parser keeps track of what has been generated so far
               — Save it as an interrupted assistant turn
            7. Publishes INTERRUPT_DETECTED event (redundant but useful for logging)
            8. Transitions back: await session.set_state(TurnState.LISTENING)
               — The normal STT loop will pick up the user's interrupting speech

    async def _handle_playback_done(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever in loop)
        WHAT IT DOES:
            Infinite loop that:
            1. Awaits self._playback_done_queue.get()
            2. If current state is SPEAKING:
               - Transitions: await session.set_state(TurnState.LISTENING)
               - Logs: "Playback done → LISTENING"
            3. If current state is INTERRUPTED:
               - Do nothing (interrupt handler already handled the transition)

    async def _handle_safety_flags(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever in loop)
        WHAT IT DOES:
            Infinite loop that:
            1. Awaits self._safety_queue.get()
            2. If event.data["direction"] == "output":
               - Cancels current LLM task
               - Publishes LLM_SPEECH_TOKEN with safe refusal message
               - Logs the safety incident
            3. If event.data["direction"] == "input":
               - Logs the flagged input
               - Optionally skips the LLM call entirely and responds with refusal

═══════════════════════════════════════════════════════════════════════════════════
STATE TRANSITION TABLE (for reference during implementation):
═══════════════════════════════════════════════════════════════════════════════════

    CURRENT STATE    │  EVENT                  │  NEW STATE    │  ACTIONS
    ─────────────────┼─────────────────────────┼───────────────┼──────────────────────
    LISTENING        │  TRANSCRIPT_READY       │  THINKING     │  Build prompt, start LLM stream
    THINKING         │  LLM_SPEECH_TOKEN(1st)  │  SPEAKING     │  Start TTS pipeline
    SPEAKING         │  PLAYBACK_DONE          │  LISTENING     │  Ready for next turn
    SPEAKING         │  INTERRUPT_DETECTED     │  INTERRUPTED  │  Cancel LLM, kill TTS
    INTERRUPTED      │  (immediate)            │  LISTENING     │  Wait for new TRANSCRIPT_READY

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - TurnManager   (class)
═══════════════════════════════════════════════════════════════════════════════════
"""
