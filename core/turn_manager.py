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

"""
VoxCore — core/turn_manager.py
The 5-state machine — orchestrates all turn-taking logic.

States: LISTENING → THINKING → SPEAKING → (SOFT_INJECT → SPEAKING | INTERRUPTED → THINKING)
"""

import asyncio
import logging
from typing import Optional

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType

from brain.llm_client import LLMClient
from brain.prompt_builder import PromptBuilder
from brain.response_parser import ResponseParser
from brain.router import BrainRouter

from safety.guard import SafetyGuard


class TurnManager:
    """State machine that drives the entire conversational flow."""

    def __init__(
        self,
        session: Session,
        event_bus: EventBus,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        brain_router: BrainRouter,
        safety_guard: Optional[SafetyGuard] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.session = session
        self.event_bus = event_bus
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.brain_router = brain_router
        self.safety_guard = safety_guard
        self._logger = logging.getLogger("TurnManager")

        self._current_llm_task: Optional[asyncio.Task] = None
        self._interrupted_content: str = ""
        self._just_interrupted: bool = False  # True when previous turn was interrupted

        # Stop words — user says these to silence the AI.  No LLM call needed.
        self._stop_phrases: set[str] = {
            "stop", "shut up", "be quiet", "enough", "okay stop",
            "stop talking", "quiet", "hush", "silence", "okay enough",
            "stop it", "that's enough", "ok stop", "ok enough",
            "please stop", "can you stop", "stop please",
        }

        # Subscribe to relevant events
        self._transcript_queue = event_bus.subscribe(EventType.TRANSCRIPT_READY)
        self._interrupt_queue = event_bus.subscribe(EventType.INTERRUPT_DETECTED)
        self._playback_done_queue = event_bus.subscribe(EventType.PLAYBACK_DONE)
        self._safety_queue = event_bus.subscribe(EventType.SAFETY_FLAGGED)
        self._tool_result_queue = event_bus.subscribe(EventType.TOOL_RESULT_READY)

    # ── main entry point ───────────────────────────────────────────────────

    async def run(self) -> None:
        """Run all event handlers concurrently. Never returns."""
        await asyncio.gather(
            self._handle_transcripts(),
            self._handle_interrupts(),
            self._handle_playback_done(),
            self._handle_safety_flags(),
            self._handle_tool_results(),
        )

    # ── transcript handling ────────────────────────────────────────────────

    async def _handle_transcripts(self) -> None:
        while True:
            event = await self._transcript_queue.get()
            text: str = event.data.get("text", "")
            if not text or not text.strip():
                continue

            self._logger.info("User said: %s", text)

            # ── Stop-word detection ──
            # If the user just interrupted with a stop command, don't call the
            # LLM — just quietly go back to listening.
            normalised = text.strip().lower().rstrip(".!?,")
            if self._just_interrupted and normalised in self._stop_phrases:
                self._logger.info("Stop phrase detected ('%s') — staying silent", normalised)
                self._just_interrupted = False
                await self.session.add_turn("user", text)
                # Don't start LLM response — stay in LISTENING
                continue

            await self.session.add_turn("user", text)

            if self.safety_guard is not None:
                asyncio.create_task(self.safety_guard.check(text, direction="input"))

            await self._start_llm_response(text)
            self._just_interrupted = False  # consumed

    # ── LLM orchestration ─────────────────────────────────────────────────

    async def _start_llm_response(self, user_text: str) -> None:
        await self.session.set_state(TurnState.THINKING)

        messages = self.prompt_builder.build(self.session)
        model = self.brain_router.route(user_text, self.session)

        self._current_llm_task = asyncio.create_task(
            self._stream_and_parse(messages, model)
        )

    async def _stream_and_parse(self, messages: list[dict], model: str) -> None:
        full_text = ""

        try:
            # Transition to SPEAKING once the stream starts (parse_stream
            # publishes LLM_SPEECH_TOKEN events internally).
            await self.session.set_state(TurnState.SPEAKING)

            stream = self.llm_client.stream_response(messages, model)
            full_text = await self.response_parser.parse_stream(stream)

            # Stream complete — record assistant turn
            await self.session.add_turn("assistant", full_text)

            if self.safety_guard is not None:
                asyncio.create_task(self.safety_guard.check(full_text, direction="output"))

            # NOTE: LLM_STREAM_DONE is already published by ResponseParser
            # at the end of parse_stream(). Do NOT publish it again here.

        except asyncio.CancelledError:
            # Interrupted — save what we had
            self._interrupted_content = full_text
            self._logger.info("LLM stream cancelled (interrupted), partial: %d chars", len(full_text))
            raise

        except Exception:
            self._logger.exception("Error during LLM stream")

    # ── interrupt handling ─────────────────────────────────────────────────

    async def _handle_interrupts(self) -> None:
        while True:
            event = await self._interrupt_queue.get()

            if self.session.state is not TurnState.SPEAKING:
                continue

            # Check if this interrupt came from the speaking monitor with a transcript
            monitor_transcript = event.data.get("transcript", "")
            source = event.data.get("source", "")

            self._logger.info("INTERRUPT detected — killing TTS (source=%s)", source or "gate1")
            await self.session.set_state(TurnState.INTERRUPTED)

            # Cancel the running LLM task
            if self._current_llm_task is not None and not self._current_llm_task.done():
                self._current_llm_task.cancel()
                try:
                    await self._current_llm_task
                except asyncio.CancelledError:
                    pass

            # Save partial output as interrupted assistant turn
            if self._interrupted_content:
                await self.session.add_turn(
                    "assistant",
                    self._interrupted_content,
                    was_interrupted=True,
                )
                self._interrupted_content = ""

            self._just_interrupted = True  # next transcript is the interrupting speech

            # If the speaking monitor already has a transcript, pre-load it
            # so the user doesn't have to wait for a full STT cycle
            if monitor_transcript and source == "speaking_monitor":
                self._logger.info(
                    "Pre-loaded interrupt transcript from monitor: '%s'",
                    monitor_transcript,
                )
                await self.session.set_state(TurnState.LISTENING)
                # Directly process the monitor transcript as user speech
                await self.session.add_turn("user", monitor_transcript)
                self._just_interrupted = False
                await self._start_llm_response(monitor_transcript)
            else:
                await self.session.set_state(TurnState.LISTENING)

    # ── playback done handling ─────────────────────────────────────────────

    async def _handle_playback_done(self) -> None:
        while True:
            await self._playback_done_queue.get()

            if self.session.state in (TurnState.SPEAKING, TurnState.SOFT_INJECT):
                await self.session.set_state(TurnState.LISTENING)
                self._logger.info("Playback done → LISTENING")
            # If INTERRUPTED, the interrupt handler already transitioned state

    # ── safety flag handling ───────────────────────────────────────────────

    async def _handle_safety_flags(self) -> None:
        while True:
            event = await self._safety_queue.get()
            direction = event.data.get("direction", "")

            if direction == "output":
                self._logger.warning("Safety flagged output — cancelling LLM and sending refusal")
                if self._current_llm_task is not None and not self._current_llm_task.done():
                    self._current_llm_task.cancel()
                    try:
                        await self._current_llm_task
                    except asyncio.CancelledError:
                        pass

                await self.event_bus.publish(
                    EventType.LLM_SPEECH_TOKEN,
                    {"text": "I'm sorry, I can't respond to that."},
                )

            elif direction == "input":
                self._logger.warning("Safety flagged input: %s", event.data.get("reason", "unknown"))

    # ── tool result handling ───────────────────────────────────────────────

    async def _handle_tool_results(self) -> None:
        """Auto-trigger a new LLM call after a tool result is injected.

        When a tool (e.g. web_search) finishes, its result is injected into
        the session's text_in_queue by TextInjector.  We fire a fresh LLM
        call so the AI can read the result and speak it to the user.
        The tool result appears as [CONTEXT] in the prompt via PromptBuilder.
        """
        while True:
            event = await self._tool_result_queue.get()
            action = event.data.get("action", "unknown")
            success = event.data.get("success", True)

            self._logger.info("Tool '%s' result received (success=%s) — auto-responding", action, success)

            # Wait briefly for the TextInjector to finish adding to text_in_queue
            await asyncio.sleep(0.05)

            # Cancel the previous LLM task so we don't have two streams running
            # at the same time (which causes raw tool result text to be spoken).
            if self._current_llm_task is not None and not self._current_llm_task.done():
                self._current_llm_task.cancel()
                try:
                    await self._current_llm_task
                except asyncio.CancelledError:
                    pass
                self._interrupted_content = ""  # discard partial from cancelled stream

            # Don't add a fake user turn — just build prompt from existing state
            # (the tool result is already in text_in_queue, PromptBuilder.build()
            # will flush it into the [CONTEXT] block)
            await self.session.set_state(TurnState.THINKING)
            messages = self.prompt_builder.build(self.session)
            model = self.brain_router.route("tool result", self.session)
            self._current_llm_task = asyncio.create_task(
                self._stream_and_parse(messages, model)
            )