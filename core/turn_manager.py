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
The 6-state machine — orchestrates all turn-taking logic.

States: LISTENING → THINKING → SPEAKING → (PAUSED → SPEAKING | SOFT_INJECT → SPEAKING | INTERRUPTED → THINKING)
"""

import asyncio
import logging
import re
from typing import Optional

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType

from brain.llm_client import LLMClient
from brain.prompt_builder import PromptBuilder
from brain.response_parser import ResponseParser
from brain.router import BrainRouter

from safety.guard import SafetyGuard

# ── Deferral phrase stripping ─────────────────────────────────────────────────
# When a user says "after this, can you X" the SpeakingMonitor stores the full
# transcript as a deferred task. Before passing it to the LLM we strip the
# lead-in so the model sees a clean imperative ("tell me the Bitcoin price")
# rather than the conditional form which confused it into continuing the story.
_DEFERRAL_PREFIXES = re.compile(
    r'^(?:'
    r'after (?:this|that|you(?:\'re)? done|you finish(?:ing)?|the story)|'
    r'when you(?:\'re)? (?:done|finished)|'
    r'once you(?:\'re)? (?:done|finished)|'
    r'after you(?:\'re)? (?:done|finished)|'
    r'then'
    r')[,\s]+(?:can you |could you |would you |please )?',
    re.IGNORECASE,
)


def _strip_deferral_prefix(text: str) -> str:
    """Remove lead-in deferral phrases and return the clean request.

    E.g. "After this, can you tell me the Bitcoin price?"
         → "tell me the Bitcoin price?"
    """
    cleaned = _DEFERRAL_PREFIXES.sub("", text).strip()
    # Capitalise first letter after stripping
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or text


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
        # Fallback model chain for rate-limit recovery (fast → smart → deep)
        _mcfg = (config or {}).get("models", {})
        self._llm_fallback_chain: list[str] = [
            m for m in (
                _mcfg.get("llm_smart"),
                _mcfg.get("llm_deep"),
                _mcfg.get("llm_agentic"),
            ) if m
        ]

        self._current_llm_task: Optional[asyncio.Task] = None
        self._interrupted_content: str = ""
        self._just_interrupted: bool = False  # True when previous turn was interrupted

        # V5: Set True while _handle_spoken_tool_output is feeding article/tool
        # text to TTS.  Prevents the PLAYBACK_DONE from the short bridge
        # phrase ("Sure, let me read that...") from transitioning the state
        # to LISTENING before the article audio has even started queuing.
        self._spoken_tool_in_flight: bool = False

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
        # V5: Spoken tool output (article_fetch etc) — goes directly to TTS
        self._spoken_tool_queue = event_bus.subscribe(EventType.SPOKEN_TOOL_OUTPUT)
        # V3: CLASSIFIED events from SpeakingMonitor's Gate 3
        self._classified_queue = event_bus.subscribe(EventType.CLASSIFIED)

        # V3: InterruptRouter reference (set via set_interrupt_router after construction)
        self._interrupt_router = None

        # V5: Memory components (set via set_memory_components after construction)
        self._background_llm = None
        self._short_term = None
        self._fact_store = None
        self._procedural = None
        self._long_term = None

    # ── main entry point ───────────────────────────────────────────────────

    async def run(self) -> None:
        """Run all event handlers concurrently. Never returns."""
        await asyncio.gather(
            self._handle_transcripts(),
            self._handle_interrupts(),
            self._handle_playback_done(),
            self._handle_safety_flags(),
            self._handle_tool_results(),
            self._handle_spoken_tool_output(),  # V5
            self._handle_classified(),  # V3
        )

    def set_interrupt_router(self, router) -> None:
        """V3: Inject InterruptRouter after construction."""
        self._interrupt_router = router
        self._logger.info("V3: InterruptRouter attached to TurnManager")

    def set_memory_components(self, *, background_llm=None, short_term=None,
                              fact_store=None, procedural=None, long_term=None) -> None:
        """V5: Inject memory components for background fact extraction after each turn."""
        self._background_llm = background_llm
        self._short_term = short_term
        self._fact_store = fact_store
        self._procedural = procedural
        self._long_term = long_term
        self._logger.info("V5: Memory components attached to TurnManager")

    # ── transcript handling ────────────────────────────────────────────────

    async def _handle_transcripts(self) -> None:
        while True:
            event = await self._transcript_queue.get()
            text: str = event.data.get("text", "")
            if not text or not text.strip():
                continue

            # V3: If InterruptRouter already added the turn, skip add_turn
            turn_already_added = event.data.get("turn_added", False)

            # V3: During PENDING/PAUSED, only process transcripts from InterruptRouter
            # (which have turn_added=True). Drop normal VAD→STT transcripts to prevent
            # race condition with SpeakingMonitor's classification pipeline.
            if not turn_already_added and self.session.state in (
                TurnState.PENDING, TurnState.PAUSED,
            ):
                self._logger.debug(
                    "Dropping transcript during %s (SpeakingMonitor has priority): '%s'",
                    self.session.state.value, text[:40],
                )
                continue

            self._logger.info("User said: %s", text)

            # ── Stop-word detection ──
            # If the user just interrupted with a stop command, don't call the
            # LLM — just quietly go back to listening.
            normalised = text.strip().lower().rstrip(".!?,")
            if self._just_interrupted and normalised in self._stop_phrases:
                self._logger.info("Stop phrase detected ('%s') — staying silent", normalised)
                self._just_interrupted = False
                if not turn_already_added:
                    await self.session.add_turn("user", text)
                # Don't start LLM response — stay in LISTENING
                continue

            if not turn_already_added:
                await self.session.add_turn("user", text)

            if self.safety_guard is not None:
                asyncio.create_task(self.safety_guard.check(text, direction="input"))

            await self._start_llm_response(text)
            self._just_interrupted = False  # consumed

    # ── LLM orchestration ─────────────────────────────────────────────────

    async def _start_llm_response(self, user_text: str) -> None:
        await self.session.set_state(TurnState.THINKING)

        messages = self.prompt_builder.build(self.session)
        primary_model = self.brain_router.route(user_text, self.session)

        # Auto-fallback: if primary model is already rate-limited, pick the
        # first available model in the fallback chain so the turn responds
        # immediately rather than blocking for 20-30 s internally.
        model = self.llm_client.pick_available(
            primary_model, *self._llm_fallback_chain
        )
        if model != primary_model:
            self._logger.warning(
                "Model %s rate limited → falling back to %s",
                primary_model, model,
            )

        self._current_llm_task = asyncio.create_task(
            self._stream_and_parse(messages, model)
        )

    async def _stream_and_parse(self, messages: list[dict], model: str) -> None:
        full_text = ""

        try:
            # Transition to SPEAKING once the stream starts (parse_stream
            # publishes LLM_SPEECH_TOKEN events internally).
            await self.session.set_state(TurnState.SPEAKING)

            # suppress_stream_done_if_empty=True: if the model is rate-limited
            # and returns nothing, LLM_STREAM_DONE is NOT fired yet so that we
            # can retry with a fallback model in the same turn (below).
            stream = self.llm_client.stream_response(messages, model)
            full_text = await self.response_parser.parse_stream(
                stream, suppress_stream_done_if_empty=True
            )

            # ── Rate-limit mid-stream recovery ────────────────────────────
            # If the model hit a 429 after pick_available() already chose it
            # (i.e. the cached rate-limit window had just expired), the stream
            # returns empty and the client now knows it is rate-limited again.
            # Retry once with the best available fallback model.
            if not full_text.strip() and self.llm_client.is_rate_limited(model):
                fallback = self.llm_client.pick_available(*self._llm_fallback_chain)
                if fallback and not self.llm_client.is_rate_limited(fallback):
                    self._logger.warning(
                        "Rate-limited mid-stream on %s → retrying with %s",
                        model, fallback,
                    )
                    stream2 = self.llm_client.stream_response(messages, fallback)
                    # This call fires LLM_STREAM_DONE normally on completion.
                    full_text = await self.response_parser.parse_stream(stream2)
                else:
                    # Every model is rate-limited — fire DONE so TTS doesn't
                    # stall indefinitely, and surface a brief spoken message.
                    self._logger.error(
                        "All fallback models rate-limited — no response for this turn"
                    )
                    await self.event_bus.publish(
                        EventType.LLM_SPEECH_TOKEN,
                        {
                            "text": "I'm sorry, I'm having trouble connecting right now. Please try again in a moment.",
                            "sentence_index": 0,
                            "emotion": "neutral",
                        },
                        source="TurnManager",
                    )
                    await self.event_bus.publish(EventType.LLM_STREAM_DONE, {})
                    return
            elif not full_text.strip():
                # Empty for a non-rate-limit reason — fire DONE so TTS doesn't stall.
                await self.event_bus.publish(EventType.LLM_STREAM_DONE, {})

            # Don't record an empty assistant turn in session history.
            if not full_text.strip():
                return

            # Stream complete — record assistant turn
            await self.session.add_turn("assistant", full_text)

            # V5: Fire background fact extraction (never awaited — zero latency impact)
            if (self._background_llm is not None
                    and self._short_term is not None
                    and self._fact_store is not None
                    and self._procedural is not None
                    and self._long_term is not None):
                turns = self._short_term.get_recent(10)
                turn_dicts = [{"role": t.role, "content": t.content} for t in turns]
                existing_summary = self.session.compressed_summary or ""
                asyncio.create_task(
                    self._background_llm.process_turn_background(
                        turns=turn_dicts,
                        fact_store=self._fact_store,
                        procedural=self._procedural,
                        long_term=self._long_term,
                        existing_summary=existing_summary,
                    )
                )

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

            # Guard: spoken tool output is still being fed to TTS.
            # The PLAYBACK_DONE here belongs to the short bridge phrase
            # ("Sure, let me read that for you!") — not to the article.
            # Ignore it; the article's own PLAYBACK_DONE will fire normally.
            if self._spoken_tool_in_flight:
                self._logger.debug(
                    "PLAYBACK_DONE ignored — spoken tool output still in flight"
                )
                continue

            if self.session.state in (TurnState.SPEAKING, TurnState.SOFT_INJECT):
                await self.session.set_state(TurnState.LISTENING)
                self._logger.info("Playback done → LISTENING")

                # ── Deferred INJECT execution ──────────────────────────────
                # If the user said "after this can you tell me X" during TTS,
                # the InterruptRouter stored the transcript as a deferred task.
                # Now that playback is done, fire it as a fresh user turn.
                if self._interrupt_router is not None:
                    deferred = self._interrupt_router.pop_deferred_task()
                    if deferred:
                        # Strip deferral lead-in phrases so the LLM sees the
                        # clean request rather than "after this, can you…"
                        clean = _strip_deferral_prefix(deferred)
                        self._logger.info(
                            "Executing deferred task: '%s' (cleaned: '%s')",
                            deferred[:60], clean[:60],
                        )
                        await self.session.add_turn("user", clean)
                        await self._start_llm_response(clean)

            elif self.session.state == TurnState.PAUSED:
                # V3: PAUSED stays PAUSED — playback done doesn't exit PAUSED
                self._logger.debug("Playback done during PAUSED — ignoring")
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

        V5: Tools with produces_spoken_output=True are handled by
        _handle_spoken_tool_output() instead — skip auto-LLM-call for them.
        """
        while True:
            event = await self._tool_result_queue.get()
            action = event.data.get("action", "unknown")
            success = event.data.get("success", True)
            spoken_output = event.data.get("spoken_output", False)

            if spoken_output:
                # Spoken tool output is handled by _handle_spoken_tool_output
                # — the result goes directly to TTS, no LLM re-call needed.
                self._logger.info(
                    "Tool '%s' is spoken output — skipping auto-LLM-call", action
                )
                continue

            self._logger.info("Tool '%s' result received (success=%s) — auto-responding", action, success)

            # Wait briefly for the TextInjector to finish adding to text_in_queue
            await asyncio.sleep(0.05)

            # Inject a guard so the LLM answers directly from the result and
            # does NOT call the same tool again (llama-3.1-8b-instant tends to
            # repeat the tool call when it sees a prior agent tag in history).
            await self.session.inject_text(
                content=(
                    f"[INSTRUCTION]: The '{action}' result above is ready. "
                    "Answer the user's question directly using this data. "
                    "Do NOT call any tools again for this response."
                ),
                priority="high",
                source="tool_result_guard",
            )

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

            # Strip <agent>…</agent> from the last assistant message so the
            # LLM doesn't see itself as mid-tool-call and repeat the same
            # tool invocation even though the result is already in context.
            _AGENT_TAG_RE = re.compile(r'\n?<agent>.*?</agent>', re.DOTALL)
            for _i in range(len(messages) - 1, -1, -1):
                if messages[_i]["role"] == "assistant":
                    messages[_i] = {
                        **messages[_i],
                        "content": _AGENT_TAG_RE.sub("", messages[_i]["content"]).strip(),
                    }
                    break

            primary_model = self.brain_router.route("tool result", self.session)
            model = self.llm_client.pick_available(
                primary_model, *self._llm_fallback_chain
            )
            if model != primary_model:
                self._logger.warning(
                    "Tool follow-up: %s rate limited → falling back to %s",
                    primary_model, model,
                )
            self._current_llm_task = asyncio.create_task(
                self._stream_and_parse(messages, model)
            )

    # ── V5: spoken tool output handling ───────────────────────────────────

    async def _handle_spoken_tool_output(self) -> None:
        """Feed spoken tool output directly to TTS — same path as main LLM.

        When a tool with produces_spoken_output=True finishes (e.g. article_fetch),
        its result is already polished spoken text. We feed it through the same
        response_parser → LLM_SPEECH_TOKEN → TTS pipeline as the main LLM,
        so downstream (TTS, audio player, state machine) sees no difference.

        Flow: SPOKEN_TOOL_OUTPUT → cancel LLM bridge → SPEAKING →
              response_parser.parse_stream(text) → LLM_SPEECH_TOKEN events →
              TTS generates audio → PLAYBACK_DONE → LISTENING
        """
        while True:
            event = await self._spoken_tool_queue.get()
            text = event.data.get("text", "")
            action = event.data.get("action", "unknown")

            if not text or not text.strip():
                self._logger.warning("Empty spoken tool output from '%s' — ignoring", action)
                continue

            self._logger.info(
                "Spoken tool output from '%s' (%d chars) — feeding to TTS",
                action, len(text),
            )

            # Cancel any running LLM task (the bridge phrase stream may still
            # be active — "Let me read that for you..." was streamed by the
            # main LLM before emitting the agent tag).  Already-queued TTS
            # audio for the bridge will finish playing; the spoken output
            # sentences queue naturally after it.
            if self._current_llm_task is not None and not self._current_llm_task.done():
                self._current_llm_task.cancel()
                try:
                    await self._current_llm_task
                except asyncio.CancelledError:
                    pass
                self._interrupted_content = ""

            # Transition to SPEAKING (if not already — bridge may still be playing)
            await self.session.set_state(TurnState.SPEAKING)

            # Flag: prevent bridge-phrase PLAYBACK_DONE from going to LISTENING
            self._spoken_tool_in_flight = True
            try:
                async def _text_as_stream():
                    yield text

                full_text = await self.response_parser.parse_stream(_text_as_stream())
            finally:
                self._spoken_tool_in_flight = False

            # Record as an assistant turn in session history
            await self.session.add_turn("assistant", full_text)

            self._logger.info(
                "Spoken tool output from '%s' fully emitted to TTS", action
            )

    # ── V3: CLASSIFIED event handling ────────────────────────────────────

    async def _handle_classified(self) -> None:
        """V3: Route CLASSIFIED events from Gate 3 to InterruptRouter.

        SpeakingMonitor publishes CLASSIFIED after Gate 3 completes and
        VAD confirms user stopped speaking. TurnManager simply delegates
        to InterruptRouter.route() for all decision/type handling.
        """
        while True:
            event = await self._classified_queue.get()

            if self._interrupt_router is None:
                self._logger.warning(
                    "CLASSIFIED received but no InterruptRouter attached — ignoring"
                )
                continue

            classified = event.data
            self._logger.info(
                "CLASSIFIED: decision=%s type=%s",
                classified.get("decision", "?"),
                classified.get("interrupt_type", "none"),
            )

            try:
                await self._interrupt_router.route(classified)
            except Exception:
                self._logger.exception("InterruptRouter.route() failed")