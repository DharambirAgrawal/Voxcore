"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE — brain/interrupt_router.py                          ║
║           V3: ALL 7 INTERRUPT SUB-TYPE ROUTING ACTIONS                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Single source of truth for interrupt routing. turn_manager calls only
    `await interrupt_router.route(classified_event)`. All behavior per
    DECISION + TYPE lives here.

    allam-2-7b (classifier) outputs DECISION + TYPE only — never speaks.
    Primary LLM (llama-3.1-8b-instant) handles all spoken responses.

DECISIONS:
    IGNORE    → resume TTS, no LLM call
    INJECT    → resume TTS, inject text at inject_point
    INTERRUPT → route by TYPE (7 sub-types below)

INTERRUPT TYPES:
    STOP, PAUSE, CORRECTION, NEW_QUESTION,
    SAME_TOPIC_REDIRECT, URGENCY, DONE_LISTENING
"""

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Resume phrases for PAUSED state detection (local, no API)
RESUME_PHRASES = {
    "go ahead", "continue", "okay", "resume",
    "keep going", "please continue", "go on", "yes go ahead",
    "go ahead please", "okay go ahead", "ok go ahead",
}


class InterruptRouter:
    """
    Routes classified interrupt events to the correct behavior.

    All 7 sub-types + IGNORE + INJECT handled here. No routing
    logic lives elsewhere.

    Args:
        config: Full config dict
        audio_player: AudioPlayer instance (pause/resume/flush/play_clip)
        mic_stream: MicStream instance (set_filter_active)
        llm_client: LLMClient for primary LLM responses
        prompt_builder: PromptBuilder for constructing LLM messages
        event_bus: EventBus for publishing events
        session: Session for state management
        tts_client: TTSClient for synthesis
        gate3_done_event: Shared asyncio.Event to stop filler clip
    """

    def __init__(
        self,
        config: dict,
        audio_player,
        mic_stream,
        llm_client,
        prompt_builder,
        event_bus,
        session,
        tts_client,
        gate3_done_event: asyncio.Event,
    ):
        self._config = config
        self._audio_player = audio_player
        self._mic_stream = mic_stream
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder
        self._event_bus = event_bus
        self._session = session
        self._tts_client = tts_client
        self._gate3_done_event = gate3_done_event

        # Clips directory from config
        bc_cfg = config.get("backchannel", {})
        self._clips_dir = bc_cfg.get("clips_dir", "backchannel/clips/heart/")

        # Bridge clip mappings
        bridge_cfg = bc_cfg.get("interrupt_bridge_clips", {})
        self._bridge_clips = {
            "correction": bridge_cfg.get("correction", "got_it.wav"),
            "new_question": bridge_cfg.get("new_question", "sure.wav"),
            "same_topic_redirect": bridge_cfg.get("same_topic_redirect", "sure.wav"),
        }

        # PAUSED state config
        sm_cfg = config.get("speaking_monitor", {})
        self._auto_resume_s = sm_cfg.get("paused_auto_resume_s", 3.0)
        self._resume_phrases = set(
            sm_cfg.get("paused_resume_phrases", list(RESUME_PHRASES))
        )

        # Deferred INJECT — stores transcript of "after this can you..." requests
        # so they fire automatically when the current TTS turn ends.
        self._pending_deferred_task: Optional[str] = None

        self._logger = logging.getLogger("InterruptRouter")

    async def route(self, classified: dict) -> None:
        """
        Route a classified interrupt event to the correct handler.

        Args:
            classified: dict with keys:
                - decision: str — "IGNORE", "INJECT", or "INTERRUPT"
                - interrupt_type: str | None — one of 7 types if INTERRUPT
                - transcript: str — what the user said
                - sentiment: str | None — for IGNORE
                - inject_point: str | None — for INJECT
                - urgency_hint: bool
        """
        decision = classified.get("decision", "IGNORE").upper()
        interrupt_type = (classified.get("interrupt_type") or "").upper()
        transcript = classified.get("transcript", "")

        self._logger.info(
            "Routing: decision=%s type=%s transcript='%s'",
            decision, interrupt_type or "none", transcript[:50],
        )

        if decision == "IGNORE":
            await self._handle_ignore(classified)
        elif decision == "INJECT":
            await self._handle_inject(classified)
        elif decision == "INTERRUPT":
            handler = {
                "STOP": self._handle_stop,
                "PAUSE": self._handle_pause,
                "CORRECTION": self._handle_correction,
                "NEW_QUESTION": self._handle_new_question,
                "SAME_TOPIC_REDIRECT": self._handle_same_topic_redirect,
                "URGENCY": self._handle_urgency,
                "DONE_LISTENING": self._handle_done_listening,
            }.get(interrupt_type)

            if handler:
                await handler(classified)
            else:
                self._logger.warning("Unknown interrupt type: %s — defaulting to PAUSE", interrupt_type)
                await self._handle_pause(classified)
        else:
            self._logger.warning("Unknown decision: %s — defaulting to IGNORE", decision)
            await self._handle_ignore(classified)

    # ── IGNORE ───────────────────────────────────────────────────────────────

    async def _handle_ignore(self, classified: dict) -> None:
        """Resume TTS as if nothing happened. No LLM call."""
        self._gate3_done_event.set()  # stop filler clip
        await self._audio_player.resume()
        self._mic_stream.set_filter_active(True)

        sentiment = classified.get("sentiment", "")
        if sentiment == "positive":
            from core.event_bus import EventType
            await self._event_bus.publish(
                EventType.POSITIVE_REACTION,
                {"transcript": classified.get("transcript", "")},
                source="InterruptRouter",
            )

        from core.session import TurnState
        await self._session.set_state(TurnState.SPEAKING)
        self._logger.info("IGNORE — resumed TTS")

    # ── INJECT ───────────────────────────────────────────────────────────────

    async def _handle_inject(self, classified: dict) -> None:
        """Resume TTS and inject user's text into LLM context.

        If the request is deferred (e.g. 'after this can you tell me...'),
        the transcript is stored for execution once the current TTS turn ends
        rather than being injected mid-stream.
        """
        self._gate3_done_event.set()  # stop filler clip
        self._mic_stream.set_filter_active(True)

        transcript = classified.get("transcript", "")
        inject_point = classified.get("inject_point", "mid_sentence")
        is_deferred = classified.get("deferred", False)

        if is_deferred:
            # Deferred request: play bridge clip first so the user knows we
            # heard them, THEN resume TTS — avoids the clip and TTS overlapping.
            self._pending_deferred_task = transcript
            bridge_clip = self._bridge_clips.get("new_question", "sure.wav")
            bridge_path = os.path.join(self._clips_dir, bridge_clip)
            if os.path.isfile(bridge_path):
                try:
                    await self._audio_player.play_clip(bridge_path)
                except Exception:
                    pass
            await self._audio_player.resume()
            self._logger.info(
                "INJECT (deferred) — stored task '%s', resuming TTS",
                transcript[:60],
            )
        else:
            # Immediate inject: resume TTS, then respond AFTER TTS ends.
            # Store as pending deferred task so _handle_playback_done fires
            # a fresh LLM turn once the current sentence finishes — identical
            # to deferred but without the bridge clip.
            self._pending_deferred_task = transcript
            await self._audio_player.resume()
            self._logger.info(
                "INJECT — queued '%s' for post-TTS LLM response (at %s)",
                transcript[:40], inject_point,
            )

        from core.session import TurnState
        await self._session.set_state(TurnState.SOFT_INJECT)
        # Return to SPEAKING so TTS continues
        await self._session.set_state(TurnState.SPEAKING)

    def pop_deferred_task(self) -> Optional[str]:
        """Return and clear any pending deferred task transcript.

        Called by TurnManager after playback ends to fire deferred requests
        (e.g. 'after this can you tell me the bitcoin price').
        Returns None if no deferred task is pending.
        """
        task = self._pending_deferred_task
        self._pending_deferred_task = None
        return task

    # ── INTERRUPT:STOP ───────────────────────────────────────────────────────

    async def _handle_stop(self, classified: dict) -> None:
        """Complete silence. No bridge clip, no response."""
        self._gate3_done_event.set()
        await self._audio_player.flush()
        # No bridge clip, no LLM call — user wants silence
        from core.session import TurnState
        await self._session.set_state(TurnState.LISTENING)
        self._logger.info("STOP — full silence, LISTENING")

    # ── INTERRUPT:PAUSE ──────────────────────────────────────────────────────

    async def _handle_pause(self, classified: dict) -> None:
        """Freeze TTS in RAM. Resumable without LLM call."""
        self._gate3_done_event.set()
        # TTS already paused from PENDING — do NOT flush
        from core.session import TurnState
        await self._session.set_state(TurnState.PAUSED)
        self._logger.info("PAUSE — TTS frozen, starting auto-resume timer")

        # Start auto-resume timer in background
        asyncio.create_task(self._paused_auto_resume())

    async def _paused_auto_resume(self) -> None:
        """Auto-resume after configured silence duration."""
        from core.session import TurnState
        await asyncio.sleep(self._auto_resume_s)

        if self._session.state != TurnState.PAUSED:
            return  # State already changed by user action

        # Check if user is speaking before auto-resuming
        self._logger.info("Auto-resume check after %.1fs", self._auto_resume_s)
        await self._audio_player.resume()
        self._mic_stream.set_filter_active(True)
        await self._session.set_state(TurnState.SPEAKING)
        self._logger.info("Auto-resumed TTS from PAUSED")

    # ── INTERRUPT:CORRECTION ─────────────────────────────────────────────────

    async def _handle_correction(self, classified: dict) -> None:
        """Flush TTS, play bridge clip, start primary LLM with correction context."""
        self._gate3_done_event.set()
        await self._audio_player.flush()

        # Play bridge clip while LLM starts
        clip_name = self._bridge_clips.get("correction", "got_it.wav")
        clip_path = os.path.join(self._clips_dir, clip_name)
        asyncio.create_task(self._audio_player.play_clip(clip_path))

        from core.session import TurnState
        from core.event_bus import EventType
        await self._session.set_state(TurnState.INTERRUPTED)

        # Add user turn and trigger LLM via TRANSCRIPT_READY
        transcript = classified.get("transcript", "")
        await self._session.add_turn("user", transcript)
        await self._event_bus.publish(
            EventType.TRANSCRIPT_READY,
            {"text": transcript, "turn_added": True, "source": "interrupt_correction"},
            source="InterruptRouter",
        )
        self._logger.info("CORRECTION — got_it.wav + LLM with correction context")

    # ── INTERRUPT:NEW_QUESTION ───────────────────────────────────────────────

    async def _handle_new_question(self, classified: dict) -> None:
        """Flush TTS, play bridge clip, start primary LLM with new query."""
        self._gate3_done_event.set()
        await self._audio_player.flush()

        clip_name = self._bridge_clips.get("new_question", "sure.wav")
        clip_path = os.path.join(self._clips_dir, clip_name)
        asyncio.create_task(self._audio_player.play_clip(clip_path))

        from core.session import TurnState
        from core.event_bus import EventType
        await self._session.set_state(TurnState.INTERRUPTED)

        transcript = classified.get("transcript", "")
        await self._session.add_turn("user", transcript)
        await self._event_bus.publish(
            EventType.TRANSCRIPT_READY,
            {"text": transcript, "turn_added": True, "source": "interrupt_new_question"},
            source="InterruptRouter",
        )
        self._logger.info("NEW_QUESTION — sure.wav + LLM with new query")

    # ── INTERRUPT:SAME_TOPIC_REDIRECT ────────────────────────────────────────

    async def _handle_same_topic_redirect(self, classified: dict) -> None:
        """Flush TTS, play bridge clip, LLM with prior topic context."""
        self._gate3_done_event.set()
        await self._audio_player.flush()

        clip_name = self._bridge_clips.get("same_topic_redirect", "sure.wav")
        clip_path = os.path.join(self._clips_dir, clip_name)
        asyncio.create_task(self._audio_player.play_clip(clip_path))

        from core.session import TurnState
        from core.event_bus import EventType
        await self._session.set_state(TurnState.INTERRUPTED)

        transcript = classified.get("transcript", "")
        await self._session.add_turn("user", transcript)
        await self._event_bus.publish(
            EventType.TRANSCRIPT_READY,
            {"text": transcript, "turn_added": True, "source": "interrupt_redirect"},
            source="InterruptRouter",
        )
        self._logger.info("SAME_TOPIC_REDIRECT — sure.wav + LLM with topic context")

    # ── INTERRUPT:URGENCY ────────────────────────────────────────────────────

    async def _handle_urgency(self, classified: dict) -> None:
        """Immediate flush, no bridge clip, zero delay. LLM at max priority."""
        self._gate3_done_event.set()
        await self._audio_player.flush()
        # No bridge clip — zero delay for urgency

        from core.session import TurnState
        from core.event_bus import EventType
        await self._session.set_state(TurnState.INTERRUPTED)

        transcript = classified.get("transcript", "")
        await self._session.add_turn("user", transcript)
        await self._event_bus.publish(
            EventType.TRANSCRIPT_READY,
            {"text": transcript, "turn_added": True, "source": "interrupt_urgency"},
            source="InterruptRouter",
        )
        self._logger.info("URGENCY — immediate flush, LLM at max priority")

    # ── INTERRUPT:DONE_LISTENING ─────────────────────────────────────────────

    async def _handle_done_listening(self, classified: dict) -> None:
        """Stop TTS at sentence boundary, go to LISTENING. No LLM call."""
        self._gate3_done_event.set()
        # Try to stop at sentence boundary (flush but don't respond)
        await self._audio_player.flush()

        from core.session import TurnState
        await self._session.set_state(TurnState.LISTENING)
        self._logger.info("DONE_LISTENING — stopped, no response needed")

    # ── UTILITY ──────────────────────────────────────────────────────────────

    def is_resume_phrase(self, text: str) -> bool:
        """Check if text is a resume phrase (for PAUSED state)."""
        normalized = text.lower().strip().rstrip(".")
        return normalized in self._resume_phrases
