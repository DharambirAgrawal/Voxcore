"""
VOXCORE — backchannel/cue_detector.py
Phrase boundary detection — finds natural backchannel opportunities.
"""

import asyncio
import logging
import time
import numpy as np

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream

# Number of energy buffer entries for ~500ms at 30ms chunks
_ENERGY_BUFFER_SIZE = 17
# Upper bound for phrase boundary pause (ms) — beyond this, VAD handles it
_MAX_PAUSE_MS = 600


class CueDetector:
    """Detects natural phrase boundaries in user speech for backchannel timing."""

    def __init__(
        self,
        session: Session,
        event_bus: EventBus,
        mic_stream: MicStream,
        config: dict,
    ) -> None:
        self.session: Session = session
        self.event_bus: EventBus = event_bus
        self.mic_stream: MicStream = mic_stream

        # Config — backchannel
        bc_cfg = config.get("backchannel", {})
        self.min_pause_ms: int = bc_cfg.get("min_pause_ms", 350)
        self.min_gap_between_s: int = bc_cfg.get("min_gap_between_s", 8)
        self.min_user_speech_s: int = bc_cfg.get("min_user_speech_s", 2)
        self.energy_threshold: float = bc_cfg.get("energy_threshold", 0.02)

        # Config — audio
        audio_cfg = config.get("audio", {})
        self.sample_rate: int = audio_cfg.get("sample_rate", 16000)
        self.chunk_ms: int = audio_cfg.get("chunk_ms", 30)

        # State
        self._last_backchannel_time: float = 0.0
        self._user_speech_start: float = 0.0
        self._low_energy_start: float = 0.0
        self._energy_buffer: list[float] = []
        self._audio_queue: asyncio.Queue | None = None

        self._logger: logging.Logger = logging.getLogger("CueDetector")

    async def run(self) -> None:
        """Main loop — consume mic audio and detect phrase boundaries."""
        self._audio_queue = self.mic_stream.add_consumer()

        # Register for speech start events
        self.event_bus.subscribe(EventType.SPEECH_START, self._on_speech_start_event)

        self._logger.info("CueDetector running")

        while True:
            try:
                chunk = await self._audio_queue.get()

                # Only analyze during LISTENING state
                if self.session.state != TurnState.LISTENING:
                    continue

                rms = self._compute_rms(chunk)

                # Maintain rolling energy buffer (~500ms window)
                self._energy_buffer.append(rms)
                if len(self._energy_buffer) > _ENERGY_BUFFER_SIZE:
                    self._energy_buffer = self._energy_buffer[-_ENERGY_BUFFER_SIZE:]

                await self._detect_phrase_boundary(rms)

            except asyncio.CancelledError:
                self._logger.info("CueDetector cancelled")
                break
            except Exception:
                self._logger.exception("Error in CueDetector loop")
                await asyncio.sleep(0.1)

    @staticmethod
    def _compute_rms(chunk: np.ndarray) -> float:
        """Compute Root Mean Square energy of an audio chunk."""
        return float(np.sqrt(np.mean(chunk ** 2)))

    async def _detect_phrase_boundary(self, rms: float) -> None:
        """Check if current energy pattern indicates a phrase boundary."""
        now = time.time()

        # Guard: must be in LISTENING state
        if self.session.state != TurnState.LISTENING:
            return

        # Guard: user must have been speaking long enough
        if self._user_speech_start == 0:
            return
        if now - self._user_speech_start < self.min_user_speech_s:
            return

        # Guard: cooldown since last backchannel
        if now - self._last_backchannel_time < self.min_gap_between_s:
            return

        # Detection logic
        if rms < self.energy_threshold:
            # Energy dropped — possible phrase boundary
            if self._low_energy_start == 0:
                self._low_energy_start = now
            else:
                pause_ms = (now - self._low_energy_start) * 1000

                if self.min_pause_ms <= pause_ms <= _MAX_PAUSE_MS:
                    # Phrase boundary detected
                    is_question = self._detect_question_intonation()
                    speech_duration = now - self._user_speech_start
                    avg_energy = (
                        float(np.mean(self._energy_buffer))
                        if self._energy_buffer
                        else 0.0
                    )

                    await self.event_bus.publish(
                        EventType.BACKCHANNEL_OPPORTUNITY,
                        {
                            "is_question": is_question,
                            "energy_level": avg_energy,
                            "speech_duration_s": speech_duration,
                        },
                    )

                    self._last_backchannel_time = now
                    self._low_energy_start = 0
                    self._logger.debug("Backchannel opportunity detected")
        else:
            # Energy above threshold — speech resumed
            self._low_energy_start = 0

    def _detect_question_intonation(self) -> bool:
        """Heuristic: check if energy was rising before pause (upward intonation)."""
        buf = self._energy_buffer
        if len(buf) < 10:
            return False

        # Compare energy at two points before the pause
        # Rising energy before silence suggests question intonation
        recent = buf[-3] if len(buf) >= 3 else 0.0
        earlier = buf[-6] if len(buf) >= 6 else 0.0

        return recent > earlier

    async def _on_speech_start_event(self, data: dict) -> None:
        """Callback for SPEECH_START event — only track when user is actually speaking."""
        # Ignore SPEECH_START fired during AI speaking/thinking/pending states;
        # those come from echo picked up by the mic and must not prime backchannel tracking.
        if self.session.state != TurnState.LISTENING:
            return
        self._on_speech_start()

    def _on_speech_start(self) -> None:
        """Reset tracking state when user starts a new speech segment."""
        self._user_speech_start = time.time()
        self._energy_buffer.clear()
        self._low_energy_start = 0

    def reset(self) -> None:
        """Reset all internal state."""
        self._low_energy_start = 0
        self._energy_buffer.clear()
        self._user_speech_start = 0