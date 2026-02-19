"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE — backchannel/cue_detector.py                         ║
║        PHRASE BOUNDARY DETECTION — FINDS NATURAL BACKCHANNEL OPPORTUNITIES      ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Runs a secondary energy analysis on the raw mic stream (NOT on STT transcripts —
    this runs before STT for zero latency). Uses librosa to detect:
    - RMS energy drops below threshold mid-utterance (phrase boundary)
    - Pauses of 300-600ms that are NOT end-of-turn (shorter than VAD's 400ms silence)
    
    When these patterns appear during LISTENING state and enough time has passed
    since the last backchannel, fires a BACKCHANNEL_OPPORTUNITY event.

    This module runs on a completely independent async track. It consumes the
    same audio stream as VAD but does its own analysis. It never blocks any
    other module.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import time                             # Timing for backchannel cooldown
import numpy as np                      # Audio array operations
import librosa                          # Audio analysis (RMS energy, spectral features)

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: CueDetector
──────────────────────────────────────────────────────────────────────────────────
    Detects natural phrase boundaries in user speech for backchannel timing.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus,
                          mic_stream: MicStream, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — For checking TurnState and speech duration
            - event_bus: EventBus — For publishing BACKCHANNEL_OPPORTUNITY events
            - mic_stream: MicStream — Source of raw audio chunks
            - config: dict — The "backchannel" section from config.yaml:
                - min_pause_ms: int (350)
                - min_gap_between_s: int (8)
                - min_user_speech_s: int (2)
                - energy_threshold: float (0.02)
              AND "audio" section:
                - sample_rate: int (16000)
                - chunk_ms: int (30)
        
        INITIALIZES:
            self.session: Session                = session
            self.event_bus: EventBus             = event_bus
            self.mic_stream: MicStream           = mic_stream
            
            # Config
            self.min_pause_ms: int               = config["backchannel"].get("min_pause_ms", 350)
            self.min_gap_between_s: int          = config["backchannel"].get("min_gap_between_s", 8)
            self.min_user_speech_s: int          = config["backchannel"].get("min_user_speech_s", 2)
            self.energy_threshold: float         = config["backchannel"].get("energy_threshold", 0.02)
            self.sample_rate: int                = config["audio"].get("sample_rate", 16000)
            self.chunk_ms: int                   = config["audio"].get("chunk_ms", 30)
            
            # State
            self._last_backchannel_time: float   = 0.0     # When last backchannel fired
            self._user_speech_start: float       = 0.0     # When user started current speech segment
            self._low_energy_start: float        = 0.0     # When energy first dropped below threshold
            self._energy_buffer: list[float]     = []      # Rolling window of RMS values (last 500ms)
            self._audio_queue: asyncio.Queue     = None    # Consumer queue from mic_stream
            
            self._logger: logging.Logger         = logging.getLogger("CueDetector")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Gets audio consumer queue: self._audio_queue = mic_stream.add_consumer()
            2. Registers callback for SPEECH_START to track user speech onset
            3. Enters infinite loop:
               a. chunk = await self._audio_queue.get()
               b. If session.state != TurnState.LISTENING: continue
               c. rms = self._compute_rms(chunk)
               d. self._energy_buffer.append(rms)
                  Keep only last ~500ms worth (500/30 ≈ 17 entries)
               e. await self._detect_phrase_boundary(rms)

    def _compute_rms(self, chunk: np.ndarray) -> float
        INPUTS:
            - chunk: np.ndarray — Audio chunk, float32, normalized [-1, 1]
        OUTPUT:
            - float — Root Mean Square energy of the chunk
        WHAT IT DOES:
            1. Computes: np.sqrt(np.mean(chunk ** 2))
            2. Returns the RMS value
        
        RMS gives a measure of audio loudness. Low RMS = silence/pause.
        High RMS = speech or noise.

    async def _detect_phrase_boundary(self, rms: float) -> None
        INPUTS:
            - rms: float — Current chunk's RMS energy
        OUTPUT: None (may fire BACKCHANNEL_OPPORTUNITY event)
        WHAT IT DOES:
            GUARD CONDITIONS (all must be true to fire):
                1. Session state is LISTENING (user is talking)
                2. User has been speaking > self.min_user_speech_s seconds
                3. Time since last backchannel > self.min_gap_between_s seconds
            
            DETECTION LOGIC:
                If rms < self.energy_threshold:
                    # Energy dropped — possible phrase boundary
                    If self._low_energy_start == 0:
                        self._low_energy_start = time.time()
                    Else:
                        pause_ms = (time.time() - self._low_energy_start) * 1000
                        If self.min_pause_ms <= pause_ms <= 600:
                            # This is a phrase boundary pause (not end-of-turn)
                            # End-of-turn would be caught by VAD at 400ms silence
                            1. Compute context:
                               is_question = self._detect_question_intonation()
                               speech_duration = time.time() - self._user_speech_start
                            2. Publish BACKCHANNEL_OPPORTUNITY event with data:
                               {"is_question": is_question, "energy_level": avg_energy, "speech_duration_s": speech_duration}
                            3. Set self._last_backchannel_time = time.time()
                            4. Reset self._low_energy_start = 0
                            5. Log: "Backchannel opportunity detected"
                Else:
                    # Energy is above threshold — speech resumed
                    self._low_energy_start = 0

    def _detect_question_intonation(self) -> bool
        INPUTS: None (uses self._energy_buffer)
        OUTPUT: bool — Whether the recent speech pattern suggests a question
        WHAT IT DOES:
            1. Looks at the last 10 entries in self._energy_buffer
            2. If energy was rising before the pause (upward intonation), returns True
            3. Simple heuristic: if energy_buffer[-3] < energy_buffer[-6], likely question
            4. Returns False by default
        
        NOTE: This is a rough heuristic. True pitch analysis would require
              F0 extraction which is more expensive. This simple energy-based
              approach works "well enough" for backchannel selection.

    def _on_speech_start(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Sets self._user_speech_start = time.time()
            2. Clears self._energy_buffer
            3. Resets self._low_energy_start = 0
        
        Called when SPEECH_START event fires (registered in run()).

    def reset(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            Resets all state: _low_energy_start=0, _energy_buffer=[], _user_speech_start=0

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - CueDetector   (class)
═══════════════════════════════════════════════════════════════════════════════════

HOW THIS DIFFERS FROM VAD:
    VAD answers: "Is the user speaking or silent?"
    CueDetector answers: "Is this a natural pause within speech where I should react?"
    
    VAD triggers SPEECH_END (end of turn). CueDetector triggers BACKCHANNEL_OPPORTUNITY
    (natural reaction point within a turn). They look at the same audio but serve
    completely different purposes.
"""
