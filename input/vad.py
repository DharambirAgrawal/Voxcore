"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           VOXCORE — input/vad.py                                ║
║          SILERO VAD WRAPPER — VOICE ACTIVITY DETECTION ON AUDIO CHUNKS          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Wraps the Silero VAD model to process continuous audio chunks from MicStream.
    Fires SPEECH_START when speech begins and SPEECH_END when silence exceeds
    the configured threshold (400ms default). Maintains a rolling audio buffer
    of the current speech segment for submission to STT.

    Runs on CPU. Each VAD inference takes <1ms per 30ms chunk.
    This is the core gatekeeper — nothing downstream happens until VAD says
    "the user stopped talking."

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async main loop
import logging                          # Module logger
import time                             # Tracking speech duration
import numpy as np                      # Audio array operations
import torch                            # Silero VAD runs on PyTorch

from core.event_bus import EventBus, EventType
from input.mic_stream import MicStream

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: VADProcessor
──────────────────────────────────────────────────────────────────────────────────
    Silero VAD wrapper that processes audio chunks and fires speech events.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, mic_stream: MicStream, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For publishing SPEECH_START / SPEECH_END events
            - mic_stream: MicStream — Source of audio chunks
            - config: dict — The "audio" section from config.yaml:
                - sample_rate: int (16000)
                - vad_speech_threshold: float (0.5)
                - vad_silence_threshold: float (0.7)
                - vad_silence_duration_ms: int (400)
        
        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            self.mic_stream: MicStream           = mic_stream
            self.sample_rate: int                = config.get("sample_rate", 16000)
            self.speech_threshold: float         = config.get("vad_speech_threshold", 0.5)
            self.silence_threshold: float        = config.get("vad_silence_threshold", 0.7)
            self.silence_duration_ms: int        = config.get("vad_silence_duration_ms", 400)
            self.chunk_ms: int                   = config.get("chunk_ms", 30)
            
            # Silero VAD model
            self._model: torch.nn.Module         = None  # Loaded in _load_model()
            
            # State tracking
            self._is_speaking: bool              = False   # Currently in speech segment?
            self._speech_start_time: float       = 0.0     # When current speech started
            self._silence_start_time: float      = 0.0     # When silence began (for end-of-turn detection)
            self._audio_buffer: list[np.ndarray] = []      # Accumulates audio during speech
            self._audio_queue: asyncio.Queue     = None    # Consumer queue from mic_stream
            
            self._logger: logging.Logger         = logging.getLogger("VAD")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def _load_model(self) -> None
        INPUTS: None
        OUTPUT: None (sets self._model)
        WHAT IT DOES:
            1. Loads Silero VAD using torch.hub or the silero-vad package:
               model, utils = torch.hub.load(
                   repo_or_dir='snakers4/silero-vad',
                   model='silero_vad',
                   force_reload=False
               )
               Or using the silero_vad package:
               from silero_vad import load_silero_vad
               model = load_silero_vad()
            2. Sets self._model = model
            3. Logs: "Silero VAD model loaded"
        
        NOTE: Model runs on CPU. No GPU needed.
              Inference time: ~0.5ms per 30ms audio chunk.

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever until cancelled)
        WHAT IT DOES:
            1. Calls self._load_model()
            2. Gets audio consumer queue: self._audio_queue = mic_stream.add_consumer()
            3. Enters infinite loop:
               a. chunk = await self._audio_queue.get()
               b. speech_prob = self._get_speech_probability(chunk)
               c. await self._process_probability(speech_prob, chunk)
            4. Never returns — runs until task is cancelled

    def _get_speech_probability(self, chunk: np.ndarray) -> float
        INPUTS:
            - chunk: np.ndarray — Audio chunk, shape (480,), float32, normalized [-1, 1]
        OUTPUT:
            - float — Speech probability between 0.0 and 1.0
        WHAT IT DOES:
            1. Converts numpy array to torch tensor:
               tensor = torch.from_numpy(chunk).float()
            2. Runs VAD inference:
               prob = self._model(tensor, self.sample_rate).item()
            3. Returns prob

    async def _process_probability(self, speech_prob: float, chunk: np.ndarray) -> None
        INPUTS:
            - speech_prob: float — VAD speech probability for this chunk
            - chunk: np.ndarray — The audio chunk
        OUTPUT: None (fires events, updates state)
        WHAT IT DOES:
            If NOT currently speaking (self._is_speaking == False):
                If speech_prob > self.speech_threshold:
                    1. Set self._is_speaking = True
                    2. Set self._speech_start_time = time.time()
                    3. Clear self._audio_buffer = []
                    4. Append chunk to self._audio_buffer
                    5. Publish SPEECH_START event
                    6. Log: "Speech started"
            
            If currently speaking (self._is_speaking == True):
                1. Always append chunk to self._audio_buffer
                
                If speech_prob < (1.0 - self.silence_threshold):
                    # This chunk is silence
                    If self._silence_start_time == 0:
                        Set self._silence_start_time = time.time()
                    
                    silence_duration = (time.time() - self._silence_start_time) * 1000
                    
                    If silence_duration >= self.silence_duration_ms:
                        # End of turn detected!
                        a. Set self._is_speaking = False
                        b. audio_bytes = self._buffer_to_bytes()
                        c. duration = time.time() - self._speech_start_time
                        d. Publish SPEECH_END event with data:
                           {"audio_buffer": audio_bytes, "duration_s": duration}
                        e. Reset: self._silence_start_time = 0, self._audio_buffer = []
                        f. Log: "Speech ended ({duration:.1f}s)"
                Else:
                    # Still speaking — reset silence timer
                    self._silence_start_time = 0

    def _buffer_to_bytes(self) -> bytes
        INPUTS: None (uses self._audio_buffer)
        OUTPUT: bytes — Concatenated audio buffer as raw PCM bytes (int16)
        WHAT IT DOES:
            1. Concatenates all chunks: full_audio = np.concatenate(self._audio_buffer)
            2. Converts float32 to int16: audio_int16 = (full_audio * 32767).astype(np.int16)
            3. Returns audio_int16.tobytes()
        
        This is what gets sent to STT for transcription.

    @property
    def is_speaking(self) -> bool
        OUTPUT: bool — Whether VAD currently detects active speech

    @property
    def speech_duration(self) -> float
        OUTPUT: float — Duration in seconds of current speech segment (0.0 if not speaking)
        WHAT IT DOES:
            If self._is_speaking:
                return time.time() - self._speech_start_time
            return 0.0

    def reset(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Resets all state: _is_speaking=False, _audio_buffer=[], _silence_start_time=0
            2. Resets Silero VAD internal state: self._model.reset_states()
            3. Used after interruption to clean up

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - VADProcessor   (class)
═══════════════════════════════════════════════════════════════════════════════════

PERFORMANCE:
    - Silero VAD inference: ~0.5ms per 30ms chunk on CPU
    - Total VAD overhead: < 2% of real-time audio
    - Memory: ~40MB for model weights
    - No GPU required
"""




"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           VOXCORE — input/vad.py                                ║
║          SILERO VAD WRAPPER — VOICE ACTIVITY DETECTION ON AUDIO CHUNKS          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# import asyncio
# import logging
# import time
# from typing import List, Optional

# import numpy as np
# import torch

# from core.event_bus import EventBus, EventType
# from core.session import Session, TurnState
# from input.mic_stream import MicStream


# class VADProcessor:
#     """Silero VAD wrapper that processes audio chunks and fires speech events."""

#     def __init__(
#         self, event_bus: EventBus, mic_stream: MicStream, config: dict,
#         session: Session = None,
#     ) -> None:
#         self.event_bus: EventBus = event_bus
#         self.mic_stream: MicStream = mic_stream
#         self.session: Optional[Session] = session
#         self.sample_rate: int = config.get("sample_rate", 16000)
#         self.speech_threshold: float = config.get("vad_speech_threshold", 0.5)
#         self.silence_threshold: float = config.get("vad_silence_threshold", 0.7)
#         self.silence_duration_ms: int = config.get("vad_silence_duration_ms", 400)
#         self.chunk_ms: int = config.get("chunk_ms", 30)
#         # Minimum RMS multiplier — chunk must exceed noise_floor × this to be
#         # considered speech.  Filters keyboard clicks, throat clears, etc.
#         self._min_rms_multiplier: float = config.get("vad_min_rms_multiplier", 5)

#         # Silero VAD model — loaded lazily in _load_model()
#         self._model: Optional[torch.nn.Module] = None

#         # State tracking
#         self._is_speaking: bool = False
#         self._speech_start_time: float = 0.0
#         self._silence_start_time: float = 0.0
#         self._audio_buffer: List[np.ndarray] = []
#         self._audio_queue: Optional[asyncio.Queue] = None

#         # Post-speaking cooldown — ignore mic for N ms after AI stops speaking
#         # to let speaker tail / reverb decay before processing audio.
#         self._post_speak_cooldown_ms: float = 700.0
#         self._last_speaking_end: float = 0.0
#         self._was_speaking: bool = False

#         self._logger: logging.Logger = logging.getLogger("VAD")

#     def _load_model(self) -> None:
#         """Load the Silero VAD model (CPU only)."""
#         try:
#             model, _ = torch.hub.load(
#                 repo_or_dir="snakers4/silero-vad",
#                 model="silero_vad",
#                 force_reload=False,
#             )
#         except Exception:
#             # Fallback to the silero_vad package if torch.hub fails
#             from silero_vad import load_silero_vad  # type: ignore[import-untyped]

#             model = load_silero_vad()

#         self._model = model
#         self._logger.info("Silero VAD model loaded")

#     async def run(self) -> None:
#         """Process audio chunks forever, firing speech start/end events."""
#         self._load_model()
#         self._audio_queue = self.mic_stream.add_consumer()
#         audio_queue = self._audio_queue
#         assert audio_queue is not None, "Audio queue not initialized"

#         while True:
#             chunk: np.ndarray = await audio_queue.get()

#             # ── Echo suppression: ignore mic input while AI is speaking ──
#             # Without hardware AEC, the mic picks up TTS audio from speakers.
#             # Suppress VAD entirely during SPEAKING to avoid false triggers.
#             ai_speaking = (
#                 self.session is not None
#                 and self.session.state == TurnState.SPEAKING
#             )
#             if ai_speaking:
#                 # Track that we were in SPEAKING so we can apply cooldown later
#                 self._was_speaking = True
#                 # Reset Silero internal state so stale activations don't carry over
#                 if self._model is not None:
#                     self._model.reset_states()
#                 continue

#             # ── Post-speaking cooldown ──
#             # Right after SPEAKING→LISTENING, residual speaker reverb bleeds
#             # into the mic for a few hundred ms. Ignore audio during this window.
#             if self._was_speaking:
#                 self._was_speaking = False
#                 self._last_speaking_end = time.time()
#                 # Also reset any in-progress speech detection
#                 if self._is_speaking:
#                     self._is_speaking = False
#                     self._audio_buffer = []
#                     self._silence_start_time = 0.0
#                 self._logger.debug("Post-speaking cooldown started (%.0fms)", self._post_speak_cooldown_ms)

#             if self._last_speaking_end > 0:
#                 elapsed_ms = (time.time() - self._last_speaking_end) * 1000.0
#                 if elapsed_ms < self._post_speak_cooldown_ms:
#                     continue  # Still in cooldown — discard chunk
#                 else:
#                     self._last_speaking_end = 0.0  # Cooldown expired

#             speech_prob = self.get_speech_probability(chunk)

#             # Prevent false starts from low-energy transients (keyboard
#             # clicks, throat clears) by requiring RMS to be significantly
#             # above the calibrated noise floor.
#             rms = float(np.sqrt(np.mean(chunk**2)))
#             if rms < self.mic_stream.noise_floor * self._min_rms_multiplier:
#                 speech_prob = 0.0
                
#             await self._process_probability(speech_prob, chunk)

#     def get_speech_probability(self, chunk: np.ndarray) -> float:
#         """Run Silero VAD inference on a single audio chunk."""
#         tensor = torch.from_numpy(chunk).float()
#         prob: float = self._model(tensor, self.sample_rate).item()  # type: ignore[union-attr]
#         return prob

#     async def _process_probability(
#         self, speech_prob: float, chunk: np.ndarray
#     ) -> None:
#         """Update speech state machine and fire events as needed."""
#         if not self._is_speaking:
#             # ── Waiting for speech to start ──
#             if speech_prob > self.speech_threshold:
#                 self._is_speaking = True
#                 self._speech_start_time = time.time()
#                 self._audio_buffer = []
#                 self._audio_buffer.append(chunk)
#                 self._silence_start_time = 0.0
#                 await self.event_bus.publish(EventType.SPEECH_START, {})
#                 self._logger.info("Speech started")
#         else:
#             # ── Currently in a speech segment ──
#             self._audio_buffer.append(chunk)

#             # If we were doing 1.0 - 0.7 = 0.3, it's very hard to hit if there's noise.
#             # So let's use a fixed threshold of 0.35, or max(0.35, 1.0 - self.silence_threshold)
#             effective_silence_prob = max(0.35, 1.0 - self.silence_threshold)
#             if speech_prob < effective_silence_prob:
#                 # This chunk is silence
#                 if self._silence_start_time == 0.0:
#                     self._silence_start_time = time.time()

#                 silence_duration_ms = (
#                     time.time() - self._silence_start_time
#                 ) * 1000.0

#                 if silence_duration_ms >= self.silence_duration_ms:
#                     # End of turn detected
#                     self._is_speaking = False
#                     audio_bytes = self._buffer_to_bytes()
#                     duration = time.time() - self._speech_start_time

#                     await self.event_bus.publish(
#                         EventType.SPEECH_END,
#                         {
#                             "audio_buffer": audio_bytes,
#                             "duration_s": duration,
#                         },
#                     )

#                     self._logger.info("Speech ended (%.1fs)", duration)
#                     self._silence_start_time = 0.0
#                     self._audio_buffer = []
#             else:
#                 # Still speaking — reset silence timer
#                 self._silence_start_time = 0.0

#     def _buffer_to_bytes(self) -> bytes:
#         """Concatenate buffered float32 chunks and convert to int16 PCM bytes."""
#         full_audio = np.concatenate(self._audio_buffer)
#         audio_int16 = (full_audio * 32767).astype(np.int16)
#         return audio_int16.tobytes()

#     @property
#     def is_speaking(self) -> bool:
#         """Whether VAD currently detects active speech."""
#         return self._is_speaking

#     @property
#     def speech_duration(self) -> float:
#         """Duration in seconds of the current speech segment (0.0 if silent)."""
#         if self._is_speaking:
#             return time.time() - self._speech_start_time
#         return 0.0

#     def reset(self) -> None:
#         """Reset all VAD state — used after interruption."""
#         self._is_speaking = False
#         self._audio_buffer = []
#         self._silence_start_time = 0.0
#         self._speech_start_time = 0.0
#         model = self._model
#         if model is not None:
#             model.reset_states()
#         self._logger.debug("VAD state reset")


"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           VOXCORE — input/vad.py                                ║
║          SILERO VAD WRAPPER — VOICE ACTIVITY DETECTION ON AUDIO CHUNKS          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
import time
from collections import deque
from typing import List, Optional

import numpy as np
import torch

from core.event_bus import EventBus, EventType
from core.session import Session, TurnState
from input.mic_stream import MicStream


class VADProcessor:
    """Silero VAD wrapper that processes audio chunks and fires speech events.

    State machine
    ─────────────
        IDLE ──(enough speech frames)──► CONFIRMING ──(ratio met)──► SPEAKING
          ▲                                   │                         │
          │                        (ratio drops)                  (silence 500ms)
          └───────────────────────────────────┘                         │
          ▲                                                             │
          │         (segment < 300ms → discard silently)                │
          └─────────────────────────────────────────────────────────────┘
          ▲                                                             │
          │         (segment ≥ 300ms → fire SPEECH_END)                 │
          └─────────────────────────────────────────────────────────────┘

    Confirmation uses RATIO-BASED counting, not consecutive frames.
    Speech naturally dips between syllables — consecutive would never confirm.
    """

    def __init__(
        self,
        event_bus: EventBus,
        mic_stream: MicStream,
        config: dict,
        session: Session = None,
    ) -> None:
        self.event_bus: EventBus = event_bus
        self.mic_stream: MicStream = mic_stream
        self.session: Optional[Session] = session
        self.sample_rate: int = config.get("sample_rate", 16000)
        self.speech_threshold: float = config.get("vad_speech_threshold", 0.75)
        self.silence_threshold: float = config.get("vad_silence_threshold", 0.6)
        self.silence_duration_ms: int = config.get("vad_silence_duration_ms", 500)
        self.chunk_ms: int = config.get("chunk_ms", 30)

        # ── Confirmation: ratio-based, NOT consecutive ──
        # "4 of last 6 frames must be speech" ≈ 67% hit rate over ~180ms
        self._confirm_window_size: int = config.get("vad_confirm_window", 6)
        self._confirm_min_hits: int = config.get("vad_confirm_hits", 4)

        # ── Minimum segment duration — discard segments shorter than this ──
        self._min_speech_duration_ms: float = config.get(
            "vad_min_speech_duration_ms", 300
        )

        # ── RMS energy gate ──
        self._min_rms_multiplier: float = config.get("vad_min_rms_multiplier", 5)

        # Silero VAD model
        self._model: Optional[torch.nn.Module] = None

        # ── State tracking ──
        self._is_speaking: bool = False
        self._speech_start_time: float = 0.0
        self._silence_start_time: float = 0.0
        self._audio_buffer: List[np.ndarray] = []
        self._audio_queue: Optional[asyncio.Queue] = None

        # ── Confirmation state: sliding window of recent speech/silence ──
        self._confirming: bool = False
        self._confirm_ring: deque = deque(maxlen=self._confirm_window_size)
        self._pre_speech_buffer: List[np.ndarray] = []
        self._confirm_start_time: float = 0.0

        # ── Echo suppression ──
        self._post_speak_cooldown_ms: float = 1500.0
        self._last_speaking_end: float = 0.0
        self._was_speaking: bool = False

        self._logger: logging.Logger = logging.getLogger("VAD")

    # ─────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the Silero VAD model (CPU only)."""
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
        except Exception:
            from silero_vad import load_silero_vad  # type: ignore[import-untyped]
            model = load_silero_vad()

        self._model = model
        self._logger.info("Silero VAD model loaded")

    # ─────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Process audio chunks forever, firing speech start/end events."""
        self._load_model()
        self._audio_queue = self.mic_stream.add_consumer()
        audio_queue = self._audio_queue
        assert audio_queue is not None

        self._logger.info(
            "VAD started (threshold=%.2f, confirm=%d/%d frames, "
            "min_segment=%dms, silence=%dms, rms_mult=%.1f)",
            self.speech_threshold,
            self._confirm_min_hits,
            self._confirm_window_size,
            int(self._min_speech_duration_ms),
            self.silence_duration_ms,
            self._min_rms_multiplier,
        )

        while True:
            chunk: np.ndarray = await audio_queue.get()

            # ── Echo suppression: skip while AI is speaking ──
            ai_speaking = (
                self.session is not None
                and self.session.state == TurnState.SPEAKING
            )
            if ai_speaking:
                self._was_speaking = True
                if self._model is not None:
                    self._model.reset_states()
                continue

            # ── Post-speaking cooldown ──
            if self._was_speaking:
                self._was_speaking = False
                self._last_speaking_end = time.time()
                self._reset_to_idle()
                self._logger.debug(
                    "Post-speaking cooldown (%.0fms)",
                    self._post_speak_cooldown_ms,
                )

            if self._last_speaking_end > 0:
                elapsed_ms = (time.time() - self._last_speaking_end) * 1000.0
                if elapsed_ms < self._post_speak_cooldown_ms:
                    continue
                else:
                    self._last_speaking_end = 0.0

            # ── VAD inference ──
            speech_prob = self.get_speech_probability(chunk)

            # ── RMS energy gate ──
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            noise_gate = self.mic_stream.noise_floor * self._min_rms_multiplier
            if rms < noise_gate:
                speech_prob = 0.0

            await self._process_probability(speech_prob, chunk)

    # ─────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────

    def get_speech_probability(self, chunk: np.ndarray) -> float:
        """Run Silero VAD inference on a single audio chunk."""
        tensor = torch.from_numpy(chunk).float()
        prob: float = self._model(tensor, self.sample_rate).item()  # type: ignore[union-attr]
        return prob

    # ─────────────────────────────────────────────────────────────────────
    # State machine
    # ─────────────────────────────────────────────────────────────────────

    async def _process_probability(
        self, speech_prob: float, chunk: np.ndarray
    ) -> None:
        """Three-state machine: IDLE → CONFIRMING → SPEAKING → IDLE."""

        is_speech_frame = speech_prob > self.speech_threshold

        # ═════════════════════════════════════════════════════════════════
        # STATE: IDLE — waiting for first speech frame
        # ═════════════════════════════════════════════════════════════════
        if not self._is_speaking and not self._confirming:
            if is_speech_frame:
                self._confirming = True
                self._confirm_ring.clear()
                self._confirm_ring.append(True)
                self._confirm_start_time = time.time()
                self._pre_speech_buffer = [chunk]
                self._logger.debug(
                    "Speech candidate (prob=%.3f), confirming [1/%d]...",
                    speech_prob,
                    self._confirm_min_hits,
                )

        # ═════════════════════════════════════════════════════════════════
        # STATE: CONFIRMING — collecting evidence (ratio-based)
        # ═════════════════════════════════════════════════════════════════
        elif self._confirming:
            self._confirm_ring.append(is_speech_frame)
            self._pre_speech_buffer.append(chunk)

            hits = sum(self._confirm_ring)
            total = len(self._confirm_ring)

            if hits >= self._confirm_min_hits:
                # ✓ CONFIRMED — enough speech frames in window
                self._confirming = False
                self._is_speaking = True
                self._speech_start_time = self._confirm_start_time
                self._audio_buffer = list(self._pre_speech_buffer)
                self._pre_speech_buffer = []
                self._silence_start_time = 0.0

                elapsed_ms = (time.time() - self._confirm_start_time) * 1000
                await self.event_bus.publish(EventType.SPEECH_START, {})
                self._logger.info(
                    "Speech started (confirmed %d/%d in %.0fms)",
                    hits,
                    total,
                    elapsed_ms,
                )

            elif total >= self._confirm_window_size and hits < self._confirm_min_hits:
                # ✗ Window full but not enough speech — reject
                elapsed_ms = (time.time() - self._confirm_start_time) * 1000
                self._logger.debug(
                    "Speech candidate rejected (%d/%d hits in %.0fms, "
                    "prob=%.3f)",
                    hits,
                    total,
                    elapsed_ms,
                    speech_prob,
                )
                self._confirming = False
                self._confirm_ring.clear()
                self._pre_speech_buffer = []

            # If window not yet full, keep collecting (don't reject early)

        # ═════════════════════════════════════════════════════════════════
        # STATE: SPEAKING — accumulating audio, watching for end-of-turn
        # ═════════════════════════════════════════════════════════════════
        elif self._is_speaking:
            self._audio_buffer.append(chunk)

            effective_silence_prob = max(0.35, 1.0 - self.silence_threshold)

            if speech_prob < effective_silence_prob:
                # This chunk is silence
                if self._silence_start_time == 0.0:
                    self._silence_start_time = time.time()

                silence_ms = (time.time() - self._silence_start_time) * 1000.0

                if silence_ms >= self.silence_duration_ms:
                    # End of turn — check minimum duration
                    duration = time.time() - self._speech_start_time
                    duration_ms = duration * 1000.0

                    if duration_ms < self._min_speech_duration_ms:
                        # ✗ TOO SHORT — discard
                        self._logger.debug(
                            "Speech discarded (%.0fms < %.0fms min)",
                            duration_ms,
                            self._min_speech_duration_ms,
                        )
                        self._is_speaking = False
                        self._audio_buffer = []
                        self._silence_start_time = 0.0
                    else:
                        # ✓ VALID — fire SPEECH_END
                        self._is_speaking = False
                        audio_bytes = self._buffer_to_bytes()

                        await self.event_bus.publish(
                            EventType.SPEECH_END,
                            {
                                "audio_buffer": audio_bytes,
                                "duration_s": duration,
                            },
                        )
                        self._logger.info("Speech ended (%.1fs)", duration)
                        self._silence_start_time = 0.0
                        self._audio_buffer = []
            else:
                # Still speaking
                self._silence_start_time = 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _reset_to_idle(self) -> None:
        """Reset all state back to IDLE."""
        self._is_speaking = False
        self._confirming = False
        self._audio_buffer = []
        self._pre_speech_buffer = []
        self._confirm_ring.clear()
        self._silence_start_time = 0.0
        self._confirm_start_time = 0.0

    def _buffer_to_bytes(self) -> bytes:
        """Concatenate buffered float32 chunks → int16 PCM bytes."""
        full_audio = np.concatenate(self._audio_buffer)
        audio_int16 = (full_audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def speech_duration(self) -> float:
        if self._is_speaking:
            return time.time() - self._speech_start_time
        return 0.0

    def reset(self) -> None:
        """Reset all VAD state — used after interruption."""
        self._reset_to_idle()
        self._speech_start_time = 0.0
        model = self._model
        if model is not None:
            model.reset_states()
        self._logger.debug("VAD state reset")