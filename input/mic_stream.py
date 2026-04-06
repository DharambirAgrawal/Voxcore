"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        VOXCORE — input/mic_stream.py                            ║
║             CONTINUOUS MICROPHONE CAPTURE — 16kHz MONO PCM STREAM              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Opens a sounddevice input stream at 16kHz, mono, int16 format and continuously
    produces 30ms audio chunks into an asyncio queue. This queue is consumed by
    both the VAD processor and the backchannel cue detector.

    Never blocks. Runs as a continuous background async task. On startup, performs
    a 0.5-second silence check to calibrate the noise floor for VAD.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async queue for audio chunks
import logging                          # Module logger
import numpy as np                      # Audio array manipulation (int16 → float32)
import sounddevice as sd                # Microphone capture

from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: MicStream
──────────────────────────────────────────────────────────────────────────────────
    Continuous microphone audio capture producing fixed-size chunks.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — Shared event bus (not heavily used here, but available)
            - config: dict — The "audio" section from config.yaml:
                - sample_rate: int (16000)
                - chunk_ms: int (30)
                - noise_floor_calibration_s: float (0.5)
        
        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            self.sample_rate: int                = config.get("sample_rate", 16000)
            self.chunk_ms: int                   = config.get("chunk_ms", 30)
            self.chunk_samples: int              = int(self.sample_rate * self.chunk_ms / 1000)
                                                   # = 480 samples for 16kHz @ 30ms
            self.calibration_duration: float     = config.get("noise_floor_calibration_s", 0.5)
            self.noise_floor: float              = 0.0  # Set during calibration
            self.audio_queue: asyncio.Queue      = asyncio.Queue(maxsize=200)
                                                   # 200 chunks = 6 seconds of buffered audio
            self._stream: Optional[sd.InputStream] = None
            self._running: bool                  = False
            self._logger: logging.Logger         = logging.getLogger("MicStream")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever until cancelled)
        WHAT IT DOES:
            1. Calls await self._calibrate_noise_floor()
            2. Creates sd.InputStream with:
               - samplerate=self.sample_rate
               - channels=1 (mono)
               - dtype='int16'
               - blocksize=self.chunk_samples
               - callback=self._audio_callback
            3. Starts the stream: self._stream.start()
            4. Sets self._running = True
            5. Logs: "Mic stream started (16kHz, mono, {chunk_ms}ms chunks)"
            6. Enters await asyncio.Event().wait() — blocks forever (sounddevice
               runs in its own thread and pushes to the callback)
        
        NOTE: sounddevice's callback runs in a separate C thread. We use
              a thread-safe mechanism (loop.call_soon_threadsafe) to put
              chunks into the asyncio queue.

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info: Any, status: sd.CallbackFlags) -> None
        INPUTS:
            - indata: np.ndarray — Raw audio data, shape (chunk_samples, 1), dtype int16
            - frames: int — Number of frames (should equal chunk_samples)
            - time_info: CData — Timing info from PortAudio
            - status: sd.CallbackFlags — Error/overflow flags
        OUTPUT: None
        WHAT IT DOES:
            1. If status has input overflow, logs warning
            2. Copies the audio data: chunk = indata[:, 0].copy()  (flatten mono)
            3. Converts to float32 normalized: chunk_f32 = chunk.astype(np.float32) / 32768.0
            4. Puts chunk_f32 into self.audio_queue using:
               loop.call_soon_threadsafe(self.audio_queue.put_nowait, chunk_f32)
               — If queue is full, drops the chunk (logs warning)
        
        CRITICAL: This callback runs in a PortAudio C thread. It must NOT:
            - Await anything
            - Do heavy computation
            - Allocate large objects
            It should ONLY copy data and push to the queue.

    async def _calibrate_noise_floor(self) -> None
        INPUTS: None
        OUTPUT: None (sets self.noise_floor)
        WHAT IT DOES:
            1. Logs: "Calibrating noise floor ({calibration_duration}s)..."
            2. Records self.calibration_duration seconds of audio using sd.rec()
            3. Computes RMS energy: np.sqrt(np.mean(audio ** 2))
            4. Sets self.noise_floor = rms_energy * 1.5  (margin above ambient)
            5. Logs: "Noise floor set to {noise_floor:.6f}"
        
        This gives the VAD and backchannel cue detector a baseline for
        distinguishing speech from ambient noise.

    async def get_chunk(self) -> np.ndarray
        INPUTS: None
        OUTPUT: np.ndarray — One audio chunk, shape (chunk_samples,), dtype float32, normalized [-1, 1]
        WHAT IT DOES:
            1. Awaits self.audio_queue.get()
            2. Returns the chunk
        
        USED BY: VADProcessor.run() and CueDetector.run() — they both
                 call this in their main loops to get audio frames.
        
        NOTE: Multiple consumers can read from the same queue only if we
              fan out. Implementation should use a broadcast pattern:
              mic_stream publishes chunks to event_bus, or maintain
              multiple queues (one per consumer). Recommended: maintain
              self._consumer_queues: list[asyncio.Queue] and push to all.

    def add_consumer(self) -> asyncio.Queue
        INPUTS: None
        OUTPUT: asyncio.Queue — A new queue that will receive all audio chunks
        WHAT IT DOES:
            1. Creates a new asyncio.Queue(maxsize=200)
            2. Appends to self._consumer_queues
            3. Returns the queue
        
        This allows multiple modules (VAD, backchannel) to each get their own
        copy of every audio chunk without interfering with each other.

    async def stop(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Sets self._running = False
            2. Stops the sounddevice stream: self._stream.stop()
            3. Closes the stream: self._stream.close()
            4. Logs: "Mic stream stopped"

    @property
    def is_running(self) -> bool
        OUTPUT: bool — Whether the mic stream is currently active

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - MicStream    (class)
═══════════════════════════════════════════════════════════════════════════════════

NOTES ON AUDIO FORMAT:
    - Input: 16kHz, mono, int16 (from sounddevice)
    - Internal: float32 normalized [-1.0, 1.0] (for VAD and analysis)
    - For STT: converted back to int16 and wrapped in WAV by stt.py
    - chunk_samples = sample_rate * chunk_ms / 1000 = 16000 * 30 / 1000 = 480 samples/chunk
    - Each chunk represents 30ms of audio
    - Queue holds 200 chunks = 6 seconds of buffered audio
"""

"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        VOXCORE — input/mic_stream.py                            ║
║             CONTINUOUS MICROPHONE CAPTURE — 16kHz MONO PCM STREAM              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import collections
import logging
from typing import Any, Optional, List

import numpy as np
import sounddevice as sd

from core.event_bus import EventBus

# V2: Default correlation threshold for echo suppression (legacy fallback)
DEFAULT_ECHO_CORRELATION_THRESHOLD = 0.7


class MicStream:
    """Continuous microphone audio capture producing fixed-size chunks.

    V2 additions:
    - Cross-correlation echo suppression: maintains a reference buffer of
      recently played audio. When an incoming mic chunk correlates strongly
      (above threshold) with the reference, it's treated as echo and dropped.

    V3 additions:
    - AriaVoiceFilter integration: speaker-identity echo rejection (Layer 2)
    - Gate0EchoCheck integration: spectral+temporal echo gate (Layer 3)
    - set_filter_active(): scoped toggle — filters ON only during real TTS playback
    - last_200ms(): returns recent audio for interrupt context VAD
    """

    def __init__(self, event_bus: EventBus, config: dict, echo_config: dict | None = None) -> None:
        self.event_bus: EventBus = event_bus
        self.sample_rate: int = config.get("sample_rate", 16000)
        self.chunk_ms: int = config.get("chunk_ms", 30)
        self.chunk_samples: int = int(self.sample_rate * self.chunk_ms / 1000)
        self.calibration_duration: float = config.get("noise_floor_calibration_s", 0.5)
        self.noise_floor: float = 0.0
        self._consumer_queues: List[asyncio.Queue] = []
        self._raw_consumer_queues: List[asyncio.Queue] = []  # V2: bypass echo suppression
        self._stream: Optional[sd.InputStream] = None
        self._running: bool = False
        self._logger: logging.Logger = logging.getLogger("MicStream")
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # ── V3: Echo suppression config ──────────────────────────────────
        _echo_cfg = echo_config or {}

        self._logger.info(
            "Legacy AEC disabled — using denoiser + AriaVoiceFilter + Gate0"
        )

        # ── V3: Four-layer echo suppression filters ──────────────────────
        # Layer 2: AriaVoiceFilter (speaker-identity rejection)
        # Layer 3: Gate0EchoCheck (spectral+temporal echo gate)
        # Set externally via set_v3_filters() after construction in main.py
        self._aria_filter = None    # AriaVoiceFilter instance
        self._gate0 = None          # Gate0EchoCheck instance

        # V3: Scoped filter toggle — only active when TTS plays real speech
        self._filter_active: bool = False

        # V3: Rolling buffer of recent audio for last_200ms()
        recent_samples = int(self.sample_rate * 0.2)  # 200ms
        self._recent_audio: collections.deque = collections.deque(maxlen=recent_samples)

        # ── V4: WebSocket audio source ───────────────────────────────────
        # When set, read from this asyncio queue instead of sounddevice.
        # When None (default), use sounddevice as today.
        self._audio_source_queue: Optional[asyncio.Queue] = None

    async def run(self) -> None:
        """Start mic capture and run forever until cancelled."""
        self._loop = asyncio.get_running_loop()

        # ── V4: WebSocket mode — skip sounddevice entirely ───────────────
        if self._audio_source_queue is not None:
            self._running = True
            self._logger.info(
                "Mic stream started (WebSocket source, 16kHz, mono, %dms chunks)",
                self.chunk_ms,
            )
            await self._ws_audio_loop()
            return

        # ── Original sounddevice path — UNCHANGED ────────────────────────
        await self._calibrate_noise_floor()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        stream = self._stream
        assert stream is not None, "Stream could not be initialized"
        stream.start()
        self._running = True
        self._logger.info(
            "Mic stream started (16kHz, mono, %dms chunks)", self.chunk_ms
        )

        # Block forever — sounddevice pushes audio via its own C thread callback
        await asyncio.Event().wait()

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called from PortAudio C thread. Must be fast, no awaits, no heavy work.
        Copies data and pushes float32 normalised chunks to all consumer queues.

        V2: Performs cross-correlation echo check before distributing.
        """
        if status.input_overflow:
            self._logger.warning("Mic input overflow detected")

        # Flatten mono and copy
        chunk: np.ndarray = indata[:, 0].copy()
        chunk_f32: np.ndarray = chunk.astype(np.float32) / 32768.0

        loop = self._loop
        if loop is None:
            return

        # ── V3: Always update recent audio buffer (for last_200ms()) ─────
        self._recent_audio.extend(chunk_f32.tolist())

        # ══════════════════════════════════════════════════════════════════
        # V3 FIX: Fan-out to RAW consumers FIRST, BEFORE echo filtering.
        # InterruptionDetector uses a raw consumer so Gate0 blocking
        # can't fragment its duration accumulator.
        # ══════════════════════════════════════════════════════════════════
        for q in self._raw_consumer_queues:
            def _enqueue_raw(queue: asyncio.Queue = q) -> None:
                try:
                    queue.put_nowait(chunk_f32)
                except asyncio.QueueFull:
                    pass  # drop silently — raw consumer overwhelmed
            loop.call_soon_threadsafe(_enqueue_raw)

        # ══════════════════════════════════════════════════════════════════
        # V3: Multi-layer echo suppression (Layer 2 + Layer 3)
        # Only apply when filter_active is True (TTS real speech playing)
        # ══════════════════════════════════════════════════════════════════
        if self._filter_active:
            # Layer 2: AriaVoiceFilter — speaker-identity rejection
            if self._aria_filter is not None:
                try:
                    if self._aria_filter.is_aria_echo(chunk_f32):
                        self._logger.debug("AriaVoiceFilter BLOCKED chunk")
                        # V3 FIX: Don't return — still need to skip filtered consumers
                        # but raw consumers already got this chunk above
                        for q in self._consumer_queues:
                            pass  # Intentionally skip filtered consumers
                        return
                except Exception:
                    pass  # Filter error — let audio through

            # Layer 3: Gate0EchoCheck — spectral+temporal echo gate
            if self._gate0 is not None:
                try:
                    if self._gate0.is_echo(chunk_f32):
                        self._logger.debug("Gate0 BLOCKED echo chunk")
                        return  # Drop for filtered consumers only — raw got it above
                except Exception:
                    pass  # Filter error — let audio through

        # Fan-out to every registered (echo-filtered) consumer queue
        for q in self._consumer_queues:

            def _enqueue_consumer(queue: asyncio.Queue = q) -> None:
                try:
                    queue.put_nowait(chunk_f32)
                except asyncio.QueueFull:
                    self._logger.warning("Consumer audio queue full — dropping chunk")

            loop.call_soon_threadsafe(_enqueue_consumer)

    async def _calibrate_noise_floor(self) -> None:
        """Record a short silence segment to estimate ambient noise RMS."""
        self._logger.info(
            "Calibrating noise floor (%.1fs)...", self.calibration_duration
        )

        num_samples = int(self.sample_rate * self.calibration_duration)
        audio = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
        )
        # sd.rec is non-blocking; wait in a thread-friendly way
        await asyncio.get_running_loop().run_in_executor(None, sd.wait)

        audio_f32 = audio.astype(np.float32) / 32768.0
        rms_energy: float = float(np.sqrt(np.mean(audio_f32 ** 2)))
        # Use 2.5× ambient RMS as noise floor — generous margin to reject
        # background noise while still catching real speech.
        self.noise_floor = rms_energy * 2.5
        self._logger.info("Noise floor set to %.6f (ambient RMS=%.6f)", self.noise_floor, rms_energy)

    async def get_chunk(self) -> np.ndarray:
        """Await and return the next audio chunk from the first consumer queue... wait, removed main eq.
        This method should not be used anymore since consumers add their own queues."""
        raise NotImplementedError("Use add_consumer() instead of get_chunk()")

    def add_consumer(self) -> asyncio.Queue:
        """
        Register a new consumer and return a dedicated queue that will
        receive a copy of every audio chunk independently.
        Echo-suppressed chunks are NOT delivered to this queue.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._consumer_queues.append(q)
        return q

    def add_raw_consumer(self) -> asyncio.Queue:
        """V2: Register a raw consumer that receives ALL audio, including
        chunks that would be dropped by echo suppression.

        Used by InterruptionDetector which has its own echo discrimination
        (echo EMA + energy gate) and needs every chunk to work correctly.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._raw_consumer_queues.append(q)
        return q

    def inject_chunk(self, audio_bytes: bytes) -> None:
        """
        Inject external audio (e.g. from WebSocket) into all consumer queues.
        Converts raw 16-bit PCM bytes to float32 normalised numpy and pushes.
        """
        chunk_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
        chunk_f32 = chunk_i16.astype(np.float32) / 32768.0

        loop = self._loop
        if loop is None:
            return

        for q in self._consumer_queues:
            def _enqueue(queue: asyncio.Queue = q) -> None:
                try:
                    queue.put_nowait(chunk_f32)
                except asyncio.QueueFull:
                    pass  # drop silently
            loop.call_soon_threadsafe(_enqueue)

    def _fast_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """V2: Fast normalized cross-correlation between two same-length arrays."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-10 or b_norm < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    # ── V4: WebSocket audio source ─────────────────────────────────────────

    def set_audio_source(self, queue: asyncio.Queue) -> None:
        """V4: Route audio from a WebSocket queue instead of sounddevice.

        When set, run() will skip sounddevice entirely and read PCM bytes
        from this queue. When not set (default), sounddevice is used.

        Called by PipelineInstance to wire the AudioBridge mic queue.

        Args:
            queue: asyncio.Queue providing raw 16-bit PCM bytes from client.
        """
        self._audio_source_queue = queue
        self._logger.info("V4: Audio source set to WebSocket queue")

    async def _ws_audio_loop(self) -> None:
        """V4: Read audio from WebSocket queue and distribute to consumers.

        Replaces _audio_callback for WebSocket mode. Same fan-out logic:
        raw consumers first, then echo filtering, then filtered consumers.
        """
        while self._running:
            try:
                raw = await self._audio_source_queue.get()
            except asyncio.CancelledError:
                break

            # Convert raw 16-bit PCM bytes to float32 normalised
            chunk_f32: np.ndarray = np.frombuffer(
                raw, dtype=np.int16
            ).astype(np.float32) / 32768.0

            # Update recent audio buffer (for last_200ms())
            self._recent_audio.extend(chunk_f32.tolist())

            # Fan-out to raw consumers FIRST (before echo filtering)
            for q in self._raw_consumer_queues:
                try:
                    q.put_nowait(chunk_f32)
                except asyncio.QueueFull:
                    pass  # drop silently — raw consumer overwhelmed

            # Echo filtering (same logic as _audio_callback)
            if self._filter_active:
                blocked = False
                if self._aria_filter is not None:
                    try:
                        if self._aria_filter.is_aria_echo(chunk_f32):
                            self._logger.debug("V4/WS: AriaVoiceFilter BLOCKED chunk")
                            blocked = True
                    except Exception:
                        pass
                if not blocked and self._gate0 is not None:
                    try:
                        if self._gate0.is_echo(chunk_f32):
                            self._logger.debug("V4/WS: Gate0 BLOCKED echo chunk")
                            blocked = True
                    except Exception:
                        pass
                if blocked:
                    continue  # Skip filtered consumers, raw already got it

            # Fan-out to filtered consumer queues
            for q in self._consumer_queues:
                try:
                    q.put_nowait(chunk_f32)
                except asyncio.QueueFull:
                    self._logger.warning("V4/WS: Consumer audio queue full — dropping chunk")

    # ── V3: Filter control + recent audio ─────────────────────────────────

    def set_v3_filters(self, aria_filter, gate0) -> None:
        """V3: Inject AriaVoiceFilter and Gate0EchoCheck after construction.

        Called once during main.py initialization, after the filters are
        created and pre-warmed.
        """
        self._aria_filter = aria_filter
        self._gate0 = gate0
        self._logger.info("V3 echo filters attached (AriaVoiceFilter + Gate0)")

    def set_filter_active(self, active: bool) -> None:
        """V3: Toggle AriaVoiceFilter + Gate0 on/off.

        ON:  When TTS is actively playing *real speech* (not clips, not paused)
        OFF: During clips, LISTENING, PAUSED, silence

        Called by:
          - AudioPlayer._play_tts_chunk() → set_filter_active(True)
          - AudioPlayer.pause()           → set_filter_active(False)
          - AudioPlayer.flush()           → set_filter_active(False)
          - AudioPlayer._playback_loop()  → set_filter_active(False) when done
        """
        if self._filter_active != active:
            self._filter_active = active
            self._logger.debug("V3 filter_active = %s", active)

    def check_echo(self, chunk: np.ndarray) -> bool:
        """V3: Query whether a chunk is AI echo — for raw-consumer callers.

        InterruptionDetector uses add_raw_consumer() so echo filtering does
        not happen upstream. This method lets it query the same filters
        directly without routing through the filtered consumer path.

        Returns True only when echo suppression is active (during real TTS
        playback) AND at least one filter identifies the chunk as Aria's
        voice / TTS spectral echo. Safe to call frequently — exceptions are
        caught and treated as non-echo.

        Returns:
            True  → chunk is AI echo, caller should discard it.
            False → chunk is not echo (or filter inactive), caller handles it.
        """
        if not self._filter_active:
            return False
        try:
            if self._aria_filter is not None and self._aria_filter.is_aria_echo(chunk):
                return True
        except Exception:
            pass
        try:
            if self._gate0 is not None and self._gate0.is_echo(chunk):
                return True
        except Exception:
            pass
        return False

    def last_200ms(self) -> np.ndarray:
        """V3: Return the most recent ~200ms of mic audio.

        Used by InterruptionDetector for short-burst energy analysis,
        and by SpeakingMonitor for VAD-end checks.
        """
        if not self._recent_audio:
            return np.array([], dtype=np.float32)
        return np.array(self._recent_audio, dtype=np.float32)

    async def stop(self) -> None:
        """Stop and close the microphone stream."""
        self._running = False
        stream = self._stream
        if stream is not None:
            stream.stop()
            stream.close()
        self._logger.info("Mic stream stopped")

    @property
    def is_running(self) -> bool:
        """Whether the mic stream is currently active."""
        return self._running