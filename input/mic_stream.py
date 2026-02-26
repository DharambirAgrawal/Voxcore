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
from input.echo_cancel import StreamingAEC, RefRingBuffer

# V2: Default correlation threshold for echo suppression (legacy fallback)
DEFAULT_ECHO_CORRELATION_THRESHOLD = 0.7


class MicStream:
    """Continuous microphone audio capture producing fixed-size chunks.

    V2 additions:
    - Cross-correlation echo suppression: maintains a reference buffer of
      recently played audio. When an incoming mic chunk correlates strongly
      (above threshold) with the reference, it's treated as echo and dropped.
    """

    def __init__(self, event_bus: EventBus, config: dict) -> None:
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

        # V2: Echo suppression via cross-correlation (legacy fallback)
        self._echo_threshold: float = config.get(
            "echo_correlation_threshold", DEFAULT_ECHO_CORRELATION_THRESHOLD
        )
        # Rolling reference buffer — stores ~0.5s of playback audio for correlation
        ref_samples = int(self.sample_rate * 0.5)
        self._reference_buffer: collections.deque = collections.deque(maxlen=ref_samples)
        self._echo_suppression_enabled: bool = config.get("echo_suppression", True)

        # V3: WebRTC-style Acoustic Echo Cancellation (AEC)
        # When enabled, replaces the legacy cross-correlation approach with a
        # proper adaptive filter that learns the speaker→mic transfer function.
        self._aec_enabled: bool = config.get("echo_cancellation", True)
        if self._aec_enabled:
            aec_blocks = config.get("aec_filter_blocks", 12)  # 12×30ms = 360ms tail
            aec_mu = config.get("aec_step_size", 0.3)
            self._aec = StreamingAEC(
                frame_size=self.chunk_samples,
                num_blocks=aec_blocks,
                mu=aec_mu,
                delta=1e-2,     # high regularisation prevents divergence
                leak=0.9995,    # slow weight decay keeps filter bounded
            )
            # SPSC ring buffer for reference signal (24kHz→16kHz resampled)
            self._aec_ref_buf = RefRingBuffer(capacity=self.sample_rate * 2)  # 2s
            self._logger.info(
                "AEC enabled (blocks=%d, tail=%dms, mu=%.2f)",
                aec_blocks,
                aec_blocks * self.chunk_ms,
                aec_mu,
            )
        else:
            self._aec = None
            self._aec_ref_buf = None

    async def run(self) -> None:
        """Start mic capture and run forever until cancelled."""
        self._loop = asyncio.get_running_loop()

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

        # ── V3: AEC processing (when enabled) ────────────────────────────
        # Apply adaptive echo cancellation BEFORE distributing to consumers.
        # This gives ALL consumers clean audio with echo removed.
        if self._aec is not None and self._aec_ref_buf is not None:
            ref_frame = self._aec_ref_buf.read(len(chunk_f32))
            clean = self._aec.process(chunk_f32, ref_frame)

            # Distribute AEC-cleaned audio to ALL consumers
            for q in self._raw_consumer_queues:
                def _enqueue_raw(queue: asyncio.Queue = q, data: np.ndarray = clean) -> None:
                    try:
                        queue.put_nowait(data)
                    except asyncio.QueueFull:
                        pass
                loop.call_soon_threadsafe(_enqueue_raw)

            for q in self._consumer_queues:
                def _enqueue_consumer(queue: asyncio.Queue = q, data: np.ndarray = clean) -> None:
                    try:
                        queue.put_nowait(data)
                    except asyncio.QueueFull:
                        pass
                loop.call_soon_threadsafe(_enqueue_consumer)
            return

        # ── Legacy path: no AEC ───────────────────────────────────────────
        # Raw consumers get ALL audio (no echo suppression).
        for q in self._raw_consumer_queues:

            def _enqueue_raw(queue: asyncio.Queue = q) -> None:
                try:
                    queue.put_nowait(chunk_f32)
                except asyncio.QueueFull:
                    pass  # drop silently — raw consumers can tolerate loss

            loop.call_soon_threadsafe(_enqueue_raw)

        # V2: Echo suppression — check if mic chunk correlates with recent playback
        if self._echo_suppression_enabled and len(self._reference_buffer) >= len(chunk_f32):
            ref_arr = np.array(list(self._reference_buffer)[-len(chunk_f32):], dtype=np.float32)
            correlation = self._fast_correlation(chunk_f32, ref_arr)
            if correlation > self._echo_threshold:
                # This chunk is likely echo — drop it for filtered consumers only
                return

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

    def set_playback_reference(self, frame: np.ndarray, src_sample_rate: int = 24000) -> None:
        """Feed recently played audio into the echo cancellation pipeline.

        V2: Cross-correlation reference buffer (legacy).
        V3: Resamples to mic sample rate and feeds the adaptive AEC filter.

        Args:
            frame: float32 mono audio frame from AudioPlayer's output.
            src_sample_rate: Sample rate of the frame (default 24 kHz for Kokoro TTS).
        """
        # Legacy reference buffer (always updated for backwards compatibility)
        self._reference_buffer.extend(frame.tolist())

        # V3: AEC reference — resample from output rate (24 kHz) to mic rate (16 kHz)
        if self._aec_ref_buf is not None:
            if src_sample_rate != self.sample_rate:
                n_out = int(len(frame) * self.sample_rate / src_sample_rate)
                frame_resampled = np.interp(
                    np.linspace(0, len(frame) - 1, n_out),
                    np.arange(len(frame)),
                    frame,
                ).astype(np.float32)
            else:
                frame_resampled = frame
            self._aec_ref_buf.write(frame_resampled)

    @staticmethod
    def _fast_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """V2: Fast normalized cross-correlation between two same-length arrays.

        Returns a value in [-1, 1]. Values > 0.7 indicate the mic signal
        closely matches the playback signal (likely echo).
        """
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-10 or b_norm < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

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