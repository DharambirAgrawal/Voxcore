"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                       VOXCORE — output/audio_player.py                          ║
║    ASYNC AUDIO PLAYER — STREAMS WAV CHUNKS TO SPEAKER (INTERRUPTIBLE)          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    An asyncio-driven audio player using sounddevice output stream. Consumes
    WAV audio chunks from TTSClient as they arrive and plays them through the
    system speakers.

    CRITICAL FEATURE: Subscribes to INTERRUPT_DETECTED events. On interrupt,
    immediately empties the play queue and cancels the current audio. Does NOT
    cut off mid-word — waits for the current 20ms audio frame to finish for
    a clean cutoff.

    Also handles backchannel clip playback via BACKCHANNEL_FIRE events.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import io                               # BytesIO for WAV parsing

import numpy as np                      # Audio array manipulation
import sounddevice as sd                # Audio output playback
from scipy.io import wavfile            # WAV file reading

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: AudioPlayer
──────────────────────────────────────────────────────────────────────────────────
    Interruptible async audio player for TTS output and backchannel clips.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — For reading current TurnState
            - event_bus: EventBus — Subscribe to TTS_CHUNK_READY, INTERRUPT_DETECTED, BACKCHANNEL_FIRE
            - config: dict — The "audio" section from config.yaml:
                - output_sample_rate: int (24000 — Kokoro-ONNX output rate)
        
        INITIALIZES:
            self.session: Session                = session
            self.event_bus: EventBus             = event_bus
            self.output_sample_rate: int         = config.get("output_sample_rate", 24000)
            
            # Audio queue — holds WAV chunks waiting to be played
            self._play_queue: asyncio.Queue      = asyncio.Queue(maxsize=50)
            
            # Event subscriptions
            self._tts_chunk_queue: asyncio.Queue = event_bus.subscribe(EventType.TTS_CHUNK_READY)
            self._interrupt_queue: asyncio.Queue = event_bus.subscribe(EventType.INTERRUPT_DETECTED)
            self._backchannel_queue: asyncio.Queue = event_bus.subscribe(EventType.BACKCHANNEL_FIRE)
            
            # State
            self._is_playing: bool               = False
            self._is_interrupted: bool           = False
            self._current_stream: Optional[sd.OutputStream] = None
            self._audio_accumulator: bytearray   = bytearray()  # Accumulates TTS chunks into complete WAV
            
            self._logger: logging.Logger         = logging.getLogger("AudioPlayer")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Runs three concurrent tasks via asyncio.gather():
            1. self._process_tts_chunks()      — Collects TTS chunks into playable audio
            2. self._process_interrupts()       — Handles interrupt events
            3. self._process_backchannels()     — Handles backchannel clip playback
            4. self._playback_loop()            — Main playback loop from play queue

    async def _process_tts_chunks(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Infinite loop:
            1. event = await self._tts_chunk_queue.get()
            2. If self._is_interrupted: discard chunk, continue
            3. audio_chunk = event.data["audio"]
            4. sentence_index = event.data["sentence_index"]
            5. self._audio_accumulator.extend(audio_chunk)
            6. Check for TTS_SENTENCE_DONE event (subscribe separately or check flag):
               When a sentence's audio is complete:
               a. Parse accumulated bytes as WAV:
                  sr, audio_array = wavfile.read(io.BytesIO(bytes(self._audio_accumulator)))
               b. Put audio_array into self._play_queue
               c. Reset self._audio_accumulator = bytearray()

    async def _playback_loop(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Infinite loop:
            1. audio_array = await self._play_queue.get()
            2. If self._is_interrupted:
               self._play_queue = asyncio.Queue(maxsize=50)  # Clear queue
               self._is_interrupted = False
               Continue
            3. self._is_playing = True
            4. await self._play_audio(audio_array)
            5. self._is_playing = False
            6. If self._play_queue.empty():
               Publish PLAYBACK_DONE event
               Log: "Playback complete"

    async def _play_audio(self, audio: np.ndarray) -> None
        INPUTS:
            - audio: np.ndarray — Audio data to play (int16 or float32)
        OUTPUT: None (blocks until audio finishes or is interrupted)
        WHAT IT DOES:
            1. Convert to float32 if needed:
               if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
            2. Determine frame size: frame_ms = 20  (20ms frames for clean interrupt cutoff)
               frame_samples = self.output_sample_rate * frame_ms // 1000
            3. For each frame in audio (chunks of frame_samples):
               a. If self._is_interrupted: break immediately
               b. Write frame to sounddevice output stream
               c. await asyncio.sleep(0)  # Yield to event loop for interrupt detection
            4. Alternative simpler approach:
               sd.play(audio, samplerate=self.output_sample_rate, blocking=False)
               While sd.get_stream().active:
                   If self._is_interrupted: sd.stop(); break
                   await asyncio.sleep(0.02)  # Check every 20ms
        
        INTERRUPT BEHAVIOR:
            On interrupt, playback stops at the next 20ms frame boundary.
            This prevents jarring mid-sample cutoff sounds. The silence
            gap is imperceptible (< 20ms).

    async def _process_interrupts(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Infinite loop:
            1. event = await self._interrupt_queue.get()
            2. Log: "INTERRUPT — stopping playback"
            3. self._is_interrupted = True
            4. Clear play queue: while not self._play_queue.empty(): self._play_queue.get_nowait()
            5. Clear audio accumulator: self._audio_accumulator = bytearray()
            6. Stop current sounddevice playback: sd.stop()
            7. After brief delay: self._is_interrupted = False

    async def _process_backchannels(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Infinite loop:
            1. event = await self._backchannel_queue.get()
            2. If self._is_playing: skip (don't play backchannel during TTS)
            3. If session.state == TurnState.SPEAKING: skip
            4. clip_path = event.data["clip_path"]
            5. sr, audio = wavfile.read(clip_path)
            6. Play immediately at low volume (mix with main audio if needed):
               sd.play(audio * 0.8, samplerate=sr, blocking=False)
            7. Log: "Backchannel: played '{event.data['clip_name']}'"

    async def stop(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. sd.stop()
            2. self._is_playing = False
            3. Clear play queue

    @property
    def is_playing(self) -> bool
        OUTPUT: bool — Whether audio is currently being played

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - AudioPlayer   (class)
═══════════════════════════════════════════════════════════════════════════════════

AUDIO FORMAT NOTES:
    - Kokoro-ONNX outputs 24kHz WAV audio
    - Backchannel clips are also 24kHz WAV (generated by Kokoro)
    - sounddevice handles sample rate conversion automatically if needed
    - Audio frames are 20ms (480 samples at 24kHz) for interrupt granularity

LATENCY:
    - sounddevice output latency: ~10ms
    - This is the final step — after this, the user hears the AI's voice
"""


import asyncio
import collections
import logging
import io
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType

# V2: Reference buffer duration for echo suppression (seconds)
REFERENCE_BUFFER_DURATION_S = 2.0


class AudioPlayer:
    """Interruptible async audio player for TTS output and backchannel clips.

    V2 additions:
    - PAUSE_MARKER event handling: injects 300ms silence into playback queue
    - Reference signal buffer: maintains rolling 2s buffer of played audio
      for correlation-based echo suppression in MicStream

    V3 additions:
    - Playback pause/resume: on PLAYBACK_PAUSE, immediately stops writing
      audio frames to the speaker while keeping position. On PLAYBACK_RESUME,
      continues from the exact frame where it paused.
    - Volume ducking kept as fallback but primary flow is pause/resume.
    - play_clip() for one-shot clip playback without filter feeds
    - play_clip_looping() for looping a clip until a stop event
    - pause()/resume()/flush() as proper async methods for InterruptRouter
    - Filter feed control: only feeds AriaVoiceFilter/Gate0 during TTS chunks
    """

    def __init__(self, session: Session, event_bus: EventBus, config: dict) -> None:
        self.session: Session = session
        self.event_bus: EventBus = event_bus
        self.output_sample_rate: int = config.get("output_sample_rate", 24000)

        # Audio queue — holds parsed audio arrays waiting to be played
        self._play_queue: asyncio.Queue = asyncio.Queue(maxsize=50)

        # Event subscriptions
        self._tts_chunk_queue: asyncio.Queue = event_bus.subscribe(EventType.TTS_CHUNK_READY)
        self._interrupt_queue: asyncio.Queue = event_bus.subscribe(EventType.INTERRUPT_DETECTED)
        self._backchannel_queue: asyncio.Queue = event_bus.subscribe(EventType.BACKCHANNEL_FIRE)
        self._tts_all_done_queue: asyncio.Queue = event_bus.subscribe(EventType.TTS_ALL_DONE)
        self._volume_duck_queue: asyncio.Queue = event_bus.subscribe(EventType.VOLUME_DUCK)
        self._volume_restore_queue: asyncio.Queue = event_bus.subscribe(EventType.VOLUME_RESTORE)
        self._pause_queue: asyncio.Queue = event_bus.subscribe(EventType.PLAYBACK_PAUSE)
        self._resume_queue: asyncio.Queue = event_bus.subscribe(EventType.PLAYBACK_RESUME)

        # State
        self._is_playing: bool = False
        self._is_interrupted: bool = False
        self._is_paused: bool = False           # V3: true pause — freeze in place
        self._pause_event: asyncio.Event = asyncio.Event()  # set = not paused
        self._pause_event.set()  # start unpaused
        self._current_stream: Optional[sd.OutputStream] = None
        self._audio_accumulator: bytearray = bytearray()
        self._tts_stream_done: bool = False

        # V3: Volume ducking — applied per-frame during playback
        self._volume: float = 1.0
        self._target_volume: float = 1.0
        self._volume_ramp_speed: float = 0.05

        # V2: Reference signal buffer for echo suppression
        # Stores recent playback audio so mic_stream can cross-correlate
        ref_buffer_samples = int(self.output_sample_rate * REFERENCE_BUFFER_DURATION_S)
        self._reference_buffer: collections.deque = collections.deque(maxlen=ref_buffer_samples)

        # V2: Callback for mic_stream to receive playback reference signal
        self._reference_callback = None

        # V3: Direct references for filter feed control
        self._mic_stream = None     # Set via set_mic_stream()
        self._aria_filter = None    # AriaVoiceFilter instance
        self._gate0 = None          # Gate0EchoCheck instance

        self._logger: logging.Logger = logging.getLogger("AudioPlayer")

    def set_reference_callback(self, callback) -> None:
        """V2: Register a callback that receives played audio frames for echo suppression.

        The callback receives (frame: np.ndarray) of float32 audio data.
        MicStream uses this to build its reference buffer for cross-correlation.
        """
        self._reference_callback = callback

    def get_reference_buffer(self) -> np.ndarray:
        """V2: Return the current reference signal buffer as a numpy array.

        Used by MicStream for cross-correlation echo suppression.
        """
        if not self._reference_buffer:
            return np.array([], dtype=np.float32)
        return np.array(self._reference_buffer, dtype=np.float32)

    async def run(self) -> None:
        """Run all player loops concurrently."""
        # V2: Subscribe to PAUSE_MARKER events
        async def _on_pause_marker(event) -> None:
            duration_ms = event.data.get("duration_ms", 300)
            await self._inject_silence(duration_ms)

        self.event_bus.subscribe(EventType.PAUSE_MARKER, _on_pause_marker)

        await asyncio.gather(
            self._process_tts_chunks(),
            self._process_interrupts(),
            self._process_backchannels(),
            self._playback_loop(),
            self._watch_tts_all_done(),
            self._process_volume_duck(),
            self._process_volume_restore(),
            self._process_pause(),
            self._process_resume(),
        )

    async def _watch_tts_all_done(self) -> None:
        """Set _tts_stream_done when TTSClient signals all sentences synthesized.

        If the playback loop has already drained the queue (it's blocked on
        ``_play_queue.get()``), we fire PLAYBACK_DONE directly here.
        """
        while True:
            await self._tts_all_done_queue.get()
            self._tts_stream_done = True
            self._logger.debug("Received TTS_ALL_DONE — will fire PLAYBACK_DONE when queue drains")

            # Edge case: playback already finished but was waiting for this flag
            if self._play_queue.empty() and not self._is_playing:
                await self.event_bus.publish(
                    EventType.PLAYBACK_DONE, {}, source="AudioPlayer"
                )
                self._tts_stream_done = False
                self._logger.debug("Playback complete (TTS_ALL_DONE arrived after queue drained)")

    async def _process_tts_chunks(self) -> None:
        """Collect TTS audio chunks; when a sentence is complete, enqueue for playback."""
        while True:
            event = await self._tts_chunk_queue.get()

            if self._is_interrupted:
                continue

            audio_chunk: bytes = event.data["audio"]
            is_sentence_done: bool = event.data.get("sentence_done", False)

            # First chunk of a new response — reset TTS-done flag
            if not self._audio_accumulator and not self._tts_stream_done:
                pass  # already False
            if len(self._audio_accumulator) == 0:
                self._tts_stream_done = False

            self._audio_accumulator.extend(audio_chunk)

            if is_sentence_done:
                try:
                    sr, audio_array = wavfile.read(
                        io.BytesIO(bytes(self._audio_accumulator))
                    )
                    await self._play_queue.put(audio_array)
                except Exception as e:
                    self._logger.error("Failed to parse accumulated WAV: %s", e)
                finally:
                    self._audio_accumulator = bytearray()

    async def _playback_loop(self) -> None:
        """Main loop — pull audio arrays from queue and play them.

        PLAYBACK_DONE is only published when both conditions are met:
          1. The play queue is empty (all enqueued audio has been played)
          2. _tts_stream_done is True (TTSClient has synthesized ALL sentences)
        This prevents premature SPEAKING→LISTENING transitions between sentences. 
        """
        while True:
            audio_array = await self._play_queue.get()

            if self._is_interrupted:
                # Drain the queue
                while not self._play_queue.empty():
                    try:
                        self._play_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self._is_interrupted = False
                continue

            self._is_playing = True
            if self._mic_stream is not None:
                self._mic_stream.set_filter_active(True)
            try:
                await self._play_audio(audio_array)
            finally:
                if self._mic_stream is not None:
                    self._mic_stream.set_filter_active(False)
            self._is_playing = False

            # Only fire PLAYBACK_DONE when TTS has finished synthesizing
            # all sentences AND the play queue has drained.
            if self._play_queue.empty() and self._tts_stream_done:
                await self.event_bus.publish(
                    EventType.PLAYBACK_DONE, {}, source="AudioPlayer"
                )
                self._tts_stream_done = False  # Reset for next response
                self._logger.debug("Playback complete")

    async def _play_audio(self, audio: np.ndarray) -> None:
        """Play audio array through speakers with 20ms frame interrupt granularity.

        V3: Supports true pause/resume. When _is_paused is True, the playback
        loop waits on _pause_event instead of writing frames. When resumed,
        it continues from the exact frame offset where it paused.
        Also applies volume ducking per-frame for smooth transitions.
        """
        # Normalize to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure mono is shaped correctly
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)

        frame_ms = 20
        frame_samples = self.output_sample_rate * frame_ms // 1000
        total_samples = audio.shape[0]

        try:
            self._current_stream = sd.OutputStream(
                samplerate=self.output_sample_rate,
                channels=audio.shape[1],
                dtype="float32",
            )
            self._current_stream.start()

            offset = 0
            while offset < total_samples:
                if self._is_interrupted:
                    break

                # V3: True pause — wait here until resumed or interrupted
                if self._is_paused:
                    # Stop the stream while paused to avoid underflow clicks
                    if self._current_stream is not None:
                        try:
                            self._current_stream.stop()
                        except Exception:
                            pass
                    self._logger.debug("Playback PAUSED at frame %d/%d", offset, total_samples)
                    # Wait for resume or interrupt
                    while self._is_paused and not self._is_interrupted:
                        await asyncio.sleep(0.02)
                    if self._is_interrupted:
                        break
                    # Restart the stream from where we left off
                    self._logger.debug("Playback RESUMED at frame %d/%d", offset, total_samples)
                    try:
                        self._current_stream = sd.OutputStream(
                            samplerate=self.output_sample_rate,
                            channels=audio.shape[1],
                            dtype="float32",
                        )
                        self._current_stream.start()
                    except Exception as e:
                        self._logger.error("Failed to restart stream after resume: %s", e)
                        break

                end = min(offset + frame_samples, total_samples)
                frame = audio[offset:end].copy()

                # V3: Smooth volume ramping per-frame
                if abs(self._volume - self._target_volume) > 0.001:
                    if self._volume < self._target_volume:
                        self._volume = min(
                            self._volume + self._volume_ramp_speed,
                            self._target_volume,
                        )
                    else:
                        self._volume = max(
                            self._volume - self._volume_ramp_speed,
                            self._target_volume,
                        )

                # Apply volume multiplier
                if self._volume < 0.99:
                    frame = frame * self._volume

                self._current_stream.write(frame)

                # V2: Feed reference buffer for echo suppression
                mono_frame = frame[:, 0] if frame.ndim > 1 else frame
                self._reference_buffer.extend(mono_frame.tolist())
                if self._reference_callback is not None:
                    try:
                        self._reference_callback(mono_frame)
                    except Exception:
                        pass

                # V3: Feed AriaVoiceFilter and Gate0 for multi-layer echo suppression
                if hasattr(self, '_aria_filter') and self._aria_filter is not None:
                    try:
                        self._aria_filter.feed_aria_audio(mono_frame)
                    except Exception:
                        pass
                if hasattr(self, '_gate0') and self._gate0 is not None:
                    try:
                        self._gate0.feed_tts_spectrum(mono_frame)
                    except Exception:
                        pass

                offset = end

                # Yield to event loop so interrupts/pauses can be detected
                await asyncio.sleep(0)

            stream = self._current_stream
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception as e:
            self._logger.error("Playback error: %s", e)
        finally:
            self._current_stream = None

    async def _inject_silence(self, duration_ms: int) -> None:
        """V2: Inject silence into the playback queue for pause markers.

        Creates a silent audio array of the specified duration and enqueues it
        for the playback loop. This creates natural pauses when the LLM emits [...].
        """
        num_samples = int(self.output_sample_rate * duration_ms / 1000)
        silence = np.zeros(num_samples, dtype=np.float32)
        try:
            await self._play_queue.put(silence)
            self._logger.debug("Injected %dms silence", duration_ms)
        except asyncio.QueueFull:
            self._logger.warning("Play queue full — could not inject silence")

    async def _process_interrupts(self) -> None:
        """Listen for interrupt events and immediately stop playback."""
        while True:
            await self._interrupt_queue.get()
            self._logger.info("INTERRUPT — stopping playback")

            self._is_interrupted = True
            self._is_paused = False  # clear pause state

            # V3: Reset volume to full on interrupt
            self._volume = 1.0
            self._target_volume = 1.0

            # Clear play queue
            while not self._play_queue.empty():
                try:
                    self._play_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Clear accumulator
            self._audio_accumulator = bytearray()

            # Stop sounddevice immediately
            sd.stop()

            # Close current stream if active
            if self._current_stream is not None:
                try:
                    self._current_stream.stop()
                    self._current_stream.close()
                except Exception:
                    pass
                self._current_stream = None

            # Brief delay then re-enable
            await asyncio.sleep(0.05)
            self._is_interrupted = False

    async def _process_volume_duck(self) -> None:
        """Listen for VOLUME_DUCK events and reduce playback volume.

        When the user starts speaking (Gate 1 passes), the SpeakingMonitor
        fires VOLUME_DUCK. We immediately set the target volume low so the
        user feels heard. The per-frame ramp in _play_audio handles the
        smooth transition to avoid audio clicks.
        """
        while True:
            event = await self._volume_duck_queue.get()
            level = event.data.get("level", 0.15) if event.data else 0.15
            self._target_volume = max(0.0, min(1.0, level))
            # Use faster ramp for ducking (immediate feedback)
            self._volume_ramp_speed = 0.15
            self._logger.info(
                "Volume DUCK → %.0f%% (reason: %s)",
                self._target_volume * 100,
                event.data.get("reason", "unknown") if event.data else "unknown",
            )

    async def _process_volume_restore(self) -> None:
        """Listen for VOLUME_RESTORE events and ramp volume back to 100%.

        Fires after classification determines IGNORE or INJECT — the user
        wasn't actually interrupting, so restore normal volume.
        """
        while True:
            event = await self._volume_restore_queue.get()
            self._target_volume = 1.0
            # Use slower ramp for restore (smooth fade-in)
            self._volume_ramp_speed = 0.05
            self._logger.info(
                "Volume RESTORE → 100%% (reason: %s)",
                event.data.get("reason", "unknown") if event.data else "unknown",
            )

    async def _process_pause(self) -> None:
        """Listen for PLAYBACK_PAUSE events — freeze playback immediately.

        PersonaPlex-inspired: when user starts speaking, we don't just duck
        the volume — we completely stop outputting audio within 20ms. This
        gives the mic a clean signal (no echo) to transcribe the user's speech.
        The playback position is preserved so we can resume later.
        """
        while True:
            event = await self._pause_queue.get()
            if not self._is_paused and self._is_playing:
                self._is_paused = True
                self._logger.info(
                    "Playback PAUSED (reason: %s)",
                    event.data.get("reason", "user_speaking") if event.data else "user_speaking",
                )

    async def _process_resume(self) -> None:
        """Listen for PLAYBACK_RESUME events — continue from where we paused.

        Fires when classification returns IGNORE or INJECT — the user was
        just reacting, not interrupting. AI picks up exactly where it left off.
        """
        while True:
            event = await self._resume_queue.get()
            if self._is_paused:
                self._is_paused = False
                self._logger.info(
                    "Playback RESUMED (reason: %s)",
                    event.data.get("reason", "classification_done") if event.data else "classification_done",
                )

    async def _process_backchannels(self) -> None:
        """Play backchannel audio clips (e.g. 'mhm', 'yeah') when not speaking."""
        while True:
            event = await self._backchannel_queue.get()

            if self._is_playing:
                continue
            if self.session.state == TurnState.SPEAKING:
                continue

            clip_path: str = event.data["clip_path"]
            clip_name: str = event.data.get("clip_name", clip_path)

            try:
                sr, audio = wavfile.read(clip_path)
                audio_float = audio.astype(np.float32)
                if audio.dtype == np.int16:
                    audio_float = audio_float / 32768.0
                sd.play(audio_float * 0.8, samplerate=sr, blocking=False)
                self._logger.debug("Backchannel: played '%s'", clip_name)
            except Exception as e:
                self._logger.error("Backchannel playback failed: %s", e)

    async def stop(self) -> None:
        """Stop all playback and clear the queue."""
        sd.stop()
        self._is_playing = False
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently being played."""
        return self._is_playing

    # ── V3: Direct API for InterruptRouter ────────────────────────────────

    def set_v3_refs(self, mic_stream, aria_filter=None, gate0=None) -> None:
        """V3: Inject direct references for filter feed control.

        Called once during main.py initialization.
        """
        self._mic_stream = mic_stream
        self._aria_filter = aria_filter
        self._gate0 = gate0
        self._logger.info("V3 refs attached (mic_stream, aria_filter, gate0)")

    async def pause(self) -> None:
        """V3: Freeze TTS playback in RAM. Called by InterruptRouter.

        - Immediately stops audio output within 20ms
        - Preserves playback position for resume()
        - Toggles filter OFF (no echo suppression needed when silent)
        """
        if not self._is_paused and self._is_playing:
            self._is_paused = True
            if self._mic_stream is not None:
                self._mic_stream.set_filter_active(False)
            self._logger.info("V3: Playback PAUSED (filter OFF)")

    async def resume(self) -> None:
        """V3: Resume TTS from exact frame where it paused. Called by InterruptRouter.

        - Restarts audio output from preserved position
        - Toggles filter ON (echo suppression needed again)
        """
        if self._is_paused:
            self._is_paused = False
            if self._mic_stream is not None:
                self._mic_stream.set_filter_active(True)
            self._logger.info("V3: Playback RESUMED (filter ON)")

    async def flush(self) -> None:
        """V3: Discard ALL buffered TTS. Called by InterruptRouter for STOP/CORRECTION/etc.

        - Clears play queue
        - Clears audio accumulator
        - Stops current audio stream
        - Toggles filter OFF
        """
        self._is_interrupted = True
        self._is_paused = False
        self._volume = 1.0
        self._target_volume = 1.0

        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._audio_accumulator = bytearray()
        sd.stop()

        if self._current_stream is not None:
            try:
                self._current_stream.stop()
                self._current_stream.close()
            except Exception:
                pass
            self._current_stream = None

        if self._mic_stream is not None:
            self._mic_stream.set_filter_active(False)

        await asyncio.sleep(0.05)
        self._is_interrupted = False
        self._logger.info("V3: Playback FLUSHED (filter OFF)")

    async def play_clip(self, clip_path: str) -> None:
        """V3: Play a clip once without feeding echo filters.

        Used for bridge clips (sure.wav, got_it.wav) and pre-pause clips.
        Does NOT feed AriaVoiceFilter or Gate0 — these are not Aria's voice.
        """
        try:
            sr, audio = wavfile.read(clip_path)
            audio_float = audio.astype(np.float32)
            if audio.dtype == np.int16:
                audio_float = audio_float / 32768.0
            sd.play(audio_float * 0.9, samplerate=sr, blocking=False)
            self._logger.debug("V3: played clip '%s'", clip_path)
        except Exception as e:
            self._logger.error("V3: clip playback failed: %s", e)

    async def play_clip_looping(self, clip_path: str, stop_event: asyncio.Event) -> None:
        """V3: Loop a clip until stop_event is set (for pre-pause during PENDING).

        Plays the clip, waits for its duration, checks stop_event, repeats.
        Stops between loops (not mid-playback) when event is set.
        """
        try:
            sr, audio = wavfile.read(clip_path)
            audio_float = audio.astype(np.float32)
            if audio.dtype == np.int16:
                audio_float = audio_float / 32768.0

            clip_duration = len(audio_float) / sr
            max_loops = 4  # ~2s max for a 0.5s clip

            for _ in range(max_loops):
                if stop_event.is_set():
                    break
                sd.play(audio_float * 0.7, samplerate=sr, blocking=False)
                # Wait for clip to finish, checking stop_event periodically
                elapsed = 0.0
                while elapsed < clip_duration:
                    if stop_event.is_set():
                        sd.stop()
                        break
                    await asyncio.sleep(0.05)
                    elapsed += 0.05

            self._logger.debug("V3: clip loop done '%s'", clip_path)
        except Exception as e:
            self._logger.error("V3: clip loop failed: %s", e)