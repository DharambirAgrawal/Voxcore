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
import logging
import io
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType


class AudioPlayer:
    """Interruptible async audio player for TTS output and backchannel clips."""

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

        # State
        self._is_playing: bool = False
        self._is_interrupted: bool = False
        self._current_stream: Optional[sd.OutputStream] = None
        self._audio_accumulator: bytearray = bytearray()
        self._tts_stream_done: bool = False  # True once TTSClient has synthesized all sentences

        self._logger: logging.Logger = logging.getLogger("AudioPlayer")

    async def run(self) -> None:
        """Run all player loops concurrently."""
        await asyncio.gather(
            self._process_tts_chunks(),
            self._process_interrupts(),
            self._process_backchannels(),
            self._playback_loop(),
            self._watch_tts_all_done(),
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
            await self._play_audio(audio_array)
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
        """Play audio array through speakers with 20ms frame interrupt granularity."""
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

                end = min(offset + frame_samples, total_samples)
                frame = audio[offset:end]
                self._current_stream.write(frame)
                offset = end

                # Yield to event loop so interrupts can be detected
                await asyncio.sleep(0)

            stream = self._current_stream
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception as e:
            self._logger.error("Playback error: %s", e)
        finally:
            self._current_stream = None

    async def _process_interrupts(self) -> None:
        """Listen for interrupt events and immediately stop playback."""
        while True:
            await self._interrupt_queue.get()
            self._logger.info("INTERRUPT — stopping playback")

            self._is_interrupted = True

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