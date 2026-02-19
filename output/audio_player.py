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
                - output_sample_rate: int (48000 — Orpheus output rate)
        
        INITIALIZES:
            self.session: Session                = session
            self.event_bus: EventBus             = event_bus
            self.output_sample_rate: int         = config.get("output_sample_rate", 48000)
            
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
    - Orpheus TTS outputs 48kHz WAV audio
    - Backchannel clips are also 48kHz WAV (generated by Orpheus)
    - sounddevice handles sample rate conversion automatically if needed
    - Audio frames are 20ms (960 samples at 48kHz) for interrupt granularity

LATENCY:
    - sounddevice output latency: ~10ms
    - This is the final step — after this, the user hears the AI's voice
"""
