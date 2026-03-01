"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE — input/gate0_echo_check.py                          ║
║           LAYER 3: SPECTRAL + TEMPORAL ECHO GATE (BEFORE GATE 1)               ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Catches loud echo residual that survived denoiser and AriaVoiceFilter.
    When speakers are loud or the room has strong acoustics, echo residual
    can be energetic enough to pass Gate 1's energy threshold.
    Gate 0 catches it before Gate 1 runs.

    Two independent checks:
      1. Temporal — was TTS playing very recently? (<80ms ago → echo)
      2. Spectral — does mic frequency profile match TTS EMA fingerprint?

    Runs only during SPEAKING state (filter_active=True in mic_stream).
    Pre-warmed at startup from synthesize_silent() output.
"""

import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class Gate0EchoCheck:
    """
    Spectral + temporal echo gate.
    Runs before Gate 1, only during SPEAKING state.
    Pre-warmed at startup from synthesize_silent() output.

    Maintains a rolling exponential moving average of recent TTS
    output spectra. Compares each mic chunk's spectrum against it.
    High similarity = mic sounds like TTS = echo → drop.
    """

    def __init__(
        self,
        spectral_threshold: float = 0.85,
        temporal_gate_ms: int = 80,
        ema_alpha: float = 0.1,
    ):
        self._threshold = spectral_threshold
        self._temporal_gate_ms = temporal_gate_ms
        self._ema_alpha = ema_alpha
        self._tts_spectrum_ema: np.ndarray | None = None
        self._is_ready = False
        self._last_tts_chunk_time: float = 0.0
        self._logger = logging.getLogger("Gate0")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def feed_tts_spectrum(self, chunk: np.ndarray) -> None:
        """
        Update rolling spectral fingerprint with TTS output.

        Called by:
          1. startup_warmup() — from synthesize_silent() output (pre-warm)
          2. audio_player._play_tts_chunk() — from live TTS output

        NOT called for clips. Same exclusion rule as AriaVoiceFilter.
        """
        self._last_tts_chunk_time = time.time()

        spectrum = np.abs(np.fft.rfft(chunk.astype(np.float32)))

        if self._tts_spectrum_ema is None:
            self._tts_spectrum_ema = spectrum
        else:
            # Truncate/pad to match lengths (chunk sizes may vary)
            min_len = min(len(spectrum), len(self._tts_spectrum_ema))
            self._tts_spectrum_ema = (
                self._ema_alpha * spectrum[:min_len]
                + (1 - self._ema_alpha) * self._tts_spectrum_ema[:min_len]
            )
        self._is_ready = True

    def is_echo(self, mic_chunk: np.ndarray) -> bool:
        """
        Check if mic audio is echo from TTS output.

        Returns:
            True  → mic sounds like recent TTS output → drop (echo)
            False → mic sounds spectrally different → pass to Gate 1

        Two independent checks — either alone can catch echo:
          Check 1: Temporal — was TTS playing very recently?
          Check 2: Spectral — does mic frequency profile match TTS?
        """
        if not self._is_ready:
            return False

        # Check 1: Temporal gate
        # Echo physically cannot arrive before TTS plays.
        # If TTS chunk played < 80ms ago, any mic audio is likely echo.
        ms_since_tts = (time.time() - self._last_tts_chunk_time) * 1000
        if ms_since_tts < self._temporal_gate_ms:
            return True

        # Check 2: Spectral similarity
        mic_spectrum = np.abs(np.fft.rfft(mic_chunk.astype(np.float32)))

        # Match lengths for comparison
        min_len = min(len(mic_spectrum), len(self._tts_spectrum_ema))
        if min_len == 0:
            return False

        mic_norm = mic_spectrum[:min_len]
        tts_norm = self._tts_spectrum_ema[:min_len]

        # Normalize to unit vectors (cosine similarity)
        mic_mag = np.linalg.norm(mic_norm) + 1e-8
        tts_mag = np.linalg.norm(tts_norm) + 1e-8

        similarity = float(np.dot(mic_norm / mic_mag, tts_norm / tts_mag))
        return similarity > self._threshold

    def reset(self) -> None:
        """Reset on persona/voice change."""
        self._tts_spectrum_ema = None
        self._is_ready = False
        self._last_tts_chunk_time = 0.0
        self._logger.info("Gate0 reset")
