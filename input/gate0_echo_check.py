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

    V3 fix: Uses a fixed FFT size (512 points) with zero-padding to
    guarantee consistent spectral resolution (257 bins) regardless
    of input chunk size. Previous implementation truncated to the
    shortest chunk ever fed, permanently degrading to as few as 6 bins.
    """

    # Fixed FFT size — guarantees 257 bins (512/2 + 1) regardless of chunk size.
    # 512 samples at 16 kHz = 32ms — good spectral resolution for voice.
    _FFT_SIZE: int = 512

    def __init__(
        self,
        spectral_threshold: float = 0.92,
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

    def _fixed_spectrum(self, chunk: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrum with fixed FFT size (zero-padded).

        Always produces exactly (_FFT_SIZE // 2 + 1) bins regardless of
        input chunk length. Short chunks are zero-padded; long chunks are
        truncated to _FFT_SIZE samples before FFT.
        """
        data = chunk.astype(np.float32)
        if len(data) >= self._FFT_SIZE:
            data = data[:self._FFT_SIZE]
        else:
            # Zero-pad to _FFT_SIZE
            padded = np.zeros(self._FFT_SIZE, dtype=np.float32)
            padded[:len(data)] = data
            data = padded
        return np.abs(np.fft.rfft(data))

    def feed_tts_spectrum(self, chunk: np.ndarray) -> None:
        """
        Update rolling spectral fingerprint with TTS output.

        Called by:
          1. startup_warmup() — from synthesize_silent() output (pre-warm)
          2. audio_player._play_tts_chunk() — from live TTS output

        NOT called for clips. Same exclusion rule as AriaVoiceFilter.

        V3 fix: Uses fixed FFT size so EMA bin count never degrades.
        """
        self._last_tts_chunk_time = time.time()

        spectrum = self._fixed_spectrum(chunk)

        if self._tts_spectrum_ema is None:
            self._tts_spectrum_ema = spectrum.copy()
            self._logger.debug("Gate0 EMA initialized (bins=%d, ready=True)", len(spectrum))
        else:
            self._tts_spectrum_ema = (
                self._ema_alpha * spectrum
                + (1 - self._ema_alpha) * self._tts_spectrum_ema
            )
            self._logger.debug("Gate0 EMA updated (bins=%d, samples=%d)", len(spectrum), len(chunk))
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

        # Check 1: Temporal gate (REMOVED)
        # Previously, this blocked ALL audio for 80ms after any TTS chunk.
        # Since TTS plays continuously, it muted the mic entirely.
        # We now rely purely on the spectral check below for live double-talk.
        pass

        # Check 2: Spectral similarity
        mic_spectrum = self._fixed_spectrum(mic_chunk)

        # Both spectra are guaranteed to have the same length (_FFT_SIZE//2 + 1)
        mic_norm = mic_spectrum
        tts_norm = self._tts_spectrum_ema

        # Normalize to unit vectors (cosine similarity)
        mic_mag = np.linalg.norm(mic_norm) + 1e-8
        tts_mag = np.linalg.norm(tts_norm) + 1e-8

        similarity = float(np.dot(mic_norm / mic_mag, tts_norm / tts_mag))
        if similarity > self._threshold:
            self._logger.debug("Gate0 BLOCKED (spectral): sim=%.3f > %.2f threshold",
                               similarity, self._threshold)
            return True

        self._logger.debug("Gate0 PASSED: spectral=%.3f", similarity)
        return False

    def reset(self) -> None:
        """Reset on persona/voice change."""
        self._tts_spectrum_ema = None
        self._is_ready = False
        self._last_tts_chunk_time = 0.0
        self._logger.info("Gate0 reset")
