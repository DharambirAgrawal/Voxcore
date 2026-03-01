"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE — input/aria_voice_filter.py                         ║
║           LAYER 2: RESEMBLYZER SPEAKER-IDENTITY ECHO FILTER                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Holds Aria's speaker embedding (256-dim vector from resemblyzer).
    Pre-warmed at startup via feed_aria_audio() from synthesize_silent().
    Rejects mic chunks whose cosine similarity to Aria exceeds threshold.
    Passes all other voices unchanged.

    ON only while TTS real speech is playing (filter_active=True in mic_stream).
    OFF during clips, pauses, LISTENING state.
"""

import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

# Lazy-load resemblyzer to avoid import cost at module level
_encoder = None


def _get_encoder():
    """Lazy-load VoiceEncoder — ~1s one-time cost."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
        logger.info("VoiceEncoder loaded")
    return _encoder


class AriaVoiceFilter:
    """
    Speaker-identity echo filter using resemblyzer embeddings.

    Pre-warmed at startup via feed_aria_audio() from synthesize_silent().
    Updated continuously from live TTS output during conversation.
    Rejects mic chunks that sound like Aria.
    Passes all other voices unchanged.

    NOT called for: breath.wav, mm.wav, laughs.wav, backchannel clips,
    bridge clips (sure.wav, got_it.wav). Only real Kokoro TTS speech.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self._threshold = similarity_threshold
        self._aria_chunks: deque = deque(maxlen=50)  # ~10s rolling window
        self._aria_embedding: np.ndarray | None = None
        self._embedding_ready = False
        self._samples_since_update = 0
        self._logger = logging.getLogger("AriaVoiceFilter")

    @property
    def is_ready(self) -> bool:
        return self._embedding_ready

    def feed_aria_audio(self, chunk: np.ndarray) -> None:
        """
        Feed Aria's audio to build/update speaker embedding.

        Called by:
          1. startup_warmup() — from synthesize_silent() output (pre-warm)
          2. audio_player._play_tts_chunk() — from live TTS during conversation

        NOT called for clips, backchannels, or bridge audio.
        """
        self._aria_chunks.append(chunk)
        self._samples_since_update += len(chunk)

        # Recompute embedding every ~2s of new Aria speech (32000 samples @ 16kHz)
        if self._samples_since_update >= 32000:
            self._samples_since_update = 0
            try:
                encoder = _get_encoder()
                from resemblyzer import preprocess_wav
                combined = np.concatenate(list(self._aria_chunks))
                wav = preprocess_wav(combined, source_sr=16000)
                if len(wav) > 0:
                    self._aria_embedding = encoder.embed_utterance(wav)
                    self._embedding_ready = True
                    self._logger.debug("Aria embedding updated (%d samples)", len(combined))
            except Exception as e:
                self._logger.error("Failed to compute Aria embedding: %s", e)

    def is_aria_echo(self, chunk: np.ndarray) -> bool:
        """
        Check if mic chunk sounds like Aria (echo).

        Returns:
            True  → sounds like Aria → drop (echo)
            False → sounds different from Aria → pass to Gate 0

        Before is_ready: always returns False.
        After startup warmup, is_ready should always be True.
        """
        if not self._embedding_ready or len(chunk) < 3200:
            return False

        try:
            encoder = _get_encoder()
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(chunk, source_sr=16000)
            if len(wav) == 0:
                return False
            mic_embedding = encoder.embed_utterance(wav)
            similarity = float(np.dot(self._aria_embedding, mic_embedding))
            return similarity > self._threshold
        except Exception:
            return False

    def reset(self) -> None:
        """Reset on persona/voice change. Re-warmup required after reset."""
        self._aria_chunks.clear()
        self._aria_embedding = None
        self._embedding_ready = False
        self._samples_since_update = 0
        self._logger.info("AriaVoiceFilter reset — re-warmup required")
