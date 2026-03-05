"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  VOXCORE v4 — websocket/shared_resources.py                     ║
║         HEAVY MODELS LOADED ONCE AT BOOT — SHARED ACROSS ALL USERS             ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Loads heavy models once at server boot and shares them across all
    PipelineInstance sessions. This saves significant RAM: resemblyzer ~150MB +
    Kokoro-ONNX ~300MB. Loading once instead of 8x saves ~3.5GB.

SHARED (read-mostly, locks on EMA updates):
    - Kokoro-ONNX TTS engine  (one model, same voice for everyone)
    - AriaVoiceFilter         (resemblyzer + EMA, needs lock for concurrent feed)
    - Gate0EchoCheck           (spectral EMA, needs lock for concurrent feed)
    - Clip files in RAM        (backchannel wavs, generated once at boot)

CONCURRENCY:
    AriaVoiceFilter.feed_aria_audio() and Gate0EchoCheck.feed_tts_spectrum()
    are called concurrently when 2+ users are in SPEAKING state simultaneously.
    threading.Lock() protects the EMA update methods.
"""

import logging
import os
import threading

from kokoro_onnx import Kokoro

from input.aria_voice_filter import AriaVoiceFilter
from input.gate0_echo_check import Gate0EchoCheck
from output.voice_profile import VoiceProfile

logger = logging.getLogger("SharedResources")


class SharedResources:
    """Loaded once at server boot. Passed (read-mostly) to all pipeline instances."""

    def __init__(self, config: dict):
        self.config = config
        logger.info("Loading shared models...")

        # ── Kokoro-ONNX TTS model (~300MB) ────────────────────────────────
        models_cfg = config.get("models", {})
        model_path = models_cfg.get("tts_model", "models/kokoro-v1.0.onnx")
        voices_path = models_cfg.get("tts_voices", "models/voices-v1.0.bin")
        self.kokoro_model: Kokoro = Kokoro(model_path, voices_path)
        logger.info("Kokoro-ONNX loaded from %s", model_path)

        # ── Voice profile (shared config) ─────────────────────────────────
        self.voice_profile: VoiceProfile = VoiceProfile(config=config)

        # ── AriaVoiceFilter (~150MB resemblyzer) ──────────────────────────
        echo_cfg = config.get("echo_suppression", {})
        self.aria_filter: AriaVoiceFilter = AriaVoiceFilter(
            similarity_threshold=echo_cfg.get("aria_similarity_threshold", 0.75),
        )

        # ── Gate0EchoCheck (lightweight spectral) ─────────────────────────
        self.gate0: Gate0EchoCheck = Gate0EchoCheck(
            spectral_threshold=echo_cfg.get("gate0_spectral_threshold", 0.92),
            temporal_gate_ms=echo_cfg.get("gate0_temporal_gate_ms", 80),
            ema_alpha=echo_cfg.get("gate0_ema_alpha", 0.1),
        )

        # ── Concurrency locks for EMA updates ────────────────────────────
        self._aria_lock = threading.Lock()
        self._gate0_lock = threading.Lock()

        logger.info("Shared models loaded.")

    async def warmup(self) -> None:
        """Run once before accepting connections. Warms both echo models.

        Synthesizes ~3s of Aria's voice silently, then feeds the audio to
        AriaVoiceFilter and Gate0EchoCheck so they are calibrated before
        the first real conversation.
        """
        import asyncio
        import numpy as np

        echo_cfg = self.config.get("echo_suppression", {})
        warmup_text = echo_cfg.get(
            "warmup_text",
            "Hello, how are you doing today? I hope you're having a wonderful day.",
        )

        logger.info("Running startup warmup (synthesizing '%s'...)", warmup_text[:40])

        try:
            # Synthesize using shared Kokoro model
            loop = asyncio.get_event_loop()
            persona_cfg = self.config.get("persona", {})
            voice_name = persona_cfg.get("voice", "af_heart")

            samples, sample_rate = await loop.run_in_executor(
                None,
                lambda: self.kokoro_model.create(
                    warmup_text,
                    voice=voice_name,
                    speed=1.0,
                    lang="en-us",
                ),
            )

            # Resample from 24kHz to 16kHz for filter feeding
            if sample_rate != 16000:
                import torch
                import torchaudio.functional as F
                tensor = torch.from_numpy(samples).unsqueeze(0)
                resampled = F.resample(tensor, sample_rate, 16000)
                audio = resampled.squeeze(0).numpy().astype(np.float32)
            else:
                audio = samples.astype(np.float32)

            # Feed AriaVoiceFilter (builds speaker embedding)
            chunk_size = 3200  # 200ms chunks at 16kHz
            with self._aria_lock:
                for i in range(0, len(audio), chunk_size):
                    self.aria_filter.feed_aria_audio(audio[i:i + chunk_size])

            aria_ready = getattr(self.aria_filter, "is_ready", None)
            logger.info(
                "AriaVoiceFilter warmed: ready=%s (%d samples fed)",
                aria_ready, len(audio),
            )

            # Feed Gate0EchoCheck (builds spectral fingerprint)
            chunk_size = 480  # 30ms chunks at 16kHz
            with self._gate0_lock:
                for i in range(0, len(audio), chunk_size):
                    self.gate0.feed_tts_spectrum(audio[i:i + chunk_size])

            gate0_ready = getattr(self.gate0, "is_ready", None)
            logger.info("Gate0EchoCheck warmed: ready=%s", gate0_ready)

            logger.info("Warmup done. Echo protection fully ready.")

        except Exception as e:
            logger.warning("Warmup failed (continuing without warmup): %s", e)
