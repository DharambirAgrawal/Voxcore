"""
VoxCore - output/voice_profile.py
Configuration holder for TTS voice settings (Kokoro-ONNX voice ID, language,
sample rate, model paths). Loaded from config.yaml at startup.
"""


import logging
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

AVAILABLE_VOICES = [
    # American Female
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # American Male
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck",
    # British Female
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    # British Male
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

DEFAULT_VOICE = "af_heart"

VALID_EMOTIONS = [
    "cheerful", "calm", "concerned", "excited", "empathetic",
    "curious", "surprised", "sad", "angry", "whisper", "laugh",
]


class VoiceProfile:
    """Holds all voice-related configuration for the current persona."""

    def __init__(self, config: dict) -> None:
        self.voice_name: str = config.get("persona", {}).get("voice", DEFAULT_VOICE)
        self.language: str = config.get("persona", {}).get("language", "en")
        self.sample_rate: int = config.get("audio", {}).get("output_sample_rate", 24000)
        self.response_format: str = "wav"
        self.default_emotion: str = "calm"
        self.tts_model_path: str = config.get("models", {}).get("tts_model", "models/kokoro-v1.0.onnx")
        self.tts_voices_path: str = config.get("models", {}).get("tts_voices", "models/voices-v1.0.bin")

        self._logger: logging.Logger = logging.getLogger("VoiceProfile")

        self._validate()

    def _validate(self) -> None:
        """Validate voice name against available Kokoro voices."""
        if self.voice_name not in AVAILABLE_VOICES:
            self._logger.warning(
                "Unknown Kokoro voice '%s'. Available: %s. Falling back to '%s'.",
                self.voice_name, AVAILABLE_VOICES, DEFAULT_VOICE,
            )
            self.voice_name = DEFAULT_VOICE

    def get_clips_subdir(self) -> str:
        """Return voice short-name for clips subdirectory (e.g. 'heart' from 'af_heart')."""
        if "_" in self.voice_name:
            return self.voice_name.split("_", 1)[1]
        return self.voice_name

    def to_dict(self) -> dict:
        """Return all voice settings as a dictionary."""
        return {
            "voice_name": self.voice_name,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "response_format": self.response_format,
            "default_emotion": self.default_emotion,
            "tts_engine": "kokoro-onnx",
            "tts_model_path": self.tts_model_path,
            "clips_subdir": self.get_clips_subdir(),
        }

    def update(
        self,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        default_emotion: Optional[str] = None,
    ) -> None:
        """Hot-reload voice settings at runtime."""
        if voice:
            self.voice_name = voice
        if language:
            self.language = language
        if default_emotion and default_emotion in VALID_EMOTIONS:
            self.default_emotion = default_emotion
        self._validate()