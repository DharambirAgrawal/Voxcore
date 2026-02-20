"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — output/voice_profile.py                          ║
║          VOICE CONFIGURATION — HOLDS PERSONA VOICE SETTINGS                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Simple configuration holder for the TTS voice settings. Loaded from
    config.yaml at startup and referenced by TTSClient and ClipGenerator.

    Contains: chosen Orpheus voice name, default emotion tag, sample rates,
    response format, and any voice-specific settings.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass       # Simple data holder
import logging                          # Module logger

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

AVAILABLE_VOICES_EN = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
AVAILABLE_VOICES_AR = ["fahad", "sultan", "lulwa", "noura"]

VALID_EMOTIONS = [
    "cheerful", "calm", "concerned", "excited", "empathetic",
    "curious", "surprised", "sad", "angry", "whisper", "laugh"
]

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: VoiceProfile
──────────────────────────────────────────────────────────────────────────────────
    Holds all voice-related configuration for the current persona.

    CONSTRUCTOR: __init__(self, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - config: dict — The "persona" section from config.yaml:
                - voice: str ("tara")
                - language: str ("en")
              AND the "audio" section:
                - output_sample_rate: int (48000)
        
        INITIALIZES:
            self.voice_name: str         = config["persona"].get("voice", "tara")
            self.language: str           = config["persona"].get("language", "en")
            self.sample_rate: int        = config["audio"].get("output_sample_rate", 48000)
            self.response_format: str    = "wav"          # Orpheus output format
            self.default_emotion: str    = "calm"         # Default emotion when none specified
            
            self._logger: logging.Logger = logging.getLogger("VoiceProfile")
            
            # Validate voice name
            self._validate()

    METHODS:
    ─────────────────────────────────────────────────────────────

    def _validate(self) -> None
        INPUTS: None
        OUTPUT: None (raises ValueError on invalid config)
        WHAT IT DOES:
            1. If self.language == "en":
               If self.voice_name not in AVAILABLE_VOICES_EN:
                   Log warning: "Unknown voice '{voice}'. Available: {AVAILABLE_VOICES_EN}"
                   self.voice_name = "tara"  # Fall back to default
            2. If self.language == "ar":
               If self.voice_name not in AVAILABLE_VOICES_AR:
                   Log warning: "Unknown Arabic voice. Available: {AVAILABLE_VOICES_AR}"
                   self.voice_name = "fahad"

    def get_tts_model(self) -> str
        INPUTS: None
        OUTPUT: str — The appropriate Orpheus model for the language
        WHAT IT DOES:
            If self.language == "ar":
                return "canopylabs/orpheus-arabic-saudi"
            return "canopylabs/orpheus-v1-english"

    def to_dict(self) -> dict
        INPUTS: None
        OUTPUT: dict — All voice settings as a dictionary
        WHAT IT DOES:
            Returns {
                "voice_name": self.voice_name,
                "language": self.language,
                "sample_rate": self.sample_rate,
                "response_format": self.response_format,
                "default_emotion": self.default_emotion,
                "tts_model": self.get_tts_model(),
            }

    def update(self, voice: str = None, language: str = None, 
               default_emotion: str = None) -> None
        INPUTS:
            - voice: str — New voice name (optional)
            - language: str — New language (optional)
            - default_emotion: str — New default emotion (optional)
        OUTPUT: None
        WHAT IT DOES:
            1. If voice: self.voice_name = voice
            2. If language: self.language = language
            3. If default_emotion and default_emotion in VALID_EMOTIONS:
               self.default_emotion = default_emotion
            4. self._validate()
        
        Used for hot-reloading voice settings via the API endpoint.

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - VoiceProfile          (class)
    - AVAILABLE_VOICES_EN   (list constant)
    - AVAILABLE_VOICES_AR   (list constant)
    - VALID_EMOTIONS        (list constant)
═══════════════════════════════════════════════════════════════════════════════════
"""


import logging
from typing import Optional

# ─── Constants ────────────────────────────────────────────────────────────────

AVAILABLE_VOICES_EN = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
AVAILABLE_VOICES_AR = ["fahad", "sultan", "lulwa", "noura"]

VALID_EMOTIONS = [
    "cheerful", "calm", "concerned", "excited", "empathetic",
    "curious", "surprised", "sad", "angry", "whisper", "laugh",
]


class VoiceProfile:
    """Holds all voice-related configuration for the current persona."""

    def __init__(self, config: dict) -> None:
        self.voice_name: str = config.get("persona", {}).get("voice", "tara")
        self.language: str = config.get("persona", {}).get("language", "en")
        self.sample_rate: int = config.get("audio", {}).get("output_sample_rate", 48000)
        self.response_format: str = "wav"
        self.default_emotion: str = "calm"

        self._logger: logging.Logger = logging.getLogger("VoiceProfile")

        self._validate()

    def _validate(self) -> None:
        """Validate voice name against available voices for the selected language."""
        if self.language == "en":
            if self.voice_name not in AVAILABLE_VOICES_EN:
                self._logger.warning(
                    "Unknown voice '%s'. Available: %s", self.voice_name, AVAILABLE_VOICES_EN
                )
                self.voice_name = "tara"
        elif self.language == "ar":
            if self.voice_name not in AVAILABLE_VOICES_AR:
                self._logger.warning(
                    "Unknown Arabic voice '%s'. Available: %s", self.voice_name, AVAILABLE_VOICES_AR
                )
                self.voice_name = "fahad"

    def get_tts_model(self) -> str:
        """Return the appropriate Orpheus model for the configured language."""
        if self.language == "ar":
            return "canopylabs/orpheus-arabic-saudi"
        return "canopylabs/orpheus-v1-english"

    def to_dict(self) -> dict:
        """Return all voice settings as a dictionary."""
        return {
            "voice_name": self.voice_name,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "response_format": self.response_format,
            "default_emotion": self.default_emotion,
            "tts_model": self.get_tts_model(),
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