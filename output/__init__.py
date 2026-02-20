"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                       VOXCORE — output/__init__.py                              ║
║                      OUTPUT PACKAGE — THE VOICE                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Package initializer for the output module. Contains TTS synthesis,
    audio playback, and voice profile management.

EXPORTS:
    - TTSClient      (from output.tts_client)
    - AudioPlayer    (from output.audio_player)
    - VoiceProfile   (from output.voice_profile)

USAGE:
    from output import TTSClient, AudioPlayer, VoiceProfile
"""

from output.tts_client import TTSClient
from output.audio_player import AudioPlayer
from output.voice_profile import VoiceProfile
