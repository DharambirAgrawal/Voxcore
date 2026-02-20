"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — backchannel/__init__.py                           ║
║                  BACKCHANNEL PACKAGE — THE NATURALNESS LAYER                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Package initializer for the backchannel module. This module handles
    the "uh-huh", "mm-hmm", "go on" responses that make the AI feel natural
    and engaged during the user's speech. Runs on a completely separate
    async track from the main pipeline — never blocks it.

EXPORTS:
    - CueDetector          (from backchannel.cue_detector)
    - BackchannelSelector  (from backchannel.selector)
    - ClipGenerator        (from backchannel.generator)

USAGE:
    from backchannel import CueDetector, BackchannelSelector
"""

from backchannel.cue_detector import CueDetector
from backchannel.selector import BackchannelSelector
from backchannel.generator import ClipGenerator
