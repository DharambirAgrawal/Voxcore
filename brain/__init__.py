"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         VOXCORE — brain/__init__.py                             ║
║                        BRAIN PACKAGE — THE MIND                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Package initializer for the brain module. Contains all LLM-related logic:
    streaming client, prompt construction, response parsing, emotion tagging,
    and model routing.

EXPORTS:
    - LLMClient        (from brain.llm_client)
    - PromptBuilder    (from brain.prompt_builder)
    - ResponseParser   (from brain.response_parser)
    - EmotionTagger    (from brain.emotion_tagger)
    - BrainRouter      (from brain.router)

USAGE:
    from brain import LLMClient, PromptBuilder, ResponseParser
"""

from brain.llm_client import LLMClient
from brain.prompt_builder import PromptBuilder
from brain.response_parser import ResponseParser
from brain.emotion_tagger import EmotionTagger
from brain.router import BrainRouter
