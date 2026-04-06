"""
VoxCore — brain/emotion_tagger.py
Ensures emotion tags in speech output — fallback when LLM forgets.
"""

import re
import logging

EMOTION_TAG_PATTERN = re.compile(r'^\s*\[(\w+)\]')

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "excited": [
        "great", "awesome", "amazing", "fantastic", "wonderful", "love",
        "excellent", "brilliant", "perfect", "wow", "incredible", "exciting",
    ],
    "empathetic": [
        "sorry", "understand", "frustrating", "difficult", "tough",
        "challenging", "hear that", "feel for", "sympathize",
    ],
    "concerned": [
        "careful", "warning", "caution", "danger", "risk", "worry",
        "problem", "issue", "trouble", "unfortunately",
    ],
    "curious": [
        "interesting", "wonder", "hmm", "let me think", "curious",
        "fascinating", "intriguing",
    ],
    "surprised": [
        "really", "wow", "no way", "unexpected", "surprising",
        "didn't expect", "oh",
    ],
    "cheerful": [
        "sure", "happy", "glad", "of course", "absolutely", "definitely",
        "yes", "you bet",
    ],
}

DEFAULT_EMOTION = "calm"

# Sentence boundary pattern: split on .  ?  !  followed by whitespace or end
_SENTENCE_SPLIT = re.compile(r'(?<=[.?!])\s+')


class EmotionTagger:
    """Ensures every sentence sent to TTS has an emotion tag."""

    def __init__(self, default_emotion: str = DEFAULT_EMOTION) -> None:
        self.default_emotion = default_emotion
        self._logger = logging.getLogger("EmotionTagger")

    # ── public API ─────────────────────────────────────────────────────────

    def ensure_emotion_tag(self, sentence: str) -> str:
        """Return *sentence* guaranteed to start with an emotion tag."""
        if EMOTION_TAG_PATTERN.match(sentence):
            return sentence
        emotion = self._infer_emotion(sentence)
        return f"[{emotion}] {sentence}"

    # ── internals ──────────────────────────────────────────────────────────

    def _infer_emotion(self, sentence: str) -> str:
        """Keyword-based emotion inference. First match wins."""
        lower = sentence.lower()
        for emotion, keywords in EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in lower:
                    return emotion
        return self.default_emotion