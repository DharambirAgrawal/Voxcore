"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — brain/emotion_tagger.py                          ║
║      ENSURES EMOTION TAGS IN SPEECH OUTPUT — FALLBACK WHEN LLM FORGETS         ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    A lightweight post-processor on speech tokens. If the LLM output does NOT
    contain an emotion tag in a sentence, this module infers one based on
    content keywords and prepends it.

    This ensures Orpheus TTS ALWAYS gets emotional context even when the LLM
    forgets to include a tag. Without emotion tags, Orpheus defaults to a
    flat monotone — the tags are what give the voice personality.

    Examples:
        "I'm sorry to hear that" → "[empathetic] I'm sorry to hear that"
        "That's great news!" → "[excited] That's great news!"
        "Let me think about that" → "[calm] Let me think about that"

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import re                               # Regex for emotion tag detection
import logging                          # Module logger

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

EMOTION_TAG_PATTERN = re.compile(r'^\s*\[(\w+)\]')  # Match [emotion] at start of sentence

# Keyword-to-emotion mapping for inference (checked in order — first match wins)
EMOTION_KEYWORDS = {
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

DEFAULT_EMOTION = "calm"   # When no keyword match is found

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: EmotionTagger
──────────────────────────────────────────────────────────────────────────────────
    Ensures every sentence sent to TTS has an emotion tag.

    CONSTRUCTOR: __init__(self, default_emotion: str = "calm")
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - default_emotion: str — Fallback emotion when no keywords match (default "calm")
        
        INITIALIZES:
            self.default_emotion: str = default_emotion
            self._logger: logging.Logger = logging.getLogger("EmotionTagger")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def ensure_emotion_tag(self, sentence: str) -> str
        INPUTS:
            - sentence: str — A sentence from ResponseParser (may or may not have emotion tag)
        OUTPUT:
            - str — The sentence guaranteed to start with an emotion tag
        WHAT IT DOES:
            1. Check if sentence already has an emotion tag:
               match = EMOTION_TAG_PATTERN.match(sentence)
               If match: return sentence as-is (already tagged)
            2. Infer emotion from content:
               emotion = self._infer_emotion(sentence)
            3. Prepend tag:
               return f"[{emotion}] {sentence}"

    def _infer_emotion(self, sentence: str) -> str
        INPUTS:
            - sentence: str — Untagged sentence text
        OUTPUT:
            - str — Inferred emotion name (e.g. "excited", "empathetic", "calm")
        WHAT IT DOES:
            1. Convert sentence to lowercase for matching
            2. For each (emotion, keywords) in EMOTION_KEYWORDS.items():
               For each keyword in keywords:
                   If keyword in sentence_lower:
                       Return emotion
            3. If no match found: return self.default_emotion

    def extract_emotion(self, sentence: str) -> tuple[str, str]
        INPUTS:
            - sentence: str — A sentence that may contain an emotion tag
        OUTPUT:
            - tuple[str, str] — (emotion_tag, clean_sentence)
                - emotion_tag: str — The emotion (e.g. "cheerful") or "" if none
                - clean_sentence: str — The sentence with the emotion tag removed
        WHAT IT DOES:
            1. match = EMOTION_TAG_PATTERN.match(sentence)
            2. If match:
               emotion = match.group(1)
               clean = sentence[match.end():].strip()
               Return (emotion, clean)
            3. Return ("", sentence)
        
        Used by TTS client to separate the emotion tag from the text
        if needed for API formatting.

    def tag_full_response(self, response: str) -> str
        INPUTS:
            - response: str — Full multi-sentence response from LLM
        OUTPUT:
            - str — Response with emotion tags on every sentence
        WHAT IT DOES:
            1. Split response into sentences (on . ? ! boundaries)
            2. For each sentence: ensure_emotion_tag(sentence)
            3. Join back together
            4. Return tagged response

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - EmotionTagger       (class)
    - EMOTION_KEYWORDS    (dict constant)
    - EMOTION_TAG_PATTERN (compiled regex)
    - DEFAULT_EMOTION     (str constant)
═══════════════════════════════════════════════════════════════════════════════════

ORPHEUS EMOTION TAGS REFERENCE:
    Orpheus TTS supports these inline vocal direction tags:
    [cheerful] [calm] [concerned] [excited] [empathetic] [curious]
    [surprised] [sad] [angry] [whisper] [laugh]
    
    Tags affect prosody, pitch, and speaking rate. They can appear
    anywhere in the text but work best at the start of a phrase.
"""
