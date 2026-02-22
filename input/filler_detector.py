"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE — input/filler_detector.py                            ║
║           GATE 2 — WORD-LIST FILLER DETECTION FOR SPEAKING MONITOR              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Gate 2 of the three-gate interrupt system. Checks whether a partial
    transcript consists entirely of filler / backchannel words like "yeah",
    "uh-huh", "okay", etc.

    If the entire utterance matches the filler list, it is classified as an
    implicit backchannel and does NOT need to go to Gate 3 (the LLM classifier).
    This saves an API call and ~400ms of latency.

    Runs locally in ~2ms. No API calls.
"""

import logging
import re
from typing import Optional


# Default filler word list — can be overridden from config.yaml
DEFAULT_FILLERS = frozenset({
    "yeah", "yep", "okay", "ok", "right", "sure",
    "uh-huh", "uh huh", "mm-hmm", "mm hmm", "mmhmm",
    "haha", "ha ha", "lol", "wow", "oh", "nice",
    "cool", "great", "mhm", "hmm", "hm", "um",
    "uh", "ah", "ooh", "yea", "ya", "yup",
})


class FillerDetector:
    """Gate 2 — Detects pure filler / backchannel words in partial transcripts.

    Returns True + the matched word if the transcript is nothing but filler.
    Returns False if the transcript contains meaningful content that should
    proceed to Gate 3 (Speaking Monitor LLM classification).
    """

    def __init__(self, config: dict | None = None) -> None:
        self._logger = logging.getLogger("FillerDetector")

        # Load filler list: ALWAYS start with defaults, MERGE any extras from config
        extra_words = config.get("gate2_filler_list", []) if config else []
        extra = frozenset(w.lower().strip() for w in extra_words if w)
        self._fillers: frozenset[str] = DEFAULT_FILLERS | extra

        self._logger.info(
            "FillerDetector initialized with %d filler words (%d defaults + %d extras)",
            len(self._fillers), len(DEFAULT_FILLERS), len(extra - DEFAULT_FILLERS),
        )

    def is_filler(self, text: str) -> tuple[bool, Optional[str]]:
        """Check whether the transcript is pure filler.

        Args:
            text: Partial transcript from STT (may be noisy / short).

        Returns:
            (True, matched_word)  — if the entire text is filler
            (False, None)         — if the text has meaningful content
        """
        if not text or not text.strip():
            return True, None  # empty = nothing meaningful

        # Normalise: lowercase, strip punctuation, collapse whitespace
        cleaned = re.sub(r"[^\w\s-]", "", text.lower()).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        if not cleaned:
            return True, None

        # Direct match against filler set
        if cleaned in self._fillers:
            self._logger.debug("Filler detected (exact): '%s'", cleaned)
            return True, cleaned

        # Multi-word check: split and see if every token is a filler
        tokens = cleaned.split()
        if all(tok in self._fillers for tok in tokens):
            matched = " ".join(tokens)
            self._logger.debug("Filler detected (multi-word): '%s'", matched)
            return True, matched

        # Not a filler — meaningful content
        self._logger.debug("Not filler: '%s'", cleaned)
        return False, None
