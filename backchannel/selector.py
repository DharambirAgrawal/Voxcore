"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — backchannel/selector.py                          ║
║       SELECTS CONTEXT-APPROPRIATE BACKCHANNEL CLIP TO PLAY                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Subscribes to BACKCHANNEL_OPPORTUNITY events from CueDetector. Uses
    weighted random selection to pick a context-appropriate audio clip from
    the pre-generated clips bank. Publishes BACKCHANNEL_FIRE with the chosen
    clip path for the AudioPlayer to play.

    Guard conditions prevent excessive or poorly-timed backchannels:
    - Minimum 8 seconds between backchannels
    - Never during SPEAKING or THINKING state
    - Never in first 2 seconds of user speech
    - Never while another backchannel is playing

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import random                           # Weighted random selection
import time                             # Timing checks
from pathlib import Path                # Clip file paths

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

CLIP_CATEGORIES — dict mapping context labels to lists of clip names with weights:

    CLIP_CATEGORIES = {
        "agreement": [
            ("mm_hmm", 3),      # Most common agreement backchannel
            ("yeah", 2),
            ("right", 2),
            ("sure", 1),
            ("got_it", 1),
            ("makes_sense", 1),
        ],
        "encouragement": [
            ("go_on", 3),
            ("tell_me_more", 2),
            ("uh_huh", 2),
            ("okay", 1),
        ],
        "surprise": [
            ("interesting", 3),
            ("oh_wow", 2),
            ("really", 2),
        ],
        "understanding": [
            ("i_see", 3),
            ("understood", 2),
            ("makes_sense", 2),
            ("got_it", 1),
        ],
        "filler": [
            ("uh_huh", 3),
            ("mm_hmm", 3),
            ("yeah", 2),
            ("okay", 1),
        ],
    }

    The integer is the relative weight for random.choices(). Higher = more likely.

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: BackchannelSelector
──────────────────────────────────────────────────────────────────────────────────
    Picks and fires contextually appropriate backchannel clips.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — For checking TurnState
            - event_bus: EventBus — Subscribe to BACKCHANNEL_OPPORTUNITY, publish BACKCHANNEL_FIRE
            - config: dict — The "backchannel" section from config.yaml:
                - clips_dir: str ("backchannel/clips/heart/")
                - min_gap_between_s: int (8)
                - min_user_speech_s: int (2)
        
        INITIALIZES:
            self.session: Session                = session
            self.event_bus: EventBus             = event_bus
            self.clips_dir: Path                 = Path(config["backchannel"].get("clips_dir", "backchannel/clips/heart/"))
            self.min_gap_between_s: int          = config["backchannel"].get("min_gap_between_s", 8)
            self.min_user_speech_s: int          = config["backchannel"].get("min_user_speech_s", 2)
            
            self._last_fired_time: float         = 0.0
            self._last_clip_name: str            = ""     # Avoid repeating same clip twice in a row
            self._opportunity_queue: asyncio.Queue = event_bus.subscribe(EventType.BACKCHANNEL_OPPORTUNITY)
            self._available_clips: set[str]      = set()  # Populated in _scan_clips()
            
            self._logger: logging.Logger         = logging.getLogger("BackchannelSelector")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Calls self._scan_clips() to discover available WAV files
            2. Enters infinite loop:
               a. event = await self._opportunity_queue.get()
               b. If not self._should_fire(event): continue
               c. category = self._determine_category(event.data)
               d. clip_name = self._select_clip(category)
               e. clip_path = self.clips_dir / f"{clip_name}.wav"
               f. If clip_path does not exist:
                  Log warning, fall back to any available clip
               g. Publish BACKCHANNEL_FIRE event with data:
                  {"clip_name": clip_name, "clip_path": str(clip_path)}
               h. Set self._last_fired_time = time.time()
               i. Set self._last_clip_name = clip_name
               j. Log: "Backchannel: playing '{clip_name}'"

    def _scan_clips(self) -> None
        INPUTS: None
        OUTPUT: None (populates self._available_clips)
        WHAT IT DOES:
            1. Scans self.clips_dir for *.wav files
            2. Adds each stem (filename without .wav) to self._available_clips
            3. Logs: "Found {n} backchannel clips: {list}"
            4. If no clips found, logs warning:
               "No backchannel clips found! Run 'python -m backchannel.generator' first."

    def _should_fire(self, event) -> bool
        INPUTS:
            - event: Event — BACKCHANNEL_OPPORTUNITY event
        OUTPUT: bool — Whether we should actually fire a backchannel
        WHAT IT DOES:
            Returns False if ANY of these are true:
            1. session.state != TurnState.LISTENING (agent is speaking/thinking)
            2. time.time() - self._last_fired_time < self.min_gap_between_s
            3. event.data["speech_duration_s"] < self.min_user_speech_s
            4. No clips are available (self._available_clips is empty)
            
            Returns True otherwise.

    def _determine_category(self, data: dict) -> str
        INPUTS:
            - data: dict — BACKCHANNEL_OPPORTUNITY payload:
                - is_question: bool
                - energy_level: float
                - speech_duration_s: float
        OUTPUT: str — One of: "agreement", "encouragement", "surprise", "understanding", "filler"
        WHAT IT DOES:
            Decision logic:
            1. If data["is_question"]: return "agreement" (respond to questions with affirmation)
            2. If data["speech_duration_s"] > 10: return "encouragement" (long monologue → "go on")
            3. If data["energy_level"] > 0.08: return "surprise" (high energy = emphasis → "interesting")
            4. If data["speech_duration_s"] > 5: return "understanding" (medium speech → "I see")
            5. Default: return "filler" (general → "uh-huh")

    def _select_clip(self, category: str) -> str
        INPUTS:
            - category: str — The backchannel category
        OUTPUT: str — Selected clip name (e.g. "mm_hmm", "go_on")
        WHAT IT DOES:
            1. Gets clip list from CLIP_CATEGORIES[category]
            2. Filters to only clips that exist in self._available_clips
            3. Filters out self._last_clip_name (avoid immediate repeat)
            4. If no clips remain after filtering, fall back to any available clip
            5. Uses random.choices() with weights to select one clip
            6. Returns the clip name (without .wav extension)

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - BackchannelSelector   (class)
    - CLIP_CATEGORIES       (dict constant)
═══════════════════════════════════════════════════════════════════════════════════

CLIP PLAYBACK NOTE:
    BackchannelSelector does NOT play the clip itself. It publishes BACKCHANNEL_FIRE
    with the clip path, and AudioPlayer subscribes to this event and plays the clip.
    This maintains the zero-coupling architecture through the event bus.
"""


"""
VOXCORE — backchannel/selector.py
Selects context-appropriate backchannel clip to play.
"""

import asyncio
import logging
import random
import time
from pathlib import Path

from core.session import Session, TurnState
from core.event_bus import EventBus, EventType

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CLIP_CATEGORIES: dict[str, list[tuple[str, int]]] = {
    "agreement": [
        ("mm_hmm", 3),
        ("yeah", 2),
        ("right", 2),
        ("sure", 1),
        ("got_it", 1),
        ("makes_sense", 1),
    ],
    "encouragement": [
        ("go_on", 3),
        ("tell_me_more", 2),
        ("uh_huh", 2),
        ("okay", 1),
    ],
    "surprise": [
        ("interesting", 3),
        ("oh_wow", 2),
        ("really", 2),
    ],
    "understanding": [
        ("i_see", 3),
        ("understood", 2),
        ("makes_sense", 2),
        ("got_it", 1),
    ],
    "filler": [
        ("uh_huh", 3),
        ("mm_hmm", 3),
        ("yeah", 2),
        ("okay", 1),
    ],
}

# Energy threshold for "high energy" detection
_HIGH_ENERGY_THRESHOLD = 0.08
# Speech duration thresholds (seconds)
_LONG_MONOLOGUE_S = 10
_MEDIUM_SPEECH_S = 5


class BackchannelSelector:
    """Picks and fires contextually appropriate backchannel clips."""

    def __init__(self, session: Session, event_bus: EventBus, config: dict) -> None:
        self.session: Session = session
        self.event_bus: EventBus = event_bus

        bc_cfg = config.get("backchannel", {})
        self.clips_dir: Path = Path(bc_cfg.get("clips_dir", "backchannel/clips/heart/"))
        self.min_gap_between_s: int = bc_cfg.get("min_gap_between_s", 8)
        self.min_user_speech_s: int = bc_cfg.get("min_user_speech_s", 2)

        self._last_fired_time: float = 0.0
        self._last_clip_name: str = ""
        self._available_clips: set[str] = set()

        self._logger: logging.Logger = logging.getLogger("BackchannelSelector")

    async def run(self) -> None:
        """Main loop — listen for opportunities and fire backchannels."""
        self._scan_clips()

        # Subscribe to backchannel opportunity events
        queue = self.event_bus.subscribe(EventType.BACKCHANNEL_OPPORTUNITY)

        self._logger.info("BackchannelSelector running")

        while True:
            try:
                event = await queue.get()

                if not self._should_fire(event):
                    continue

                data = event.get("data", event) if isinstance(event, dict) else {}
                category = self._determine_category(data)
                clip_name = self._select_clip(category)

                if clip_name is None:
                    self._logger.warning("No clip available for category '%s'", category)
                    continue

                clip_path = self.clips_dir / f"{clip_name}.wav"

                if not clip_path.exists():
                    self._logger.warning(
                        "Clip file missing: %s — falling back", clip_path
                    )
                    # Fall back to any available clip
                    clip_name = self._select_any_available()
                    if clip_name is None:
                        continue
                    clip_path = self.clips_dir / f"{clip_name}.wav"

                await self.event_bus.publish(
                    EventType.BACKCHANNEL_FIRE,
                    {"clip_name": clip_name, "clip_path": str(clip_path)},
                )

                self._last_fired_time = time.time()
                self._last_clip_name = clip_name
                self._logger.info("Backchannel: playing '%s'", clip_name)

            except asyncio.CancelledError:
                self._logger.info("BackchannelSelector cancelled")
                break
            except Exception:
                self._logger.exception("Error in BackchannelSelector loop")
                await asyncio.sleep(0.1)

    def _scan_clips(self) -> None:
        """Discover available WAV clips in the clips directory."""
        self._available_clips.clear()

        if not self.clips_dir.exists():
            self._logger.warning(
                "No backchannel clips found! Run 'python -m backchannel.generator' first."
            )
            return

        for wav_file in self.clips_dir.glob("*.wav"):
            self._available_clips.add(wav_file.stem)

        if self._available_clips:
            self._logger.info(
                "Found %d backchannel clips: %s",
                len(self._available_clips),
                sorted(self._available_clips),
            )
        else:
            self._logger.warning(
                "No backchannel clips found! Run 'python -m backchannel.generator' first."
            )

    def _should_fire(self, event) -> bool:
        """Check all guard conditions before firing a backchannel."""
        # Must be in LISTENING state
        if self.session.state != TurnState.LISTENING:
            return False

        # Cooldown between backchannels
        if time.time() - self._last_fired_time < self.min_gap_between_s:
            return False

        # Extract event data
        data = event.get("data", event) if isinstance(event, dict) else {}

        # Minimum user speech duration
        speech_duration = data.get("speech_duration_s", 0)
        if speech_duration < self.min_user_speech_s:
            return False

        # Must have clips available
        if not self._available_clips:
            return False

        return True

    def _determine_category(self, data: dict) -> str:
        """Choose backchannel category based on speech context."""
        is_question = data.get("is_question", False)
        energy_level = data.get("energy_level", 0.0)
        speech_duration = data.get("speech_duration_s", 0.0)

        if is_question:
            return "agreement"

        if speech_duration > _LONG_MONOLOGUE_S:
            return "encouragement"

        if energy_level > _HIGH_ENERGY_THRESHOLD:
            return "surprise"

        if speech_duration > _MEDIUM_SPEECH_S:
            return "understanding"

        return "filler"

    def _select_clip(self, category: str) -> str | None:
        """Select a clip from the category using weighted random selection."""
        candidates = CLIP_CATEGORIES.get(category, CLIP_CATEGORIES["filler"])

        # Filter to available clips only
        available = [
            (name, weight)
            for name, weight in candidates
            if name in self._available_clips
        ]

        # Avoid immediate repeat
        if len(available) > 1:
            available = [
                (name, weight)
                for name, weight in available
                if name != self._last_clip_name
            ]

        if not available:
            return self._select_any_available()

        names = [name for name, _ in available]
        weights = [weight for _, weight in available]

        selected = random.choices(names, weights=weights, k=1)
        return selected[0]

    def _select_any_available(self) -> str | None:
        """Fall back to any available clip."""
        candidates = self._available_clips - {self._last_clip_name}
        if not candidates:
            candidates = self._available_clips

        if not candidates:
            return None

        return random.choice(sorted(candidates))