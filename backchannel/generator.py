"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — backchannel/generator.py                         ║
║        ONE-TIME SETUP SCRIPT — GENERATES BACKCHANNEL CLIPS VIA ORPHEUS TTS     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    A standalone script (NOT part of the runtime pipeline) that generates all
    backchannel audio clips using the Groq Orpheus TTS API. Run once at setup
    time or whenever you change the persona voice.

    It calls Groq Orpheus TTS to generate 15-20 short clips ("uh-huh", "yeah",
    "I see", "right", "okay go on", "interesting", "mm-hmm") in the configured
    persona voice and saves them as WAV files in backchannel/clips/.

    After generation, these clips play locally forever at zero cost and zero latency.

    RUN WITH:
        python -m backchannel.generator
        python -m backchannel.generator --voice tara --config config.yaml

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations for Groq API calls
import argparse                         # CLI argument parsing
import os                               # File operations, environment variables
import logging                          # Module logger
from pathlib import Path                # Directory and file path handling

import yaml                             # Parse config.yaml for voice setting
from dotenv import load_dotenv          # Load GROQ_API_KEY from .env
from groq import AsyncGroq              # Groq async SDK for TTS

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

BACKCHANNEL_PHRASES — dict mapping clip filename to the text to synthesize:

    BACKCHANNEL_PHRASES = {
        "uh_huh":        "[calm] Uh-huh",
        "yeah":          "[calm] Yeah",
        "i_see":         "[calm] I see",
        "go_on":         "[calm] Go on",
        "interesting":   "[excited] Interesting",
        "right":         "[calm] Right",
        "mm_hmm":        "[calm] Mm-hmm",
        "okay":          "[calm] Okay",
        "sure":          "[calm] Sure",
        "oh_wow":        "[excited] Oh wow",
        "got_it":        "[calm] Got it",
        "tell_me_more":  "[curious] Tell me more",
        "really":        "[surprised] Really?",
        "makes_sense":   "[calm] Makes sense",
        "understood":    "[calm] Understood",
    }

    NOTE: Emotion tags ([calm], [excited], etc.) are Orpheus vocal direction tags.
    They affect the tone/prosody of the generated audio.

DEFAULT_OUTPUT_DIR = "backchannel/clips/"

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: ClipGenerator
──────────────────────────────────────────────────────────────────────────────────
    Generates backchannel WAV clips using Groq Orpheus TTS.

    CONSTRUCTOR: __init__(self, voice: str = "tara", output_dir: str = DEFAULT_OUTPUT_DIR,
                          model: str = "canopylabs/orpheus-v1-english")
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - voice: str — Orpheus voice name ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
            - output_dir: str — Directory to save generated WAV files
            - model: str — Orpheus TTS model identifier on Groq
        
        INITIALIZES:
            self.voice: str                  = voice
            self.output_dir: Path            = Path(output_dir)
            self.model: str                  = model
            self._groq_client: AsyncGroq     = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
            self._logger: logging.Logger     = logging.getLogger("ClipGenerator")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def generate_all(self, phrases: dict = None) -> dict[str, bool]
        INPUTS:
            - phrases: dict — Override phrases dict (default: BACKCHANNEL_PHRASES)
        OUTPUT:
            - dict[str, bool] — Map of clip_name → success (True/False)
        WHAT IT DOES:
            1. Ensures output directory exists: self.output_dir.mkdir(parents=True, exist_ok=True)
            2. Uses phrases or BACKCHANNEL_PHRASES as default
            3. For each (clip_name, text) in phrases.items():
               a. output_path = self.output_dir / f"{clip_name}.wav"
               b. If output_path already exists:
                  Log: "Skipping {clip_name} (already exists)"
                  Continue (don't regenerate unless --force flag)
               c. success = await self._generate_clip(clip_name, text, output_path)
               d. Record result
               e. Wait 3 seconds between calls (respect Groq rate limits)
            4. Logs summary: "{success_count}/{total} clips generated successfully"
            5. Returns results dict

    async def _generate_clip(self, clip_name: str, text: str, output_path: Path) -> bool
        INPUTS:
            - clip_name: str — Name of the clip (for logging)
            - text: str — Text to synthesize (includes emotion tags)
            - output_path: Path — Where to save the WAV file
        OUTPUT:
            - bool — True if generation succeeded, False on error
        WHAT IT DOES:
            1. Log: "Generating '{clip_name}': '{text}' → {output_path}"
            2. Call Groq Orpheus TTS API:
               response = await self._groq_client.audio.speech.create(
                   model=self.model,
                   input=text,
                   voice=self.voice,
                   response_format="wav"
               )
            3. Write response content to output_path:
               output_path.write_bytes(response.content)
               OR:
               with open(output_path, "wb") as f:
                   async for chunk in response.iter_bytes():
                       f.write(chunk)
            4. Log: "✓ Generated '{clip_name}' ({output_path.stat().st_size} bytes)"
            5. Return True
        
        ERROR HANDLING:
            On any Groq error:
                - Log error: "✗ Failed to generate '{clip_name}': {error}"
                - Return False
            On rate limit (429):
                - Log: "Rate limited — waiting 60s before retry"
                - await asyncio.sleep(60)
                - Retry once

    async def regenerate_all(self, phrases: dict = None) -> dict[str, bool]
        INPUTS:
            - phrases: dict — Override phrases dict
        OUTPUT:
            - dict[str, bool] — Results
        WHAT IT DOES:
            Same as generate_all() but deletes existing files first (force regenerate).

═══════════════════════════════════════════════════════════════════════════════════
STANDALONE SCRIPT FUNCTIONS (for python -m backchannel.generator):
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
FUNCTION: parse_args() -> argparse.Namespace
──────────────────────────────────────────────────────────────────────────────────
    INPUTS: None
    OUTPUT: argparse.Namespace with:
        - voice: str (default from config.yaml or "tara")
        - config: str (default "config.yaml")
        - output_dir: str (default "backchannel/clips/")
        - force: bool (regenerate existing clips)
    WHAT IT DOES:
        Standard argparse setup for the CLI.

──────────────────────────────────────────────────────────────────────────────────
FUNCTION: main() -> None
──────────────────────────────────────────────────────────────────────────────────
    INPUTS: None
    OUTPUT: None
    WHAT IT DOES:
        1. load_dotenv()
        2. Parse args
        3. If --config exists, load voice from config
        4. Create ClipGenerator(voice=voice, output_dir=output_dir)
        5. If --force: run asyncio.run(generator.regenerate_all())
        6. Else: run asyncio.run(generator.generate_all())
        7. Print results summary

    ENTRY POINT:
        if __name__ == "__main__": main()

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - ClipGenerator         (class)
    - BACKCHANNEL_PHRASES   (dict constant)
═══════════════════════════════════════════════════════════════════════════════════

USAGE:
    # Generate with default voice from config.yaml:
    python -m backchannel.generator

    # Generate with specific voice:
    python -m backchannel.generator --voice leo

    # Force regenerate all clips:
    python -m backchannel.generator --force

    # Custom output directory:
    python -m backchannel.generator --output-dir /path/to/clips/

COST:
    One-time cost only. ~15-20 Groq TTS calls. After that, clips are local
    forever and play at zero cost and zero latency.
"""


"""
VOXCORE — backchannel/generator.py
One-time setup script — generates backchannel clips via Orpheus TTS.

Usage:
    python -m backchannel.generator
    python -m backchannel.generator --voice tara --config config.yaml
    python -m backchannel.generator --force
"""


"""
VOXCORE — backchannel/generator.py
One-time setup script — generates backchannel clips via Orpheus TTS.

Usage:
    python -m backchannel.generator
    python -m backchannel.generator --voice diana --config config.yaml
    python -m backchannel.generator --force
"""


# """
# VOXCORE — backchannel/generator.py
# One-time setup script — generates backchannel clips via Orpheus TTS.

# Usage:
#     python -m backchannel.generator
#     python -m backchannel.generator --voice hannah --config config.yaml
#     python -m backchannel.generator --force
# """

# import argparse
# import os
# import logging
# import time
# from pathlib import Path

# import yaml
# from dotenv import load_dotenv
# from groq import Groq

# # ═══════════════════════════════════════════════════════════════════════════════
# # CONSTANTS
# # ═══════════════════════════════════════════════════════════════════════════════

# BACKCHANNEL_PHRASES = {
#     "uh_huh":       "[calm] Uh-huh",
#     "yeah":         "[calm] Yeah",
#     "i_see":        "[calm] I see",
#     "go_on":        "[calm] Go on",
#     "interesting":  "[excited] Interesting",
#     "right":        "[calm] Right",
#     "mm_hmm":       "[calm] Mm-hmm",
#     "okay":         "[calm] Okay",
#     "sure":         "[calm] Sure",
#     "oh_wow":       "[excited] Oh wow",
#     "got_it":       "[calm] Got it",
#     "tell_me_more": "[curious] Tell me more",
#     "really":       "[surprised] Really?",
#     "makes_sense":  "[calm] Makes sense",
#     "understood":   "[calm] Understood",
# }

# DEFAULT_OUTPUT_DIR = "backchannel/clips/"

# # Groq Orpheus TTS — model and the confirmed valid voices
# GROQ_TTS_MODEL = "canopylabs/orpheus-v1-english"
# VALID_VOICES = {"autumn", "diana", "hannah", "austin", "daniel", "troy"}
# DEFAULT_VOICE = "hannah"  # Changed default to a confirmed valid voice

# # Delay between API calls to respect rate limits
# _INTER_CALL_DELAY_S = 3
# # Wait time on rate limit before retry
# _RATE_LIMIT_WAIT_S = 60


# def _resolve_voice(voice: str) -> str:
#     """Validate voice name against Groq Orpheus supported voices."""
#     v = voice.lower().strip()
#     if v in VALID_VOICES:
#         return v
    
#     logging.getLogger("ClipGenerator").warning(
#         "Unknown voice '%s' — falling back to '%s'. Valid Groq voices: %s",
#         voice, DEFAULT_VOICE, sorted(VALID_VOICES),
#     )
#     return DEFAULT_VOICE


# class ClipGenerator:
#     """Generates backchannel WAV clips using Groq Orpheus TTS (sync client)."""

#     def __init__(
#         self,
#         voice: str = DEFAULT_VOICE,
#         output_dir: str = DEFAULT_OUTPUT_DIR,
#         model: str = GROQ_TTS_MODEL,
#     ) -> None:
#         self.voice: str = _resolve_voice(voice)
#         self.output_dir: Path = Path(output_dir)
#         self.model: str = model
#         self._client: Groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
#         self._logger: logging.Logger = logging.getLogger("ClipGenerator")

#     def generate_all(self, phrases: dict | None = None) -> dict[str, bool]:
#         """Generate all backchannel clips, skipping existing ones."""
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         phrases = phrases or BACKCHANNEL_PHRASES
#         results: dict[str, bool] = {}
#         phrase_keys = list(phrases.keys())

#         for i, (clip_name, text) in enumerate(phrases.items()):
#             output_path = self.output_dir / f"{clip_name}.wav"

#             if output_path.exists():
#                 self._logger.info("Skipping %s (already exists)", clip_name)
#                 results[clip_name] = True
#                 continue

#             success = self._generate_clip(clip_name, text, output_path)
#             results[clip_name] = success

#             # Respect rate limits (skip delay after last item)
#             if i < len(phrase_keys) - 1:
#                 time.sleep(_INTER_CALL_DELAY_S)

#         success_count = sum(1 for v in results.values() if v)
#         total = len(results)
#         self._logger.info("%d/%d clips generated successfully", success_count, total)

#         return results

#     def _generate_clip(self, clip_name: str, text: str, output_path: Path) -> bool:
#         """Generate a single backchannel clip via Groq Orpheus TTS."""
#         self._logger.info(
#             "Generating '%s': '%s' (voice=%s) → %s",
#             clip_name, text, self.voice, output_path,
#         )

#         for attempt in range(2):
#             try:
#                 response = self._client.audio.speech.create(
#                     model=self.model,
#                     input=text,
#                     voice=self.voice,
#                     response_format="wav",
#                 )

#                 # Write stream to file manually using iter_bytes()
#                 with open(output_path, "wb") as f:
#                     for chunk in response.iter_bytes():
#                         f.write(chunk)

#                 file_size = output_path.stat().st_size
#                 self._logger.info(
#                     "✓ Generated '%s' (%d bytes)", clip_name, file_size
#                 )
#                 return True

#             except Exception as e:
#                 error_str = str(e)

#                 if "429" in error_str or "rate" in error_str.lower():
#                     if attempt == 0:
#                         self._logger.warning(
#                             "Rate limited — waiting %ds before retry",
#                             _RATE_LIMIT_WAIT_S,
#                         )
#                         time.sleep(_RATE_LIMIT_WAIT_S)
#                         continue
#                     else:
#                         self._logger.error(
#                             "✗ Failed to generate '%s' after retry: %s",
#                             clip_name, e,
#                         )
#                         return False
#                 else:
#                     self._logger.error(
#                         "✗ Failed to generate '%s': %s", clip_name, e
#                     )
#                     return False

#         return False

#     def regenerate_all(self, phrases: dict | None = None) -> dict[str, bool]:
#         """Delete existing clips and regenerate all."""
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         phrases = phrases or BACKCHANNEL_PHRASES

#         for clip_name in phrases:
#             clip_path = self.output_dir / f"{clip_name}.wav"
#             if clip_path.exists():
#                 clip_path.unlink()
#                 self._logger.info("Deleted existing clip: %s", clip_path)

#         return self.generate_all(phrases)


# # ═══════════════════════════════════════════════════════════════════════════════
# # CLI
# # ═══════════════════════════════════════════════════════════════════════════════


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Generate backchannel audio clips via Groq Orpheus TTS"
#     )
#     parser.add_argument(
#         "--voice",
#         type=str,
#         default=None,
#         help=f"Voice name. Valid Groq voices: {sorted(VALID_VOICES)}",
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="config.yaml",
#         help="Path to config.yaml (default: config.yaml)",
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default=DEFAULT_OUTPUT_DIR,
#         help=f"Output directory for clips (default: {DEFAULT_OUTPUT_DIR})",
#     )
#     parser.add_argument(
#         "--force",
#         action="store_true",
#         help="Regenerate existing clips",
#     )
#     return parser.parse_args()


# def main() -> None:
#     load_dotenv()
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
#     )

#     args = parse_args()

#     # Resolve voice: CLI flag > config.yaml > default
#     voice = args.voice
#     if voice is None:
#         config_path = Path(args.config)
#         if config_path.exists():
#             with open(config_path, "r") as f:
#                 config = yaml.safe_load(f) or {}
#             voice = config.get("persona", {}).get("voice", DEFAULT_VOICE)
#         else:
#             voice = DEFAULT_VOICE

#     resolved = _resolve_voice(voice)
#     print(f"Voice: {voice} → {resolved}")
#     print(f"Model: {GROQ_TTS_MODEL}")
#     print(f"Output: {args.output_dir}")
#     print(f"Force: {args.force}")
#     print()

#     generator = ClipGenerator(voice=voice, output_dir=args.output_dir)

#     if args.force:
#         results = generator.regenerate_all()
#     else:
#         results = generator.generate_all()

#     # Print summary
#     print("\n" + "=" * 50)
#     print("GENERATION RESULTS")
#     print("=" * 50)
#     for clip_name, success in results.items():
#         status = "✓" if success else "✗"
#         print(f"  {status} {clip_name}")

#     success_count = sum(1 for v in results.values() if v)
#     print(f"\n{success_count}/{len(results)} clips ready")


# if __name__ == "__main__":
#     main()


"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — backchannel/generator.py                         ║
║      ONE-TIME SETUP SCRIPT — GENERATES BACKCHANNEL CLIPS VIA KOKORO-ONNX       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    A standalone script that generates all backchannel audio clips locally using
    Kokoro-ONNX. This ensures the backchannel voice matches the runtime voice
    perfectly, costs nothing, and hits no rate limits.

    Run once at setup time or whenever you change the persona voice.

    RUN WITH:
        python -m backchannel.generator
        python -m backchannel.generator --voice af_heart
        python -m backchannel.generator --force
    Link: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files
    Voices: af_alloy, af_aoede, af_bella, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky, am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck, bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import time
import os
from pathlib import Path

import yaml
import soundfile as sf
from kokoro_onnx import Kokoro

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Note: Kokoro doesn't use bracketed emotion tags like [calm].
# It relies on punctuation and the voice style itself.
BACKCHANNEL_PHRASES = {
    "uh_huh":       "Uh-huh.",
    "yeah":         "Yeah.",
    "i_see":        "I see.",
    "go_on":        "Go on.",
    "interesting":  "Interesting!",
    "right":        "Right.",
    "mm_hmm":       "Mm-hmm.",
    "okay":         "Okay.",
    "sure":         "Sure.",
    "oh_wow":       "Oh wow!",
    "got_it":       "Got it.",
    "tell_me_more": "Tell me more.",
    "really":       "Really?",
    "makes_sense":  "Makes sense.",
    "understood":   "Understood.",
    "understood-100":   "[calm] Hey how are you doing today?",
    "interesting-100":   "[excited] Interesting!",

}

DEFAULT_OUTPUT_DIR = "clips/heart/"
DEFAULT_MODEL_PATH = "models/kokoro-v1.0.onnx"
DEFAULT_VOICES_PATH = "../models/voices-v1.0.bin"

# Valid Kokoro-ONNX voice IDs (partial list of popular ones)
VALID_VOICES = {
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bm_george"
}
DEFAULT_VOICE = "af_heart"  # Best general-purpose female assistant voice

# Map config persona names to Kokoro voice IDs
_VOICE_MAP = {
    "tara": "af_heart",
    "aria": "af_heart",
    "leah": "af_bella",
    "zoe":  "af_nicole",
    "leo":  "am_adam",
    "dan":  "am_michael",
    "emma": "bf_emma",
}


def _resolve_voice(voice: str) -> str:
    """Map config voice name to a valid Kokoro voice ID."""
    v = voice.lower().strip()
    
    # 1. Direct match
    if v in VALID_VOICES:
        return v
        
    # 2. Map from persona name
    resolved = _VOICE_MAP.get(v)
    if resolved:
        logging.getLogger("ClipGenerator").info(
            "Mapped persona voice '%s' → Kokoro voice '%s'", voice, resolved
        )
        return resolved

    # 3. Fallback
    logging.getLogger("ClipGenerator").warning(
        "Unknown voice '%s' — falling back to '%s'.",
        voice, DEFAULT_VOICE
    )
    return DEFAULT_VOICE


class ClipGenerator:
    """Generates backchannel WAV clips using local Kokoro-ONNX."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        model_path: str = DEFAULT_MODEL_PATH,
        voices_path: str = DEFAULT_VOICES_PATH,
    ) -> None:
        self.voice: str = _resolve_voice(voice)
        self.output_dir: Path = Path(output_dir)
        self._logger: logging.Logger = logging.getLogger("ClipGenerator")
        
        # Verify model files exist
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            raise FileNotFoundError(
                f"Kokoro model files not found!\n"
                f"Expected: {model_path} and {voices_path}\n"
                f"Please download them from HuggingFace (hexgrad/kokoro-onnx)."
            )

        self._logger.info("Loading Kokoro-ONNX model...")
        self._kokoro = Kokoro(model_path, voices_path)
        self._logger.info("Kokoro model loaded.")

    def generate_all(self, phrases: dict | None = None) -> dict[str, bool]:
        """Generate all backchannel clips, skipping existing ones."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        phrases = phrases or BACKCHANNEL_PHRASES
        results: dict[str, bool] = {}

        for clip_name, text in phrases.items():
            output_path = self.output_dir / f"{clip_name}.wav"

            if output_path.exists():
                self._logger.info("Skipping %s (already exists)", clip_name)
                results[clip_name] = True
                continue

            success = self._generate_clip(clip_name, text, output_path)
            results[clip_name] = success

        success_count = sum(1 for v in results.values() if v)
        total = len(results)
        self._logger.info("%d/%d clips generated successfully", success_count, total)

        return results

    def _generate_clip(self, clip_name: str, text: str, output_path: Path) -> bool:
        """Generate a single backchannel clip via Kokoro."""
        self._logger.info(
            "Generating '%s': '%s' (voice=%s) → %s",
            clip_name, text, self.voice, output_path,
        )

        try:
            # Generate audio (returns float32 numpy array, sample rate 24000)
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self.voice,
                speed=1.0,
                lang="en-us",
            )

            # Write to WAV file
            sf.write(output_path, samples, sample_rate)
            
            file_size = output_path.stat().st_size
            self._logger.info(
                "✓ Generated '%s' (%d bytes)", clip_name, file_size
            )
            return True

        except Exception as e:
            self._logger.error(
                "✗ Failed to generate '%s': %s", clip_name, e
            )
            return False

    def regenerate_all(self, phrases: dict | None = None) -> dict[str, bool]:
        """Delete existing clips and regenerate all."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        phrases = phrases or BACKCHANNEL_PHRASES

        for clip_name in phrases:
            clip_path = self.output_dir / f"{clip_name}.wav"
            if clip_path.exists():
                clip_path.unlink()
                self._logger.info("Deleted existing clip: %s", clip_path)

        return self.generate_all(phrases)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate backchannel audio clips via local Kokoro-ONNX TTS"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help=f"Voice ID (e.g. af_heart). Mapped from config if omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for clips (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing clips",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to kokoro-v0_19.onnx (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--voices-path",
        type=str,
        default=DEFAULT_VOICES_PATH,
        help=f"Path to voices.bin (default: {DEFAULT_VOICES_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()

    # Resolve voice: CLI flag > config.yaml > default
    voice = args.voice
    if voice is None:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
            voice = config.get("persona", {}).get("voice", DEFAULT_VOICE)
        else:
            voice = DEFAULT_VOICE

    resolved = _resolve_voice(voice)
    print(f"Voice: {voice} → {resolved}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Force: {args.force}")
    print()

    try:
        generator = ClipGenerator(
            voice=voice,
            output_dir=args.output_dir,
            model_path=args.model_path,
            voices_path=args.voices_path
        )

        if args.force:
            results = generator.regenerate_all()
        else:
            results = generator.generate_all()

        # Print summary
        print("\n" + "=" * 50)
        print("GENERATION RESULTS")
        print("=" * 50)
        for clip_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {clip_name}")

        success_count = sum(1 for v in results.values() if v)
        print(f"\n{success_count}/{len(results)} clips ready")
    
    except FileNotFoundError as e:
        print("\n❌ ERROR: Missing Model Files")
        print(e)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    main()