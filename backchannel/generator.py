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

# V2: Emotional reaction clips — used inline during AI speech when the LLM
# emits tags like [laughs], [chuckles], [light_laugh], [sighs].
EMOTIONAL_CLIPS = {
    "laughs":       "Ha, that's great!",
    "chuckles":     "Heh.",
    "light_laugh":  "Hm heh.",
    "sighs":        "Mm.",
}

# V3: Pre-pause clips — played instantly when entering PENDING state
# These give the user audio feedback while Gate 3 classifies.
PRE_PAUSE_CLIPS = {
    "breath":   "Hmm.",        # Soft breath-like sound
    "mm":       "Mm.",         # Minimal acknowledgment
    "mm_hmm":   "Mm-hmm.",    # Slightly longer (reuses backchannel)
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
        """Generate all backchannel clips and emotional clips, skipping existing ones."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        phrases = phrases or BACKCHANNEL_PHRASES
        # V2: Merge emotional clips into the generation pass
        # V3: Add pre-pause clips for PENDING state filler
        all_phrases = {**phrases, **EMOTIONAL_CLIPS, **PRE_PAUSE_CLIPS}
        results: dict[str, bool] = {}

        for clip_name, text in all_phrases.items():
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
        """Delete existing clips and regenerate all (including emotional clips)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        phrases = phrases or BACKCHANNEL_PHRASES
        # V2: Merge emotional clips
        # V3: Add pre-pause clips
        all_phrases = {**phrases, **EMOTIONAL_CLIPS, **PRE_PAUSE_CLIPS}

        for clip_name in all_phrases:
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