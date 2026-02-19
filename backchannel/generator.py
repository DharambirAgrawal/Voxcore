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
