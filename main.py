"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              VOXCORE — main.py                                  ║
║                         ENTRY POINT — BOOTS ALL ASYNC TASKS                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    This is the single entry point for the entire VoxCore system. It initializes
    all modules, loads configuration, sets up the event bus, and launches all
    concurrent async tasks (mic capture, VAD, STT, LLM, TTS, backchannel,
    safety, interfaces) as a coordinated asyncio task group.

    Run with:  python main.py
    Or:        python main.py --config custom_config.yaml --interface cli

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Core async event loop
import argparse                         # CLI argument parsing
import signal                           # Graceful shutdown on SIGINT/SIGTERM
import sys                              # sys.exit
import logging                          # Structured logging for all modules

from pathlib import Path                # Path operations for config file
from dotenv import load_dotenv          # Load .env file (GROQ_API_KEY, etc.)

import yaml                             # Parse config.yaml

# Internal module imports
from core.session import Session
from core.event_bus import EventBus
from core.turn_manager import TurnManager

from input.mic_stream import MicStream
from input.vad import VADProcessor
from input.stt import STTClient
from input.text_injector import TextInjector
from input.interruption_detector import InterruptionDetector

from backchannel.cue_detector import CueDetector
from backchannel.selector import BackchannelSelector

from brain.llm_client import LLMClient
from brain.prompt_builder import PromptBuilder
from brain.response_parser import ResponseParser
from brain.emotion_tagger import EmotionTagger
from brain.router import BrainRouter

from output.tts_client import TTSClient
from output.audio_player import AudioPlayer
from output.voice_profile import VoiceProfile

from agent.text_out import TextOut
from agent.tool_router import ToolRouter
from agent.slow_llm import SlowLLM

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.compressor import MemoryCompressor

from safety.guard import SafetyGuard

from interfaces.cli import CLIInterface
from interfaces.websocket_server import WebSocketServer
from interfaces.api import APIServer

═══════════════════════════════════════════════════════════════════════════════════
FUNCTIONS & CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
FUNCTION: load_config(config_path: str) -> dict
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - config_path: str — Path to the YAML config file (default: "config.yaml")
    
    OUTPUT:
        - dict — Parsed configuration dictionary with all settings from config.yaml
    
    WHAT IT DOES:
        1. Opens the YAML file at config_path using Path(config_path).read_text()
        2. Parses with yaml.safe_load()
        3. Validates that required top-level keys exist: persona, models, audio,
           backchannel, memory, agent, safety, server
        4. Returns the parsed dict
    
    ERROR HANDLING:
        - If file not found, logs error and sys.exit(1)
        - If YAML parse error, logs error and sys.exit(1)
        - If missing required keys, logs which keys are missing and sys.exit(1)

──────────────────────────────────────────────────────────────────────────────────
FUNCTION: setup_logging(level: str = "INFO") -> None
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - level: str — Logging level string ("DEBUG", "INFO", "WARNING", "ERROR")
    
    OUTPUT:
        - None (configures the root logger)
    
    WHAT IT DOES:
        1. Calls logging.basicConfig() with:
           - level=getattr(logging, level)
           - format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
           - datefmt="%H:%M:%S"
        2. Sets specific loggers for noisy libraries to WARNING:
           - logging.getLogger("httpx").setLevel(logging.WARNING)
           - logging.getLogger("httpcore").setLevel(logging.WARNING)
           - logging.getLogger("chromadb").setLevel(logging.WARNING)

──────────────────────────────────────────────────────────────────────────────────
FUNCTION: parse_args() -> argparse.Namespace
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - None (reads sys.argv)
    
    OUTPUT:
        - argparse.Namespace with attributes:
            - config: str (default "config.yaml")
            - interface: str (choices: "cli", "server", "both"; default "cli")
            - log_level: str (default "INFO")
            - no_mic: bool (flag, for testing without microphone)
    
    WHAT IT DOES:
        1. Creates ArgumentParser with description "VoxCore — Full-Duplex Conversational AI"
        2. Adds arguments:
           --config / -c  : path to config.yaml
           --interface / -i: which interface to launch
           --log-level / -l: logging verbosity
           --no-mic: disable mic for headless/testing mode
        3. Returns parser.parse_args()

──────────────────────────────────────────────────────────────────────────────────
ASYNC FUNCTION: initialize_system(config: dict, args: argparse.Namespace)
                -> dict[str, Any]
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - config: dict — Parsed config.yaml dictionary
        - args: argparse.Namespace — CLI arguments
    
    OUTPUT:
        - dict — A dictionary of all initialized module instances keyed by name:
          {
            "event_bus": EventBus,
            "session": Session,
            "turn_manager": TurnManager,
            "mic_stream": MicStream,
            "vad": VADProcessor,
            "stt": STTClient,
            "text_injector": TextInjector,
            "interruption_detector": InterruptionDetector,
            "cue_detector": CueDetector,
            "backchannel_selector": BackchannelSelector,
            "llm_client": LLMClient,
            "prompt_builder": PromptBuilder,
            "response_parser": ResponseParser,
            "emotion_tagger": EmotionTagger,
            "brain_router": BrainRouter,
            "tts_client": TTSClient,
            "audio_player": AudioPlayer,
            "voice_profile": VoiceProfile,
            "text_out": TextOut,
            "tool_router": ToolRouter,
            "slow_llm": SlowLLM,
            "short_term_memory": ShortTermMemory,
            "long_term_memory": LongTermMemory,
            "compressor": MemoryCompressor,
            "safety_guard": SafetyGuard,
          }
    
    WHAT IT DOES:
        1. Creates EventBus instance (no args)
        2. Creates Session(config=config["persona"], event_bus=event_bus)
        3. Creates VoiceProfile(config=config["persona"])
        4. Creates each module, injecting event_bus, session, and relevant config sections
        5. Initializes LongTermMemory (connects to ChromaDB)
        6. If long_term_memory is enabled, queries for past session context and injects
           via text_injector at startup
        7. Returns dict of all instances
    
    INITIALIZATION ORDER (matters for dependencies):
        EventBus → Session → VoiceProfile → MicStream → VADProcessor → STTClient →
        TextInjector → InterruptionDetector → LLMClient → PromptBuilder →
        ResponseParser → EmotionTagger → BrainRouter → TTSClient → AudioPlayer →
        TextOut → SlowLLM → ToolRouter → ShortTermMemory → LongTermMemory →
        MemoryCompressor → SafetyGuard → CueDetector → BackchannelSelector →
        TurnManager (last, since it orchestrates everything)

──────────────────────────────────────────────────────────────────────────────────
ASYNC FUNCTION: run_pipeline(modules: dict, config: dict, args: argparse.Namespace)
                -> None
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - modules: dict — All initialized module instances from initialize_system()
        - config: dict — Parsed config
        - args: argparse.Namespace — CLI arguments
    
    OUTPUT:
        - None (runs indefinitely until shutdown signal)
    
    WHAT IT DOES:
        1. Creates an asyncio.TaskGroup (Python 3.11+)
        2. Starts the following concurrent tasks inside the group:
           a. modules["mic_stream"].run()          — continuous mic capture loop
           b. modules["vad"].run()                  — continuous VAD processing loop
           c. modules["stt"].run()                  — listens for SPEECH_END, transcribes
           d. modules["turn_manager"].run()         — state machine event loop
           e. modules["response_parser"].run()      — LLM token parsing loop
           f. modules["audio_player"].run()          — audio playback loop
           g. modules["interruption_detector"].run() — interrupt monitoring loop
           h. modules["text_out"].run()              — agent JSON emission loop
           i. modules["tool_router"].run()           — tool execution loop
           j. modules["safety_guard"].run()          — async safety checking loop
           k. If backchannel enabled:
              - modules["cue_detector"].run()
              - modules["backchannel_selector"].run()
           l. If args.interface in ("cli", "both"):
              - modules["cli"].run()
           m. If args.interface in ("server", "both"):
              - Start uvicorn server for WebSocket + API
        3. All tasks run concurrently via asyncio.gather() or TaskGroup
        4. On any task exception, logs and initiates graceful shutdown

──────────────────────────────────────────────────────────────────────────────────
ASYNC FUNCTION: shutdown(modules: dict) -> None
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - modules: dict — All module instances
    
    OUTPUT:
        - None
    
    WHAT IT DOES:
        1. Logs "Shutting down VoxCore..."
        2. Calls modules["session"].save() to persist any unsaved state
        3. If long_term_memory enabled, saves current session to ChromaDB
        4. Closes mic stream (releases sounddevice resources)
        5. Closes any open WebSocket connections
        6. Cancels all running asyncio tasks
        7. Logs "VoxCore shutdown complete."

──────────────────────────────────────────────────────────────────────────────────
FUNCTION: main() -> None
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - None
    
    OUTPUT:
        - None (entry point)
    
    WHAT IT DOES:
        1. Calls load_dotenv() to load .env
        2. Calls parse_args()
        3. Calls setup_logging(args.log_level)
        4. Calls load_config(args.config)
        5. Registers signal handlers for SIGINT and SIGTERM to trigger shutdown()
        6. Runs asyncio.run(async_main(config, args))
    
    This is the function called by:  if __name__ == "__main__": main()

──────────────────────────────────────────────────────────────────────────────────
ASYNC FUNCTION: async_main(config: dict, args: argparse.Namespace) -> None
──────────────────────────────────────────────────────────────────────────────────
    INPUTS:
        - config: dict — Parsed configuration
        - args: argparse.Namespace
    
    OUTPUT:
        - None
    
    WHAT IT DOES:
        1. Calls modules = await initialize_system(config, args)
        2. Wraps run_pipeline(modules, config, args) in try/except
        3. On KeyboardInterrupt or CancelledError, calls await shutdown(modules)
        4. On any unexpected exception, logs traceback and calls shutdown

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
═══════════════════════════════════════════════════════════════════════════════════
    This file exports nothing. It is the entry point only.
    
    Entry: if __name__ == "__main__": main()

═══════════════════════════════════════════════════════════════════════════════════
EXECUTION:
    python main.py
    python main.py --config my_config.yaml --interface server --log-level DEBUG
    python main.py --no-mic --interface cli   # Testing mode without microphone
═══════════════════════════════════════════════════════════════════════════════════
"""
