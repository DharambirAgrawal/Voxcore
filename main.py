"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              VOXCORE — main.py                                  ║
║                         ENTRY POINT — BOOTS ALL ASYNC TASKS                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import argparse
import os
import signal
import sys
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

# ── Internal module imports ──────────────────────────────────────────────────
from core.session import Session
from core.event_bus import EventBus
from core.turn_manager import TurnManager

from input.mic_stream import MicStream
from input.vad import VADProcessor
from input.stt import STTClient
from input.text_injector import TextInjector
from input.interruption_detector import InterruptionDetector
from input.filler_detector import FillerDetector
from input.speaking_monitor import SpeakingMonitor
# Echo suppression layers
from input.aria_voice_filter import AriaVoiceFilter
from input.gate0_echo_check import Gate0EchoCheck

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
# Interrupt router
from brain.interrupt_router import InterruptRouter

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.compressor import MemoryCompressor
from memory.fact_store import FactStore
from memory.procedural import ProceduralStore
from memory.session_cache import SessionCache
from memory.background_llm import BackgroundLLM
from memory.loader import MemoryLoader
from memory.retriever import MemoryRetriever

from safety.guard import SafetyGuard

from interfaces.cli import CLIInterface
from interfaces.websocket_server import WebSocketServer
from interfaces.api import APIServer

logger = logging.getLogger("main")

REQUIRED_CONFIG_KEYS = (
    "persona", "models", "audio", "backchannel", "memory", "agent", "safety", "server"
)


# ═════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & LOGGING
# ═════════════════════════════════════════════════════════════════════════════════


def load_config(config_path: str) -> dict:
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logging.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        raw = path.read_text(encoding="utf-8")
        config = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logging.error("Failed to parse YAML config: %s", exc)
        sys.exit(1)

    if config is None:
        logging.error("Config file is empty: %s", config_path)
        sys.exit(1)

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        logging.error("Missing required config keys: %s", ", ".join(missing))
        sys.exit(1)

    return config


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for all modules."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VoxCore — Full-Duplex Conversational AI"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--interface", "-i",
        type=str,
        choices=["cli", "server", "both"],
        default="cli",
        help="Which interface to launch",
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--no-mic",
        action="store_true",
        help="Disable mic for headless/testing mode",
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════════
# SYSTEM INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════════


async def initialize_system(
    config: dict, args: argparse.Namespace
) -> dict[str, Any]:
    """Instantiate every module in dependency order and return a dict of them."""

    logger.info("Initializing VoxCore system...")

    # ── Core ─────────────────────────────────────────────────────────────────
    event_bus = EventBus()
    session = Session(config=config["persona"], event_bus=event_bus)
    voice_profile = VoiceProfile(config=config)

    # ── Input ────────────────────────────────────────────────────────────────
    mic_stream = MicStream(
        event_bus=event_bus,
        config=config["audio"],
        echo_config=config.get("echo_suppression", {}),
    )
    vad = VADProcessor(
        event_bus=event_bus, mic_stream=mic_stream, config=config["audio"],
        session=session,
    )
    stt = STTClient(event_bus=event_bus, config=config["models"])
    text_injector = TextInjector(session=session, event_bus=event_bus)
    interruption_detector = InterruptionDetector(
        session=session,
        event_bus=event_bus,
        mic_stream=mic_stream,
        vad=vad,
        config=config,  # pass full config so it can read speaking_monitor section
    )

    # Filler detector (Gate 2) and Speaking Monitor (Gate 3)
    filler_detector = FillerDetector(
        config=config.get("speaking_monitor", {})
    )
    # NOTE: SpeakingMonitor created below after audio_player + gate3_done_event

    # ── Brain ────────────────────────────────────────────────────────────────
    llm_client = LLMClient(event_bus=event_bus, config=config["models"])
    prompt_builder = PromptBuilder(config=config["persona"])
    response_parser = ResponseParser(event_bus=event_bus)
    emotion_tagger = EmotionTagger()
    brain_router = BrainRouter(config=config["models"])

    # ── Output ───────────────────────────────────────────────────────────────
    tts_client = TTSClient(
        event_bus=event_bus,
        voice_profile=voice_profile,
        config=config,  # pass full config so it can read backchannel.emotional_clips
    )
    audio_player = AudioPlayer(
        session=session, event_bus=event_bus, config=config["audio"]
    )


    # ── Echo suppression layers ───────────────────────────────────────
    echo_cfg = config.get("echo_suppression", {})
    aria_filter = AriaVoiceFilter(
        similarity_threshold=echo_cfg.get("aria_similarity_threshold", 0.75),
    )
    gate0 = Gate0EchoCheck(
        spectral_threshold=echo_cfg.get("gate0_spectral_threshold", 0.85),
        temporal_gate_ms=echo_cfg.get("gate0_temporal_gate_ms", 80),
        ema_alpha=echo_cfg.get("gate0_ema_alpha", 0.1),
    )
    gate3_done_event = asyncio.Event()

    # Wire v3 filters into mic_stream and audio_player
    mic_stream.set_v3_filters(aria_filter, gate0)
    audio_player.set_v3_refs(mic_stream, aria_filter, gate0)

    # Create SpeakingMonitor with audio_player and gate3_done_event
    speaking_monitor = SpeakingMonitor(
        session=session,
        event_bus=event_bus,
        mic_stream=mic_stream,
        filler_detector=filler_detector,
        config=config,
        audio_player=audio_player,
        gate3_done_event=gate3_done_event,
    )

    # ── Agent ────────────────────────────────────────────────────────────────
    text_out = TextOut(
        session=session, event_bus=event_bus, config=config.get("agent", {})
    )
    slow_llm = SlowLLM(
        llm_client=llm_client,
        brain_router=brain_router,
        text_injector=text_injector,
        event_bus=event_bus,
    )
    tool_router = ToolRouter(
        event_bus=event_bus,
        text_injector=text_injector,
        config=config.get("agent", {}),
    )

    # ── Memory ───────────────────────────────────────────────────────────────
    mem_cfg = config.get("memory", {})
    short_term_memory = ShortTermMemory(
        event_bus=event_bus,
        max_turns=mem_cfg.get("short_term_turns", 20),
    )
    long_term_memory = LongTermMemory(
        event_bus=event_bus, config=mem_cfg
    )

    compressor = MemoryCompressor(
        event_bus=event_bus,
        short_term=short_term_memory,
        long_term=long_term_memory,
        config=config,
    )

    # New memory tiers
    fact_store = FactStore(config=config)
    procedural = ProceduralStore(config=config)
    session_cache = SessionCache()
    background_llm = BackgroundLLM(config=config)
    loader = MemoryLoader(
        fact_store=fact_store,
        procedural=procedural,
        long_term=long_term_memory,
        config=mem_cfg,
    )
    retriever = MemoryRetriever(
        short_term=short_term_memory,
        long_term=long_term_memory,
        fact_store=fact_store,
    )

    # Register new tools that need memory components
    tool_router.set_memory_components(
        session_cache=session_cache,
        retriever=retriever,
        config=config,
    )

    # ── Safety ───────────────────────────────────────────────────────────────
    safety_guard = SafetyGuard(
        event_bus=event_bus,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        enabled=config.get("safety", {}).get("enabled", True),
    )

    # ── Backchannel ──────────────────────────────────────────────────────────
    cue_detector = CueDetector(
        session=session,
        event_bus=event_bus,
        mic_stream=mic_stream,
        config=config.get("backchannel", {}),
    )
    backchannel_selector = BackchannelSelector(
        session=session,
        event_bus=event_bus,
        config=config,  # pass full config so it can read backchannel section
    )

    # ── Turn Manager (last — orchestrates everything) ────────────────────
    turn_manager = TurnManager(
        session=session,
        event_bus=event_bus,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        response_parser=response_parser,
        brain_router=brain_router,
        safety_guard=safety_guard,
        config=config,
    )

    # ── Interrupt Router ───────────────────────────────────────────
    interrupt_router = InterruptRouter(
        config=config,
        audio_player=audio_player,
        mic_stream=mic_stream,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        event_bus=event_bus,
        session=session,
        tts_client=tts_client,
        gate3_done_event=gate3_done_event,
    )
    turn_manager.set_interrupt_router(interrupt_router)

    # ── Long-term memory: always initialize for MEMORY_COMPRESSED subscription ──
    try:
        await long_term_memory.initialize()

        # Initialize SQLite stores
        await fact_store.initialize()
        await procedural.initialize()

        mem_cfg = config.get("memory", {})
        if mem_cfg.get("long_term_enabled", False):
            # Build session-start memory block from all stores
            memory_block = await loader.build_session_block()
            if memory_block:
                prompt_builder.set_memory_block(memory_block)
                logger.info(
                    "Session memory block set (%d chars, ~%d tokens)",
                    len(memory_block), len(memory_block) // 4,
                )
    except Exception as exc:
        logger.warning("Could not initialize memory system: %s", exc)

    # Wire memory components to TurnManager for background extraction
    turn_manager.set_memory_components(
        background_llm=background_llm,
        short_term=short_term_memory,
        fact_store=fact_store,
        procedural=procedural,
        long_term=long_term_memory,
    )

    logger.info("All modules initialized.")

    return {
        "event_bus": event_bus,
        "session": session,
        "turn_manager": turn_manager,
        "mic_stream": mic_stream,
        "vad": vad,
        "stt": stt,
        "text_injector": text_injector,
        "interruption_detector": interruption_detector,
        "filler_detector": filler_detector,
        "speaking_monitor": speaking_monitor,
        "cue_detector": cue_detector,
        "backchannel_selector": backchannel_selector,
        "llm_client": llm_client,
        "prompt_builder": prompt_builder,
        "response_parser": response_parser,
        "emotion_tagger": emotion_tagger,
        "brain_router": brain_router,
        "tts_client": tts_client,
        "audio_player": audio_player,
        "voice_profile": voice_profile,
        "text_out": text_out,
        "tool_router": tool_router,
        "slow_llm": slow_llm,
        "short_term_memory": short_term_memory,
        "long_term_memory": long_term_memory,
        "compressor": compressor,
        "fact_store": fact_store,
        "procedural": procedural,
        "session_cache": session_cache,
        "background_llm": background_llm,
        "loader": loader,
        "retriever": retriever,
        "safety_guard": safety_guard,
        # New modules
        "aria_filter": aria_filter,
        "gate0": gate0,
        "gate3_done_event": gate3_done_event,
        "interrupt_router": interrupt_router,
    }


# ═════════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═════════════════════════════════════════════════════════════════════════════════


async def run_pipeline(
    modules: dict[str, Any],
    config: dict,
    args: argparse.Namespace,
) -> None:
    """Launch all concurrent async tasks in a supervised task group."""

    tasks: list[asyncio.Task] = []

    # ── Core pipeline tasks ──────────────────────────────────────────────────
    tasks.append(asyncio.create_task(modules["mic_stream"].run(), name="mic_stream"))
    tasks.append(asyncio.create_task(modules["vad"].run(), name="vad"))
    tasks.append(asyncio.create_task(modules["stt"].run(), name="stt"))
    tasks.append(asyncio.create_task(modules["turn_manager"].run(), name="turn_manager"))
    tasks.append(asyncio.create_task(modules["tts_client"].run(), name="tts_client"))
    tasks.append(asyncio.create_task(modules["audio_player"].run(), name="audio_player"))
    tasks.append(asyncio.create_task(
        modules["interruption_detector"].run(), name="interruption_detector"
    ))
    tasks.append(asyncio.create_task(modules["text_out"].run(), name="text_out"))
    tasks.append(asyncio.create_task(modules["tool_router"].run(), name="tool_router"))
    tasks.append(asyncio.create_task(modules["safety_guard"].run(), name="safety_guard"))
    tasks.append(asyncio.create_task(modules["short_term_memory"].run(), name="short_term_memory"))
    tasks.append(asyncio.create_task(modules["compressor"].run(), name="compressor"))

    # Speaking Monitor (Gate 2 + Gate 3 interrupt classification)
    sm_cfg = config.get("speaking_monitor", {})
    if sm_cfg.get("enabled", True):
        tasks.append(asyncio.create_task(
            modules["speaking_monitor"].run(), name="speaking_monitor"
        ))

    # ── Backchannel (conditional) ────────────────────────────────────────────
    bc_cfg = config.get("backchannel", {})
    if bc_cfg.get("enabled", True):
        tasks.append(asyncio.create_task(
            modules["cue_detector"].run(), name="cue_detector"
        ))
        tasks.append(asyncio.create_task(
            modules["backchannel_selector"].run(), name="backchannel_selector"
        ))

    # ── Interfaces ───────────────────────────────────────────────────────────
    if args.interface in ("cli", "both"):
        cli = CLIInterface(
            event_bus=modules["event_bus"],
            session=modules["session"],
            text_mode=args.no_mic,
            verbose=args.log_level == "DEBUG",
        )
        modules["cli"] = cli
        tasks.append(asyncio.create_task(cli.run(), name="cli"))

    if args.interface in ("server", "both"):
        from fastapi import FastAPI

        app = FastAPI(title="VoxCore", version="0.1.0")

        ws_server = WebSocketServer(
            event_bus=modules["event_bus"],
            session=modules["session"],
            mic_stream=modules["mic_stream"],
            app=app,
        )
        api_server = APIServer(
            event_bus=modules["event_bus"],
            session=modules["session"],
            short_term=modules["short_term_memory"],
            long_term=modules["long_term_memory"],
            safety=modules["safety_guard"],
            voice_profile=modules["voice_profile"],
            app=app,
        )
        modules["ws_server"] = ws_server
        modules["api_server"] = api_server

        # Setup API routes (WebSocket routes set up inside start)
        api_server.setup_routes()
        ws_server.setup_routes()

        srv_cfg = config.get("server", {})
        host = srv_cfg.get("host", "0.0.0.0")
        port = srv_cfg.get("port", 8765)

        import uvicorn

        uvi_config = uvicorn.Config(
            app, host=host, port=port, log_level="info"
        )
        server = uvicorn.Server(uvi_config)
        tasks.append(asyncio.create_task(server.serve(), name="uvicorn"))

    logger.info("Pipeline running with %d tasks.", len(tasks))

    # ── Wait for first failure or forever ────────────────────────────────────
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Check for exceptions
    for task in done:
        if task.exception():
            logger.error(
                "Task '%s' raised: %s", task.get_name(), task.exception()
            )

    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ═════════════════════════════════════════════════════════════════════════════════
# SHUTDOWN
# ═════════════════════════════════════════════════════════════════════════════════


async def shutdown(modules: dict[str, Any]) -> None:
    """Gracefully shut down all modules and release resources."""
    logger.info("Shutting down VoxCore...")

    # Persist session state
    try:
        if hasattr(modules.get("session"), "save"):
            await modules["session"].save() if asyncio.iscoroutinefunction(
                getattr(modules["session"], "save", None)
            ) else modules["session"].save()
    except Exception as exc:
        logger.warning("Session save error: %s", exc)

    # Save to long-term memory
    try:
        ltm = modules.get("long_term_memory")
        session = modules.get("session")
        if ltm and session:
            await ltm.save_session(session)
    except Exception as exc:
        logger.warning("Long-term memory save error: %s", exc)

    # Flush remaining turns via compressor
    try:
        compressor = modules.get("compressor")
        if compressor and hasattr(compressor, "compress_session_end"):
            await compressor.compress_session_end()
    except Exception as exc:
        logger.warning("Session-end compression error: %s", exc)

    # Clear session cache (RAM only, no persistence needed)
    try:
        sc = modules.get("session_cache")
        if sc:
            sc.clear()
    except Exception as exc:
        logger.warning("Session cache clear error: %s", exc)

    # Close mic stream
    try:
        mic = modules.get("mic_stream")
        if mic and hasattr(mic, "close"):
            mic.close() if not asyncio.iscoroutinefunction(mic.close) else await mic.close()
    except Exception as exc:
        logger.warning("Mic close error: %s", exc)

    # Close WebSocket connections
    try:
        ws = modules.get("ws_server")
        if ws and hasattr(ws, "_connections"):
            for conn_id, websocket in list(ws._connections.items()):
                try:
                    await websocket.close()
                except Exception:
                    pass
            ws._connections.clear()
    except Exception as exc:
        logger.warning("WebSocket close error: %s", exc)

    # Cancel all remaining tasks
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()

    logger.info("VoxCore shutdown complete.")


# ═════════════════════════════════════════════════════════════════════════════════
# ASYNC MAIN & ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════════


async def async_main(config: dict, args: argparse.Namespace) -> None:
    """Top-level async orchestrator: init → warmup → run → shutdown."""
    modules: dict[str, Any] = {}
    try:
        modules = await initialize_system(config, args)
        # Startup warmup — pre-warm echo filters before conversation
        await startup_warmup(modules, config)
        await run_pipeline(modules, config, args)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Received shutdown signal.")
    except Exception:
        logger.exception("Unexpected error in pipeline")
    finally:
        if modules:
            await shutdown(modules)


async def startup_warmup(modules: dict[str, Any], config: dict) -> None:
    """V3: Pre-warm echo suppression filters before conversation starts.

    Synthesizes ~3s of Aria's voice silently via Kokoro, then feeds the
    audio to AriaVoiceFilter and Gate0EchoCheck. This ensures both filters
    are fully calibrated before the first word is spoken.
    """
    from core.event_bus import EventType

    echo_cfg = config.get("echo_suppression", {})
    warmup_text = echo_cfg.get(
        "warmup_text",
        "Hello, how are you doing today? I hope you're having a wonderful day.",
    )

    tts_client = modules["tts_client"]
    aria_filter = modules.get("aria_filter")
    gate0 = modules.get("gate0")
    event_bus = modules["event_bus"]

    if aria_filter is None and gate0 is None:
        logger.info("[Warmup] No v3 filters to warm up")
        return

    logger.info("[Warmup] Synthesizing Aria voice for echo filter calibration...")
    try:
        # Synthesize silently — returns float32 at 16kHz
        audio = await tts_client.synthesize_silent(warmup_text)

        # Feed to AriaVoiceFilter (builds speaker embedding)
        if aria_filter is not None:
            chunk_size = 3200  # 200ms chunks at 16kHz
            for i in range(0, len(audio), chunk_size):
                aria_filter.feed_aria_audio(audio[i:i + chunk_size])
            logger.info(
                "[Warmup] AriaVoiceFilter ready=%s (%d samples fed)",
                aria_filter.is_ready, len(audio),
            )

        # Feed to Gate0EchoCheck (builds spectral fingerprint)
        if gate0 is not None:
            chunk_size = 480  # 30ms chunks at 16kHz
            for i in range(0, len(audio), chunk_size):
                gate0.feed_tts_spectrum(audio[i:i + chunk_size])
            logger.info(
                "[Warmup] Gate0EchoCheck ready=%s", gate0.is_ready,
            )

        await event_bus.publish(EventType.WARMUP_COMPLETE, {}, source="main")
        logger.info("[Warmup] Echo protection fully ready")

    except Exception as e:
        logger.warning("[Warmup] Failed (continuing without warmup): %s", e)


def main() -> None:
    """Entry point for VoxCore."""
    load_dotenv()
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    logger.info("VoxCore starting...")
    logger.info("Interface: %s | Mic: %s | Log level: %s",
                args.interface, "disabled" if args.no_mic else "enabled", args.log_level)

    # Register signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()

    def _signal_handler() -> None:
        logger.info("Signal received, shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: _signal_handler())

    try:
        loop.run_until_complete(async_main(config, args))
    finally:
        loop.close()


if __name__ == "__main__":
    main()