"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                 VOXCORE v4 — websocket/pipeline_instance.py                     ║
║        ONE ISOLATED PIPELINE PER CONNECTED USER — SHARED HEAVY MODELS          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Creates one complete, isolated pipeline per WebSocket connection.
    Heavy models (Kokoro-ONNX, AriaVoiceFilter, Gate0) come from SharedResources.
    Everything else (event_bus, state machine, VAD, gates, memory) is fresh per user.

    Mirrors the initialization order from main.py's initialize_system().
"""

import asyncio
import logging
import os

# ── Core ─────────────────────────────────────────────────────────────────────
from core.event_bus import EventBus, EventType
from core.session import Session
from core.turn_manager import TurnManager

# ── Input ────────────────────────────────────────────────────────────────────
from input.mic_stream import MicStream
from input.vad import VADProcessor
from input.stt import STTClient
from input.text_injector import TextInjector
from input.interruption_detector import InterruptionDetector
from input.filler_detector import FillerDetector
from input.speaking_monitor import SpeakingMonitor

# ── Brain ────────────────────────────────────────────────────────────────────
from brain.llm_client import LLMClient
from brain.prompt_builder import PromptBuilder
from brain.response_parser import ResponseParser
from brain.emotion_tagger import EmotionTagger
from brain.router import BrainRouter
from brain.interrupt_router import InterruptRouter

# ── Output ───────────────────────────────────────────────────────────────────
from output.tts_client import TTSClient
from output.audio_player import AudioPlayer

# ── Agent ────────────────────────────────────────────────────────────────────
from agent.text_out import TextOut
from agent.tool_router import ToolRouter
from agent.slow_llm import SlowLLM

# ── Memory ───────────────────────────────────────────────────────────────────
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.compressor import MemoryCompressor

# ── Safety ───────────────────────────────────────────────────────────────────
from safety.guard import SafetyGuard

# ── Backchannel ──────────────────────────────────────────────────────────────
from backchannel.cue_detector import CueDetector
from backchannel.selector import BackchannelSelector

# ── WebSocket ────────────────────────────────────────────────────────────────
from websocket.audio_bridge import AudioBridge
from websocket.client_controller import ClientController

logger = logging.getLogger("PipelineInstance")


class PipelineInstance:
    """One isolated pipeline per connected WebSocket user.

    Heavy models (TTS, echo filters) come from SharedResources and are shared.
    Everything else (event_bus, session, VAD, gates, memory) is created fresh
    per connection — complete isolation between users.
    """

    def __init__(self, websocket, user_id: str, shared):
        """
        Args:
            websocket: The WebSocket connection object.
            user_id: Unique identifier for this user session.
            shared: SharedResources instance (heavy models, config).
        """
        self.ws = websocket
        self.user_id = user_id
        self.shared = shared
        self.config = shared.config

        # Per-user plumbing
        self.bridge = AudioBridge(user_id=user_id)
        self.ctrl = ClientController(websocket, user_id=user_id)

        # Per-user isolated objects — created fresh, never shared
        self.event_bus = EventBus()
        self._tasks: list[asyncio.Task] = []
        self._modules: dict = {}

    async def start(self) -> None:
        """Initialize all pipeline modules and start concurrent tasks.

        Mirrors main.py's initialize_system() + run_pipeline() but:
        - Uses WebSocket audio source/sink instead of sounddevice
        - Shares heavy models from SharedResources
        - Everything else is per-user isolated
        """
        config = self.config

        # ══════════════════════════════════════════════════════════════════
        # INITIALIZATION (mirrors main.py initialize_system order)
        # ══════════════════════════════════════════════════════════════════

        event_bus = self.event_bus

        # ── Core ──────────────────────────────────────────────────────────
        session = Session(config=config["persona"], event_bus=event_bus)

        # ── Input ─────────────────────────────────────────────────────────
        mic_stream = MicStream(
            event_bus=event_bus,
            config=config["audio"],
            echo_config=config.get("echo_suppression", {}),
        )
        # V4: Route mic audio from WebSocket queue instead of sounddevice
        mic_stream.set_audio_source(self.bridge.mic_queue)
        # Attach shared echo filters (same as main.py set_v3_filters)
        mic_stream.set_v3_filters(self.shared.aria_filter, self.shared.gate0)

        vad = VADProcessor(
            event_bus=event_bus,
            mic_stream=mic_stream,
            config=config["audio"],
            session=session,
        )

        stt = STTClient(event_bus=event_bus, config=config["models"])

        text_injector = TextInjector(session=session, event_bus=event_bus)

        interruption_detector = InterruptionDetector(
            session=session,
            event_bus=event_bus,
            mic_stream=mic_stream,
            vad=vad,
            config=config,
        )

        filler_detector = FillerDetector(
            config=config.get("speaking_monitor", {})
        )

        # ── Brain ─────────────────────────────────────────────────────────
        llm_client = LLMClient(event_bus=event_bus, config=config["models"])
        prompt_builder = PromptBuilder(config=config["persona"])
        response_parser = ResponseParser(event_bus=event_bus)
        emotion_tagger = EmotionTagger()
        brain_router = BrainRouter(config=config["models"])

        # ── Output ────────────────────────────────────────────────────────
        tts_client = TTSClient(
            event_bus=event_bus,
            voice_profile=self.shared.voice_profile,
            config=config,
        )
        # V4: Use shared Kokoro model instead of loading per-user
        tts_client.set_kokoro(self.shared.kokoro_model)

        audio_player = AudioPlayer(
            session=session, event_bus=event_bus, config=config["audio"]
        )
        # V4: Route TTS audio to WebSocket queue instead of sounddevice
        audio_player.set_audio_sink(self.bridge.speaker_queue)
        # Attach shared echo filter refs (same as main.py set_v3_refs)
        audio_player.set_v3_refs(mic_stream, self.shared.aria_filter, self.shared.gate0)

        # ── Speaking Monitor (Gate 2 + Gate 3) ────────────────────────────
        gate3_done_event = asyncio.Event()
        speaking_monitor = SpeakingMonitor(
            session=session,
            event_bus=event_bus,
            mic_stream=mic_stream,
            filler_detector=filler_detector,
            config=config,
            audio_player=audio_player,
            gate3_done_event=gate3_done_event,
        )

        # ── Agent ─────────────────────────────────────────────────────────
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

        # ── Memory ────────────────────────────────────────────────────────
        short_term_memory = ShortTermMemory(
            event_bus=event_bus,
            max_turns=config.get("memory", {}).get("short_term_turns", 20),
        )
        long_term_memory = LongTermMemory(
            event_bus=event_bus, config=config.get("memory")
        )

        compressor = MemoryCompressor(event_bus=event_bus, llm_client=llm_client)

        # ── Safety ────────────────────────────────────────────────────────
        safety_guard = SafetyGuard(
            event_bus=event_bus,
            api_key=os.environ.get("GROQ_API_KEY", ""),
            enabled=config.get("safety", {}).get("enabled", True),
        )

        # ── Backchannel ───────────────────────────────────────────────────
        cue_detector = CueDetector(
            session=session,
            event_bus=event_bus,
            mic_stream=mic_stream,
            config=config.get("backchannel", {}),
        )
        backchannel_selector = BackchannelSelector(
            session=session,
            event_bus=event_bus,
            config=config,
        )

        # ── Turn Manager (last) ───────────────────────────────────────────
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

        # ── Interrupt Router ──────────────────────────────────────────────
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

        # ── Long-term memory initialization ───────────────────────────────
        try:
            await long_term_memory.initialize()
            mem_cfg = config.get("memory", {})
            if mem_cfg.get("long_term_enabled", False):
                past = await long_term_memory.query(
                    "user name preferences conversation summary", top_k=3
                )
                if past:
                    restored = "\n".join(item["content"] for item in past)
                    session.compressed_summary = restored
                    logger.info(
                        "[%s] Restored %d context items from long-term memory",
                        self.user_id, len(past),
                    )
        except Exception as exc:
            logger.warning("[%s] Long-term memory init failed: %s", self.user_id, exc)

        # ── Wire state changes to client controller ───────────────────────
        event_bus.on(EventType.STATE_CHANGED, self.ctrl.on_state_change)
        event_bus.on(EventType.PLAYBACK_PAUSE, self.ctrl.speaker_pause)
        event_bus.on(EventType.PLAYBACK_RESUME, self.ctrl.speaker_resume)

        # Store modules for stop() cleanup
        self._modules = {
            "session": session,
            "mic_stream": mic_stream,
            "audio_player": audio_player,
            "long_term_memory": long_term_memory,
            "compressor": compressor,
        }

        # ══════════════════════════════════════════════════════════════════
        # LAUNCH CONCURRENT TASKS (mirrors main.py run_pipeline)
        # ══════════════════════════════════════════════════════════════════

        uid = self.user_id

        # Core pipeline tasks
        self._tasks.append(asyncio.create_task(mic_stream.run(), name=f"{uid}_mic"))
        self._tasks.append(asyncio.create_task(vad.run(), name=f"{uid}_vad"))
        self._tasks.append(asyncio.create_task(stt.run(), name=f"{uid}_stt"))
        self._tasks.append(asyncio.create_task(turn_manager.run(), name=f"{uid}_turns"))
        self._tasks.append(asyncio.create_task(tts_client.run(), name=f"{uid}_tts"))
        self._tasks.append(asyncio.create_task(audio_player.run(), name=f"{uid}_player"))
        self._tasks.append(asyncio.create_task(
            interruption_detector.run(), name=f"{uid}_interrupt"
        ))
        self._tasks.append(asyncio.create_task(text_out.run(), name=f"{uid}_text_out"))
        self._tasks.append(asyncio.create_task(tool_router.run(), name=f"{uid}_tools"))
        self._tasks.append(asyncio.create_task(safety_guard.run(), name=f"{uid}_safety"))
        self._tasks.append(asyncio.create_task(short_term_memory.run(), name=f"{uid}_stm"))
        self._tasks.append(asyncio.create_task(compressor.run(), name=f"{uid}_compress"))

        # Speaking monitor (conditional)
        sm_cfg = config.get("speaking_monitor", {})
        if sm_cfg.get("enabled", True):
            self._tasks.append(asyncio.create_task(
                speaking_monitor.run(), name=f"{uid}_monitor"
            ))

        # Backchannel (conditional)
        bc_cfg = config.get("backchannel", {})
        if bc_cfg.get("enabled", True):
            self._tasks.append(asyncio.create_task(
                cue_detector.run(), name=f"{uid}_cue"
            ))
            self._tasks.append(asyncio.create_task(
                backchannel_selector.run(), name=f"{uid}_bc"
            ))

        # TTS sender: pull from bridge speaker queue → send as binary WS frames
        self._tasks.append(asyncio.create_task(
            self._tts_sender(), name=f"{uid}_ws_tts"
        ))

        # Bridge diagnostics
        self._tasks.append(asyncio.create_task(
            self._bridge_monitor(), name=f"{uid}_bridge_mon"
        ))

        logger.info(
            "[%s] Pipeline started with %d tasks",
            self.user_id, len(self._tasks),
        )

    async def _tts_sender(self) -> None:
        """Pull TTS PCM chunks from bridge and send as binary WebSocket frames."""
        while True:
            try:
                chunk = await self.bridge.get_tts_chunk()
                await self.ws.send(chunk)  # raw PCM bytes → client plays immediately
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[%s] TTS sender error: %s", self.user_id, e)
                break

    async def _bridge_monitor(self) -> None:
        """Log bridge queue depths every 5 seconds."""
        while True:
            try:
                await asyncio.sleep(5.0)
                self.bridge.log_depths()
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Clean disconnect: flush TTS, save memory, cancel all tasks."""
        logger.info("[%s] Stopping pipeline...", self.user_id)

        # Flush audio
        try:
            audio_player = self._modules.get("audio_player")
            if audio_player:
                await audio_player.flush()
            await self.ctrl.speaker_flush()
        except Exception:
            pass

        # Cancel all tasks
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Save session to long-term memory
        try:
            ltm = self._modules.get("long_term_memory")
            session = self._modules.get("session")
            if ltm and session:
                await ltm.save_session(session)
                logger.info("[%s] Session saved to long-term memory", self.user_id)
        except Exception as exc:
            logger.warning("[%s] Memory save failed: %s", self.user_id, exc)

        logger.info("[%s] Pipeline stopped.", self.user_id)
