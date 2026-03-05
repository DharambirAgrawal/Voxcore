VoxCore v4 — WebSocket Deployment
Handoff Guide for the AI building this
March 2026  •  main.py stays untouched  •  websocket/ is a fully independent entry point
 1. The One Rule That Governs Everything
main.py is not touched. Not one line. It continues to run VoxCore exactly as it does today — local mic, local speaker, all pipeline modules, startup warmup, everything.

websocket/run.py is an independent entry point. When you run it, it boots its own copy of the pipeline, warms up everything, and starts accepting WebSocket connections. It does not know main.py exists. main.py does not know websocket/ exists.

The shared code is the pipeline modules themselves (mic_stream, audio_player, state machine, interrupt gates, TTS, echo suppression). websocket/run.py imports those directly. It does not copy them. Changes to the pipeline automatically apply to both entry points.

▌ CLEAN SEPARATION: python main.py = local mode, unchanged. python websocket/run.py = WebSocket mode. Two commands, two modes, one shared pipeline codebase.
 2. Directory Structure
voxcore/
├── main.py                        ← UNTOUCHED. local mic + speaker. runs as today.
│
├── websocket/                     ← everything for WebSocket deployment lives here
│   ├── run.py                     ← entry point: python websocket/run.py
│   ├── server.py                  ← WebSocket accept, auth, per-user session manager
│   ├── audio_bridge.py            ← per-user audio queues (replaces local mic + speaker)
│   ├── client_controller.py       ← sends commands (LED, mute, speaker state) to client
│   ├── shared_resources.py        ← heavy models loaded once: TTS, echo, clips
│   ├── pipeline_instance.py       ← one isolated pipeline per connected user
│   └── config_ws.yaml             ← WebSocket-specific config (port, api_key, etc.)
│
├── input/                         ← unchanged (mic_stream, vad, stt, interrupt gates...)
├── brain/                         ← unchanged (llm_client, prompt_builder, router...)
├── output/                        ← unchanged (tts_client, audio_player...)
├── memory/                        ← unchanged
├── core/                          ← unchanged (session, event_bus, turn_manager...)
├── backchannel/                   ← unchanged
├── safety/                        ← unchanged
└── config.yaml                    ← unchanged (used by both main.py and websocket/)

Nothing outside websocket/ is touched. The existing modules get two small additions described in Section 5 — an optional audio source/sink mode. When those are not set, they behave exactly as today.
 3. What Stays on the Server (Everything)
The client is a wire. It has a mic, a speaker, maybe an LED. That is all it does. Every decision happens on the server.

Server does all of this:
•	VAD — Silero, server-side, unchanged
•	STT — Groq Whisper, server-side, unchanged
•	All 3 interrupt gates — Gate0 spectral, Gate1 energy/duration, Gate2 filler, Gate3 allam-2-7b
•	Echo suppression — AriaVoiceFilter (resemblyzer), Gate0EchoCheck (spectral FFT)
•	Echo warmup — synthesize_silent() runs on server at startup, Aria embedding built server-side
•	LLM — llama-3.1-8b, allam-2-7b classifier, all on Groq
•	TTS — Kokoro-ONNX, runs on server, PCM output sent to client
•	State machine — all 6 states, all transitions
•	Memory — ChromaDB, short_term deque, all on server
•	Clip files — breath.wav, got_it.wav, backchannel clips — all generated/stored on server
•	generator.py download — runs on server at startup, not client's concern

Client does only this:
•	Capture mic audio at 16kHz 16-bit mono → send as binary WebSocket frames
•	Receive binary WebSocket frames → play through speaker
•	Receive JSON commands → update LED / mute mic / flush speaker buffer
•	Send keepalive ping every 10s

▌ ESP32 NOTE: AriaVoiceFilter and Gate0 models are loaded and run entirely on the server. The ESP32 never touches them. The ESP32 does not know they exist.
 4. Multi-User Model — Shared Infrastructure, Isolated Conversations
Target: up to 8 simultaneous connections (prototype target: 4–5 across ESP32, app, browser). Heavy models load once at server boot and are shared across all sessions. The things that cannot mix are strictly isolated per user. This keeps RAM low without compromising conversation isolation.

Loaded ONCE at server boot — shared by all users:
•	Kokoro-ONNX TTS engine — same voice (Aria) for everyone, one model instance in RAM
•	AriaVoiceFilter resemblyzer model + Aria's voice embedding — Aria sounds the same regardless of who she's talking to. One embedding covers all sessions.
•	Gate0 EchoCheck spectral model — same TTS engine = same echo frequency pattern for all users
•	Clip files in RAM — breath.wav, got_it.wav, all backchannel clips, loaded once at boot
•	Groq API client — stateless HTTP, naturally shared, just different payloads per user
•	generator.py assets — downloaded once at boot, not repeated per connection

▌ RAM SAVING: resemblyzer ~150MB + Kokoro-ONNX ~300MB. Loading once instead of 8x saves ~3.5GB. This is the whole point of the shared model design.

Isolated per user — these never touch another user:
•	State machine — each user has their own 6-state machine running independently
•	event_bus — instantiated per user. Events from User A's pipeline cannot reach User B's handlers under any circumstances.
•	VAD + Gate1 noise_floor — calibrated to each user's mic and room environment. A noisy ESP32 room doesn't raise thresholds for a quiet browser user.
•	Gate3 classification context — the transcript being classified belongs to one user only
•	STT — each user's audio goes to their own Groq Whisper call with their own context
•	Conversation history — short_term memory deque is per user_id. User A's turns never appear in User B's LLM prompt. Ever.
•	Tool calls + search results — CRITICAL. User A's Tavily web search result never appears in User B's conversation context. Each LLM call is built from that user's isolated history only.
•	Long-term memory (ChromaDB) — queried and written per user_id. Separate namespace per user.
•	AudioBridge queues — per user. User A's mic bytes physically cannot reach User B's VAD.
•	WebSocket connection — per user. Each binary stream is completely separate.
•	Session object + turn counter — per user. Compressor runs independently per session on disconnect.

How the shared echo model stays correct with multiple users:
AriaVoiceFilter is fed Aria's TTS audio to update its EMA — the same audio going to every user's speaker because it's the same TTS model. One embedding works for all users. However when 2+ users are in SPEAKING state simultaneously, feed_aria_audio() and feed_tts_spectrum() are called concurrently. Add a threading.Lock() on both EMA update methods.

▌ CONCURRENCY FIX NEEDED: Add threading.Lock() to AriaVoiceFilter.feed_aria_audio() and Gate0EchoCheck.feed_tts_spectrum(). Two users speaking simultaneously without this lock causes EMA corruption.

SharedResources object — passed into every PipelineInstance:
# websocket/shared_resources.py
class SharedResources:
    """Loaded once at server boot. Passed (read-mostly) to all pipeline instances."""
    def __init__(self, config):
        self.tts_client = TTSClient(config)       # Kokoro-ONNX
        self.aria_filter = AriaVoiceFilter(config) # resemblyzer + EMA (needs lock)
        self.gate0 = Gate0EchoCheck(config)        # spectral EMA (needs lock)
        self.clip_player = ClipPlayer(config)      # all wav files loaded
        self.groq_client = GroqClient(config)      # stateless API client

    async def warmup(self):
        """Run once before accepting any connections."""
        silent = await self.tts_client.synthesize_silent(duration_s=3.0)
        self.aria_filter.feed_aria_audio(silent)   # builds Aria embedding
        self.gate0.feed_tts_spectrum(silent)       # seeds spectral EMA
        print("Shared resources warmed up. Ready for connections.")

Session lifecycle:
SERVER BOOT
  shared = SharedResources(config)
  await shared.warmup()        ← runs ONCE, all users benefit
  VoxCoreServer(shared).start() ← now accepting connections

USER CONNECTS
  PipelineInstance(websocket, user_id, shared)
    isolated: event_bus, state_machine, vad, stt, gates, short_term_memory, session
    injected from shared: tts_client, aria_filter, gate0, clip_player
  → send {"type": "ready"} to client
  → start receiving mic audio frames

USER DISCONNECTS
  PipelineInstance.stop()
    flush TTS, cancel tasks
    compressor.run(session, user_id) → save to ChromaDB
  shared resources untouched, ready for next user
 5. Minimal Changes to Existing Pipeline Modules
Two modules need a small audio source/sink abstraction. Nothing else changes. The abstraction is additive — default behaviour is identical to today.

5.1 mic_stream.py — audio source abstraction
Add one method: set_audio_source(queue). When set, read from this asyncio queue instead of sounddevice. When not set (default), read from sounddevice as today.

# Add to MicStream.__init__:
self._audio_source_queue = None   # None = use sounddevice (default)

# Add method:
def set_audio_source(self, queue: asyncio.Queue):
    """Called by PipelineInstance to route WebSocket audio in."""
    self._audio_source_queue = queue

# Modify _read_chunk() (one if/else, nothing else changes):
async def _read_chunk(self):
    if self._audio_source_queue is not None:
        raw = await self._audio_source_queue.get()
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        # existing sounddevice path — UNCHANGED
        ...

▌ KEY POINT: All downstream processing — denoiser, AriaVoiceFilter, Gate0, VAD — runs on whatever comes out of _read_chunk(). They do not know or care whether audio came from sounddevice or a WebSocket queue.

5.2 audio_player.py — audio sink abstraction
Add one method: set_audio_sink(queue). When set, push PCM chunks to this queue instead of sounddevice. Pause/resume/flush logic is completely unchanged — the TTS buffer in RAM works the same way regardless of sink.

# Add to AudioPlayer.__init__:
self._audio_sink_queue = None   # None = use sounddevice (default)

# Add method:
def set_audio_sink(self, queue: asyncio.Queue):
    """Called by PipelineInstance to route TTS audio out to WebSocket."""
    self._audio_sink_queue = queue

# Modify _play_tts_chunk() (one if/else at the output step):
def _play_tts_chunk(self, chunk: np.ndarray):
    # Echo filter feeds ALWAYS happen first — unchanged
    self.aria_filter.feed_aria_audio(chunk)
    self.gate0.feed_tts_spectrum(chunk)
    # Then output:
    if self._audio_sink_queue is not None:
        pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
        self._audio_sink_queue.put_nowait(pcm_bytes)
    else:
        # existing sounddevice path — UNCHANGED
        sounddevice.play(chunk, samplerate=24000)

▌ ECHO MODELS: feed_aria_audio() and feed_tts_spectrum() always run, before the sink decision. The echo suppression system does not know about WebSocket. It just sees TTS chunks.
 6. The 4 New Files in websocket/
6.1 websocket/audio_bridge.py
Two asyncio queues per user session. Inbound: WebSocket audio frames → mic_stream. Outbound: audio_player TTS chunks → WebSocket. No processing. Pure plumbing.

import asyncio

class AudioBridge:
    def __init__(self):
        self.mic_queue = asyncio.Queue()       # inbound: bytes from client mic
        self.speaker_queue = asyncio.Queue(maxsize=60)  # outbound: bytes to client speaker
        # maxsize=60 ≈ 1.5s of 24kHz audio at 512 samples/chunk
        # if full: old chunk dropped (stale audio is worse than gap)

    def push_mic_audio(self, pcm_bytes: bytes):
        """Called by server.py when binary frame arrives from client."""
        self.mic_queue.put_nowait(pcm_bytes)

    async def get_tts_chunk(self) -> bytes:
        """Called by server.py to get next chunk to send to client."""
        return await self.speaker_queue.get()

    def push_tts_chunk(self, pcm_bytes: bytes):
        """Called indirectly via audio_player's sink queue — same object."""
        try:
            self.speaker_queue.put_nowait(pcm_bytes)
        except asyncio.QueueFull:
            # drop oldest, keep queue moving
            try: self.speaker_queue.get_nowait()
            except: pass
            self.speaker_queue.put_nowait(pcm_bytes)

6.2 websocket/client_controller.py
Maps pipeline state changes to commands sent to the client. The client does whatever it wants with them — LED colour on ESP32, CSS state on browser, nothing on a client that doesn't implement them.

import json

STATE_COMMANDS = {
    "LISTENING":   "led_listening",
    "THINKING":    "led_thinking",
    "SPEAKING":    "led_speaking",
    "PAUSED":      "led_thinking",
    "INTERRUPTED": "led_thinking",
    "SOFT_INJECT": "led_speaking",
}

class ClientController:
    def __init__(self, websocket):
        self.ws = websocket

    async def send(self, action: str, extra: dict = None):
        msg = {"type": "command", "action": action}
        if extra: msg.update(extra)
        await self.ws.send(json.dumps(msg))

    async def on_state_change(self, new_state: str):
        cmd = STATE_COMMANDS.get(new_state)
        if cmd:
            await self.send(cmd)
        # Also send state_change so client knows pipeline state
        await self.ws.send(json.dumps({"type": "state_change", "state": new_state}))

    # Called by turn_manager when TTS pauses
    async def speaker_pause(self):  await self.send("speaker_pause")
    async def speaker_resume(self): await self.send("speaker_resume")
    async def speaker_flush(self):  await self.send("speaker_flush")

    # Called on THINKING entry/exit (optional mic mute to save bandwidth)
    async def mute_mic(self):   await self.send("mute_mic")
    async def unmute_mic(self): await self.send("unmute_mic")

6.3 websocket/pipeline_instance.py
One isolated pipeline per connected user. Heavy models come in via SharedResources (shared, already warmed). Lightweight per-user objects (event_bus, state machine, VAD, gates, memory) are created fresh per connection.

import asyncio
from core.event_bus import EventBus
from core.session import Session
from core.turn_manager import TurnManager
from input.mic_stream import MicStream
from input.vad import VAD
from input.stt import STT
from input.interruption_detector import InterruptionDetector
from input.filler_detector import FillerDetector
from input.speaking_monitor import SpeakingMonitor
from brain.llm_client import LLMClient
from brain.prompt_builder import PromptBuilder
from brain.interrupt_router import InterruptRouter
from output.audio_player import AudioPlayer
from memory.short_term import ShortTermMemory
from memory.compressor import Compressor
from websocket.audio_bridge import AudioBridge
from websocket.client_controller import ClientController

class PipelineInstance:
    def __init__(self, websocket, user_id: str, shared):
        # shared = SharedResources (tts_client, aria_filter, gate0, clip_player)
        self.ws = websocket
        self.user_id = user_id
        self.shared = shared

        # Per-user isolated objects — created fresh, never shared
        self.bridge = AudioBridge()
        self.ctrl = ClientController(websocket)
        self.event_bus = EventBus()          # isolated — User A's events stay with User A
        self.session = Session(user_id=user_id)
        self.short_term = ShortTermMemory()
        self._tasks = []

    async def start(self):
        # mic_stream reads from bridge.mic_queue (WebSocket source)
        self.mic_stream = MicStream(
            config=self.shared.config,
            aria_filter=self.shared.aria_filter,  # shared read-mostly
            gate0=self.shared.gate0               # shared read-mostly
        )
        self.mic_stream.set_audio_source(self.bridge.mic_queue)

        # audio_player pushes to bridge.speaker_queue (WebSocket sink)
        self.audio_player = AudioPlayer(
            tts_client=self.shared.tts_client,    # shared
            aria_filter=self.shared.aria_filter,  # shared (lock inside feed methods)
            gate0=self.shared.gate0,              # shared (lock inside feed methods)
            event_bus=self.event_bus              # ISOLATED — per user
        )
        self.audio_player.set_audio_sink(self.bridge.speaker_queue)

        # Per-user interrupt pipeline — all isolated
        self.vad = VAD(config=self.shared.config, event_bus=self.event_bus)
        self.stt = STT(config=self.shared.config)
        self.interrupt_detector = InterruptionDetector(event_bus=self.event_bus)
        self.filler_detector = FillerDetector(event_bus=self.event_bus)
        self.speaking_monitor = SpeakingMonitor(
            event_bus=self.event_bus,
            audio_player=self.audio_player,
            stt=self.stt
        )
        self.prompt_builder = PromptBuilder(
            short_term=self.short_term,
            session=self.session
        )
        self.llm_client = LLMClient(
            config=self.shared.config,
            prompt_builder=self.prompt_builder,
            event_bus=self.event_bus
        )
        self.turn_manager = TurnManager(
            event_bus=self.event_bus,
            session=self.session
        )

        # Wire state changes to this user's client controller
        self.event_bus.subscribe("STATE_CHANGE", self.ctrl.on_state_change)
        self.event_bus.subscribe("TTS_PAUSE",    self.ctrl.speaker_pause)
        self.event_bus.subscribe("TTS_RESUME",   self.ctrl.speaker_resume)
        self.event_bus.subscribe("TTS_FLUSH",    self.ctrl.speaker_flush)

        # No per-user warmup needed — shared resources already warmed at boot
        # Start pipeline tasks
        self._tasks = [
            asyncio.create_task(self.mic_stream.run(),    name=f"{self.user_id}_mic"),
            asyncio.create_task(self._tts_sender(),       name=f"{self.user_id}_tts"),
            asyncio.create_task(self.turn_manager.run(),  name=f"{self.user_id}_turns"),
        ]

    async def _tts_sender(self):
        """Pull TTS PCM chunks from bridge and send as binary WebSocket frames."""
        while True:
            chunk = await self.bridge.get_tts_chunk()
            await self.ws.send(chunk)  # raw PCM bytes → client plays immediately

    async def stop(self):
        """Clean disconnect: flush, save memory, cancel tasks."""
        try:
            self.audio_player.flush()
            await self.ctrl.speaker_flush()
        except Exception:
            pass
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # Always save session — even 2-turn conversations
        compressor = Compressor(config=self.shared.config)
        await compressor.run(self.session, user_id=self.user_id)

6.4 websocket/server.py
Accepts connections, authenticates, creates a PipelineInstance per user using the shared resources, routes frames, tears down on disconnect. Enforces max 8 concurrent sessions.

import asyncio, json, urllib.parse
import websockets
from websockets.exceptions import ConnectionClosed
from websocket.pipeline_instance import PipelineInstance

class VoxCoreServer:
    def __init__(self, shared_resources):
        self.shared = shared_resources
        self.api_key = shared_resources.config['websocket']['api_key']
        self.max_sessions = shared_resources.config['websocket'].get('max_sessions', 8)
        self.active: dict[str, PipelineInstance] = {}  # user_id → instance

    async def start(self):
        cfg = self.shared.config['websocket']
        print(f"VoxCore WebSocket listening on ws://{cfg['host']}:{cfg['port']}")
        async with websockets.serve(
            self._handler, cfg['host'], cfg['port'],
            process_request=self._authenticate,
            max_size=65536, ping_interval=10, ping_timeout=30
        ):
            print(f"Ready. Max {self.max_sessions} concurrent sessions.")
            await asyncio.Future()

    async def _authenticate(self, path, headers):
        key = (headers.get("Authorization", "").removeprefix("Bearer ")
               or headers.get("X-API-Key", ""))
        if not key:  # check query param fallback (browser)
            params = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
            key = params.get('token', [''])[0]
        if key != self.api_key:
            return (401, [], b"Unauthorized")
        if len(self.active) >= self.max_sessions:
            return (503, [], b"Server at capacity")
        return None

    async def _handler(self, websocket):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(websocket.path).query)
        user_id = params.get('user_id', [f'user_{id(websocket)}'])[0]
        print(f"[+] {user_id} from {websocket.remote_address}")

        instance = PipelineInstance(websocket, user_id, self.shared)
        self.active[user_id] = instance
        try:
            await instance.start()
            await websocket.send(json.dumps({"type": "ready"}))
            async for message in websocket:
                if isinstance(message, bytes):
                    instance.bridge.push_mic_audio(message)   # → mic pipeline
                else:
                    msg = json.loads(message)
                    if msg.get("type") == "ping":
                        await websocket.send(json.dumps({"type":"pong","ts":msg.get("ts")}))
        except ConnectionClosed as e:
            print(f"[-] {user_id} disconnected ({e.code})")
        except Exception as e:
            print(f"[!] {user_id} error: {e}")
        finally:
            await instance.stop()
            del self.active[user_id]
            print(f"[{user_id}] cleaned up. Active sessions: {len(self.active)}")

6.5 websocket/shared_resources.py — NEW
Loaded once at server boot. Passed into every PipelineInstance. This object owns the heavy models.

from output.tts_client import TTSClient
from input.aria_voice_filter import AriaVoiceFilter
from input.gate0_echo_check import Gate0EchoCheck
from backchannel.clip_player import ClipPlayer
import threading

class SharedResources:
    def __init__(self, config):
        self.config = config
        print("Loading shared models...")
        self.tts_client  = TTSClient(config)        # Kokoro-ONNX ~300MB
        self.aria_filter = AriaVoiceFilter(config)  # resemblyzer ~150MB
        self.gate0       = Gate0EchoCheck(config)   # lightweight spectral
        self.clip_player = ClipPlayer(config)       # wav files in RAM
        # Locks for concurrent EMA updates (multiple users in SPEAKING simultaneously)
        # These get added to AriaVoiceFilter and Gate0EchoCheck feed methods
        self._aria_lock = threading.Lock()
        self._gate0_lock = threading.Lock()
        print("Shared models loaded.")

    async def warmup(self):
        """Run once before accepting connections. Warms both echo models."""
        print("Running startup warmup...")
        silent = await self.tts_client.synthesize_silent(duration_s=3.0)
        with self._aria_lock:
            self.aria_filter.feed_aria_audio(silent)
        with self._gate0_lock:
            self.gate0.feed_tts_spectrum(silent)
        print(f"Warmup done. aria_embedding_norm={self.aria_filter.embedding_norm:.3f}")

6.6 websocket/run.py — entry point
Five lines. Load config, boot shared resources, warmup, start server.

import asyncio, yaml
from websocket.shared_resources import SharedResources
from websocket.server import VoxCoreServer

async def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    try:
        with open('websocket/config_ws.yaml') as f:
            config.update(yaml.safe_load(f))
    except FileNotFoundError:
        pass

    shared = SharedResources(config)
    await shared.warmup()           # heavy models warmed ONCE before first connection
    await VoxCoreServer(shared).start()

if __name__ == '__main__':
    asyncio.run(main())

# Run with: python websocket/run.py
 7. Wire Protocol
Client → Server
BINARY FRAME   raw PCM, 16kHz, 16-bit signed mono, 512 samples = 32ms
               send continuously while mic is active
               no headers, no framing, just raw bytes

JSON: { "type": "ping", "ts": 1234567890 }
JSON: { "type": "session_start", "client_id": "esp32_v1" | "mobile_ios" | "browser" }

Server → Client
BINARY FRAME   raw PCM, 24kHz, 16-bit signed mono
               TTS chunks sent as synthesized (streaming, not buffered)
               client plays immediately

JSON: { "type": "ready" }                           ← warmup complete, start sending mic
JSON: { "type": "pong", "ts": ... }
JSON: { "type": "state_change", "state": "SPEAKING" | "LISTENING" | ... }
JSON: { "type": "command", "action": "led_listening" | "led_thinking" |
                                     "led_speaking" | "led_off" |
                                     "speaker_pause" | "speaker_resume" | "speaker_flush" |
                                     "mute_mic" | "unmute_mic" }
JSON: { "type": "error", "code": 4001, "message": "Unauthorized" }

Auth Header Options (client picks one):
Authorization: Bearer <api_key>          ← preferred (all clients)
X-API-Key: <api_key>                     ← alternate for ESP32 simplicity
?token=<api_key>                         ← query param fallback for browsers

WebSocket Close Codes
4001   Unauthorized
4002   Protocol version mismatch
1000   Normal close
1001   Server shutting down
1011   Unexpected server error

▌ CLIENT FLOW: Connect → receive 'ready' → start sending binary mic frames → receive binary TTS + JSON commands. Do not send mic audio before 'ready' — warmup is still running.
 8. How Interrupts Work Over WebSocket
The interrupt pipeline is unchanged. The only difference is where audio comes from and where TTS output goes. The timing requirements are still met over local WiFi.

Flow when user interrupts mid-sentence:
1. Client sends binary PCM frames continuously (user speaks over AI)

2. Server: audio_bridge.push_mic_audio(bytes)
   → mic_stream reads from bridge.mic_queue
   → denoiser → AriaVoiceFilter → Gate0 → InterruptionDetector

3. Gate 1 fires (energy threshold met)
   → PENDING state entered
   → audio_player.pause() — TTS buffer frozen in RAM
   → client_controller.speaker_pause() → client drains audio buffer
   → breath.wav begins playing (through audio_bridge.speaker_queue → client)
   → Gate 3 starts in background (allam-2-7b on Groq)

4. Gate 3 returns decision (e.g. NEW_QUESTION)
   → interrupt_router.handle(NEW_QUESTION)
   → audio_player.flush() — TTS buffer cleared
   → client_controller.speaker_flush() → client clears buffer
   → sure.wav plays
   → primary LLM called with new query

5. New TTS chunks stream through audio_bridge → client plays immediately

▌ LATENCY: WiFi adds ~10-50ms per hop. Gate 1 detection threshold is 150ms (Path A). Total interrupt detection = 150ms + 50ms WiFi = 200ms. User does not notice.

speaker_pause vs speaker_flush — why both:
•	speaker_pause: client stops playing and holds its local buffer. Server TTS is frozen in RAM. Resumable without new LLM call. Used when Gate 3 result is PAUSE or IGNORE.
•	speaker_flush: client discards its local buffer. Server TTS buffer also cleared. Used for INTERRUPTED, NEW_QUESTION, CORRECTION — any case where the AI will say something different.

▌ TIMING: speaker_pause must reach the client before Gate 3 returns. It will — speaker_pause fires at PENDING entry (~3ms), Gate 3 takes 400-600ms. The client is paused long before the decision arrives.
 9. ESP32-S3 Client — What to Implement in C++
The ESP32 is a dumb audio I/O device. These are the only things it needs to do.

Audio capture and send:
// I2S mic at 16kHz, 16-bit, mono
// Read 512 samples = 1024 bytes = 32ms per frame
// Send as binary WebSocket frame immediately
// Drop frames on WiFi congestion — do NOT buffer more than 2 frames
i2s_read(I2S_NUM_0, buffer, 1024, &bytes_read, portMAX_DELAY);
esp_websocket_client_send_bin(client, buffer, 1024, portMAX_DELAY);

Audio receive and play:
// Binary frames arrive at 24kHz PCM
// Feed directly to I2S DAC output at 24kHz
// Buffer 2-3 frames (64-96ms) before starting playback to absorb WiFi jitter
// On "speaker_pause" command: drain ring buffer, stop I2S DMA
// On "speaker_flush" command: clear ring buffer immediately
// On "speaker_resume" command: wait for next chunk, restart DMA

Auth — store key in NVS, not firmware:
// Write once (provisioning):
nvs_handle_t h;
nvs_open("voxcore", NVS_READWRITE, &h);
nvs_set_str(h, "api_key", "your-secret-key");
nvs_commit(h); nvs_close(h);

// Read at startup:
char api_key[64]; size_t len = sizeof(api_key);
nvs_get_str(h, "api_key", api_key, &len);

// Add to WebSocket config:
esp_websocket_client_config_t cfg = {
    .uri = "ws://192.168.1.100:8765/?user_id=esp32_living_room",
    .headers = "X-API-Key: your-secret-key\r\n"
};

LED state commands:
led_listening  → green slow breathing (PWM 0→255→0, period 2s)
led_thinking   → amber fast pulse (PWM 0→255→0, period 0.5s)
led_speaking   → blue solid (PWM 255 constant)
led_off        → all off
// Smooth 200ms transition between states makes it feel alive

Start sending mic only after 'ready' message:
ws.onmessage = [](const char* data) {
    cJSON* msg = cJSON_Parse(data);
    const char* type = cJSON_GetStringValue(cJSON_GetObjectItem(msg, "type"));
    if (strcmp(type, "ready") == 0) {
        start_mic_task();  // begin sending audio frames
    }
    // handle commands...
};
 10. Build Order
Follow this order. Each step is independently testable.

Step 1  websocket/config_ws.yaml
Add websocket block: host, port, api_key. Done separately from config.yaml so main.py config is untouched.
Step 2  websocket/audio_bridge.py
Write from scratch. Two asyncio queues. push_mic_audio(), get_tts_chunk(). Test by pushing fake bytes and reading them back in isolation.
Step 3  input/mic_stream.py — add set_audio_source()
Add the one method. Default is None (sounddevice unchanged). Test: call set_audio_source(queue), push a PCM chunk, verify pipeline sees it.
Step 4  output/audio_player.py — add set_audio_sink()
Add the one method. Default is None (sounddevice unchanged). Echo filter feeds unchanged. Test: set sink, verify chunks appear in queue instead of speaker.
Step 5  websocket/client_controller.py
State → command mapping. on_state_change(), speaker_pause/resume/flush, mute/unmute. Test by instantiating with a mock websocket and verifying JSON output.
Step 6  websocket/pipeline_instance.py
Assemble all pipeline modules. Inject audio bridge. Wire event_bus to client_controller. Implement warmup. Test: start an instance, push silence through mic_queue, verify warmup completes.
Step 7  websocket/server.py
Auth in process_request. Binary → push_mic_audio. JSON → handle_control. Receive loop. Session cleanup on disconnect. Test: connect with wscat, send auth header, send binary frames.
Step 8  websocket/run.py
Five lines. Load config, create server, start. python websocket/run.py should boot and print 'Ready.'
Step 9  Browser test client (50 lines of HTML)
Connect to ws://localhost:8765, send mic via AudioWorklet at 16kHz, play received 24kHz PCM. Test full interrupt flow end-to-end over loopback before touching ESP32.
Step 10  ESP32 firmware
Flash, connect to WiFi, connect to server. Verify LED states, audio round trip, interrupt behaviour. Check server logs: bridge queue depths should stay low (0-5).
 11. Diagnostics — What to Log
Follow the v3 convention: if a number should not be zero, log it.

# server.py
LOG: [+] user_id connected from 192.168.1.42
LOG: [-] user_id disconnected: 1000 Normal
LOG: [!] auth_failed from 192.168.1.99

# pipeline_instance.py
LOG: [user_id] warmup started
LOG: [user_id] warmup complete, aria_embedding_norm=0.94 gate0_ema_populated=True
WARN if aria_embedding_norm < 0.5: embedding may be too weak — echo suppression at risk
WARN if gate0_ema_populated=False: Gate0 spectral EMA empty — check _play_tts_chunk()

# audio_bridge.py (log every 5s during active session)
LOG: mic_queue_depth=2  speaker_queue_depth=3
WARN if mic_queue_depth > 20: pipeline stalled, mic audio backing up
WARN if speaker_queue_depth > 40: client not consuming TTS, network slow?

# client_controller.py
LOG: CMD → led_speaking (state=SPEAKING)
LOG: CMD → speaker_pause

▌ KEY DIAGNOSTIC: speaker_queue_depth=0 during SPEAKING means TTS is reaching client cleanly. mic_queue_depth=0 means client audio is being processed as fast as it arrives. Both should hover near zero.
 12. What Not to Build in v4
▌ DO NOT BUILD: These are explicit non-goals. Adding them will delay v4 and complicate debugging.

•	TLS / WSS — not needed for local WiFi. Add in v5 for public deployment.
•	Opus audio compression — raw PCM is debuggable. Opus saves bandwidth but adds decoding complexity on ESP32. Defer.
•	Session tokens / reconnect state — reconnect = new session. Clean and simple.
•	Client-side VAD — all decision logic stays on server. ESP32 sends raw audio, period.
•	REST API alongside WebSocket — everything through WebSocket.
•	main.py awareness of websocket/ — these are two separate modes. No shared flags, no config merging at main.py level.
•	Memory system (ChromaDB, fact_store, extractor) — that is the other half of v4. Build WebSocket first, test it fully, then add memory on top.
 13. One-Page Summary
What v4 adds:
•	websocket/ directory — independent entry point: python websocket/run.py
•	shared_resources.py — loads TTS, echo models, clips ONCE at boot. Shared by all users.
•	pipeline_instance.py — one isolated pipeline per connected user (event_bus, state, VAD, gates, memory all per-user)
•	audio_bridge.py — two asyncio queues per user replacing local mic and speaker
•	client_controller.py — sends LED/speaker/mute commands as JSON to each client
•	server.py — auth, session management, frame routing, max 8 concurrent users

What changes in existing modules:
•	mic_stream.py — one method added: set_audio_source(queue). No other changes.
•	audio_player.py — one method added: set_audio_sink(queue). No other changes.
•	Everything else — literally unchanged.

What main.py does:
Same as today. Zero changes. python main.py works exactly as before.

The client (ESP32 / mobile / browser):
•	Sends: raw PCM 16kHz binary frames + occasional JSON control
•	Receives: raw PCM 24kHz binary frames + JSON commands
•	Knows nothing about VAD, interrupts, echo models, state machine

Start here:
audio_bridge.py → mic_stream → audio_player → pipeline_instance.py → server.py → run.py → browser test → ESP32
