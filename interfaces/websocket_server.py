"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — interfaces/websocket_server.py                    ║
║         WEBSOCKET SERVER — REAL-TIME BIDIRECTIONAL AUDIO/TEXT STREAMING          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    FastAPI WebSocket server for real-time bidirectional communication
    with browser or mobile clients. Handles:
    - Incoming: Raw audio bytes from client mic → fed into VoxCore pipeline
    - Outgoing: TTS audio chunks + transcript text → streamed to client
    - Control: State changes, interrupts, safety flags → pushed to client

    Supports multiple concurrent connections (one active session, others observe).

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging
import asyncio
import json
import struct
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.event_bus import EventBus, EventType
from core.session import Session, TurnState

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

MAX_CONNECTIONS = 5                     # Max concurrent WebSocket clients
PING_INTERVAL = 30                      # Seconds between keepalive pings
AUDIO_SAMPLE_RATE = 16000               # Expected incoming audio sample rate
AUDIO_CHANNELS = 1                      # Mono
AUDIO_CHUNK_MS = 30                     # Expected chunk duration

# WebSocket message types (JSON messages)
MSG_TYPE_AUDIO = "audio"                # Binary audio data
MSG_TYPE_TEXT = "text"                  # Text transcript
MSG_TYPE_STATE = "state"                # State change notification
MSG_TYPE_CONTROL = "control"            # Control commands (interrupt, mute, etc.)
MSG_TYPE_CONFIG = "config"              # Client configuration
MSG_TYPE_ERROR = "error"                # Error notification

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: WebSocketServer
──────────────────────────────────────────────────────────────────────────────────
    FastAPI WebSocket endpoint for real-time audio/text communication.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, session: Session,
                          app: FastAPI = None)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For publishing/subscribing events
            - session: Session — Current session for state info
            - app: FastAPI — Existing FastAPI app to attach to (creates one if None)
        INITIALIZES:
            self._bus = event_bus
            self._session = session
            self._app = app or FastAPI(title="VoxCore WebSocket")
            self._connections: dict[str, WebSocket] = {}  # id → WebSocket
            self._active_connection: Optional[str] = None  # Only one sends audio
            self._logger = logging.getLogger("WebSocketServer")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def setup_routes(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Register WebSocket route: @self._app.websocket("/ws")
               → self._handle_connection(websocket)
            2. Register health endpoint: @self._app.get("/health")
               → returns {"status": "ok", "connections": len(self._connections)}
            3. Register info endpoint: @self._app.get("/ws/info")
               → returns connection count, active state, session info

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None
        INPUTS:
            - host: str — Bind address
            - port: int — Bind port
        OUTPUT: None (runs uvicorn server)
        WHAT IT DOES:
            1. self.setup_routes()
            2. self._subscribe_events()
            3. config = uvicorn.Config(self._app, host=host, port=port,
                   log_level="info", ws_ping_interval=PING_INTERVAL)
            4. server = uvicorn.Server(config)
            5. await server.serve()

    def _subscribe_events(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            Subscribe to EventBus events for broadcasting to clients:
            1. EventType.STATE_CHANGE → self._broadcast_state
            2. EventType.TRANSCRIPT → self._broadcast_transcript
            3. EventType.LLM_SENTENCE → self._broadcast_llm_sentence
            4. EventType.TTS_CHUNK → self._broadcast_audio
            5. EventType.SAFETY_FLAG → self._broadcast_safety
            6. EventType.INTERRUPT → self._broadcast_interrupt

    async def _handle_connection(self, websocket: WebSocket) -> None
        INPUTS:
            - websocket: WebSocket — Incoming connection
        OUTPUT: None
        WHAT IT DOES:
            1. await websocket.accept()
            2. conn_id = generate unique ID (timestamp + counter)
            3. If len(self._connections) >= MAX_CONNECTIONS:
               await websocket.send_json({"type": MSG_TYPE_ERROR,
                   "message": "Max connections reached"})
               await websocket.close()
               return
            4. self._connections[conn_id] = websocket
            5. If self._active_connection is None:
               self._active_connection = conn_id
            6. Send initial state: await websocket.send_json({
                   "type": MSG_TYPE_STATE,
                   "state": self._session.state.value,
                   "is_active": conn_id == self._active_connection
               })
            7. Try:
               While True:
                   message = await websocket.receive()
                   If "bytes" in message:
                       await self._handle_audio(conn_id, message["bytes"])
                   Elif "text" in message:
                       await self._handle_text_message(conn_id, json.loads(message["text"]))
            8. Except WebSocketDisconnect:
               self._connections.pop(conn_id, None)
               If conn_id == self._active_connection:
                   self._active_connection = next(iter(self._connections), None)
               Log f"Client {conn_id} disconnected"

    async def _handle_audio(self, conn_id: str, audio_bytes: bytes) -> None
        INPUTS:
            - conn_id: str — Connection ID (must be active connection)
            - audio_bytes: bytes — Raw PCM audio from client
        OUTPUT: None
        WHAT IT DOES:
            1. If conn_id != self._active_connection: return  # Only active sends
            2. Publish EventType.AUDIO_CHUNK with data={
                   "audio": audio_bytes,
                   "source": "websocket",
                   "conn_id": conn_id
               }

    async def _handle_text_message(self, conn_id: str, message: dict) -> None
        INPUTS:
            - conn_id: str
            - message: dict — Parsed JSON with "type" field
        OUTPUT: None
        WHAT IT DOES:
            1. msg_type = message.get("type")
            2. If msg_type == MSG_TYPE_TEXT:
               Publish EventType.TRANSCRIPT with data={
                   "text": message["text"],
                   "is_final": True,
                   "source": "websocket"
               }
            3. If msg_type == MSG_TYPE_CONTROL:
               Handle control: "interrupt" → publish INTERRUPT,
               "mute" / "unmute" → toggle active connection

    async def _broadcast_state(self, event) -> None
        INPUTS: event
        OUTPUT: None
        WHAT IT DOES:
            Broadcast to all connected clients: {"type": "state", "state": new_state}

    async def _broadcast_transcript(self, event) -> None
        INPUTS: event
        OUTPUT: None
        WHAT IT DOES:
            Broadcast: {"type": "transcript", "text": ..., "is_final": ...}

    async def _broadcast_llm_sentence(self, event) -> None
        INPUTS: event
        OUTPUT: None
        WHAT IT DOES:
            Broadcast: {"type": "ai_text", "text": ...}

    async def _broadcast_audio(self, event) -> None
        INPUTS: event with data["audio"] bytes
        OUTPUT: None
        WHAT IT DOES:
            Send binary audio data to all connected clients via websocket.send_bytes()

    async def _broadcast_safety(self, event) -> None
        INPUTS: event
        OUTPUT: None
        WHAT IT DOES:
            Broadcast: {"type": "safety", "direction": ..., "categories": ...}

    async def _broadcast_interrupt(self, event) -> None
        INPUTS: event
        OUTPUT: None
        WHAT IT DOES:
            Broadcast: {"type": "interrupt"}

    async def _broadcast_json(self, data: dict) -> None
        INPUTS:
            - data: dict — JSON-serializable message
        OUTPUT: None
        WHAT IT DOES:
            1. For conn_id, ws in list(self._connections.items()):
               Try:
                   If ws.client_state == WebSocketState.CONNECTED:
                       await ws.send_json(data)
               Except:
                   self._connections.pop(conn_id, None)

    @property
    def connection_count(self) -> int
        OUTPUT: int
        WHAT IT DOES: return len(self._connections)

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - WebSocketServer   (class)
═══════════════════════════════════════════════════════════════════════════════════

NOTES:
    - Active connection model: only ONE client sends audio at a time,
      but all receive broadcasts (observer pattern)
    - Binary frames are raw PCM audio; text frames are JSON control messages
    - Ping/pong keepalive handled by uvicorn's ws_ping_interval
    - Compatible with browser MediaRecorder + AudioWorklet patterns
    - For production: add authentication middleware (JWT or API key)
"""
