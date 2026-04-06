"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — interfaces/websocket_server.py                    ║
║         WEBSOCKET SERVER — REAL-TIME BIDIRECTIONAL AUDIO/TEXT STREAMING          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
""
import logging
import json
import os
import sys
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import uvicorn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.event_bus import EventBus, EventType
from core.session import Session

# ═════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════════

MAX_CONNECTIONS = 5
PING_INTERVAL = 30
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_MS = 30

# WebSocket message types
MSG_TYPE_AUDIO = "audio"
MSG_TYPE_TEXT = "text"
MSG_TYPE_STATE = "state"
MSG_TYPE_CONTROL = "control"
MSG_TYPE_CONFIG = "config"
MSG_TYPE_ERROR = "error"


# ═════════════════════════════════════════════════════════════════════════════════
# WEBSOCKET SERVER
# ═════════════════════════════════════════════════════════════════════════════════


class WebSocketServer:
    """FastAPI WebSocket endpoint for real-time audio/text communication."""

    _conn_counter: int = 0

    def __init__(
        self,
        event_bus: EventBus,
        session: Session,
        mic_stream=None,
        app: Optional[FastAPI] = None,
    ) -> None:
        self._bus = event_bus
        self._session = session
        self._mic_stream = mic_stream  # For injecting WebSocket audio into pipeline
        self._app = app or FastAPI(title="VoxCore WebSocket")
        self._connections: dict[str, WebSocket] = {}
        self._active_connection: Optional[str] = None
        self._logger = logging.getLogger("WebSocketServer")

    # ── property ─────────────────────────────────────────────────────────────

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    # ── route setup ──────────────────────────────────────────────────────────

    def setup_routes(self) -> None:
        @self._app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket) -> None:
            await self._handle_connection(websocket)

        @self._app.get("/health")
        async def health() -> dict:
            return {"status": "ok", "connections": len(self._connections)}

        @self._app.get("/ws/info")
        async def ws_info() -> dict:
            state = (
                self._session.state.value
                if hasattr(self._session, "state") and hasattr(self._session.state, "value")
                else str(getattr(self._session, "state", "unknown"))
            )
            return {
                "connections": len(self._connections),
                "active_connection": self._active_connection,
                "state": state,
            }

    # ── event subscriptions ──────────────────────────────────────────────────

    def _subscribe_events(self) -> None:
        self._bus.subscribe(EventType.STATE_CHANGED, self._broadcast_state)
        self._bus.subscribe(EventType.TRANSCRIPT_READY, self._broadcast_transcript)
        self._bus.subscribe(EventType.LLM_STREAM_DONE, self._broadcast_llm_sentence)
        self._bus.subscribe(EventType.TTS_CHUNK_READY, self._broadcast_audio)
        self._bus.subscribe(EventType.SAFETY_FLAGGED, self._broadcast_safety)
        self._bus.subscribe(EventType.INTERRUPT_DETECTED, self._broadcast_interrupt)

    # ── server start ─────────────────────────────────────────────────────────

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.setup_routes()
        self._subscribe_events()
        config = uvicorn.Config(
            self._app,
            host=host,
            port=port,
            log_level="info",
            ws_ping_interval=PING_INTERVAL,
        )
        server = uvicorn.Server(config)
        await server.serve()

    # ── connection handler ───────────────────────────────────────────────────

    async def _handle_connection(self, websocket: WebSocket) -> None:
        await websocket.accept()

        # Generate unique connection ID
        WebSocketServer._conn_counter += 1
        conn_id = f"ws_{int(time.time())}_{WebSocketServer._conn_counter}"

        # Check capacity
        if len(self._connections) >= MAX_CONNECTIONS:
            await websocket.send_json({
                "type": MSG_TYPE_ERROR,
                "message": "Max connections reached",
            })
            await websocket.close()
            return

        self._connections[conn_id] = websocket

        # First connection becomes active
        if self._active_connection is None:
            self._active_connection = conn_id

        self._logger.info("Client %s connected (active=%s)", conn_id, conn_id == self._active_connection)

        # Send initial state
        state = (
            self._session.state.value
            if hasattr(self._session, "state") and hasattr(self._session.state, "value")
            else str(getattr(self._session, "state", "unknown"))
        )
        await websocket.send_json({
            "type": MSG_TYPE_STATE,
            "state": state,
            "is_active": conn_id == self._active_connection,
        })

        # Message loop
        try:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    await self._handle_audio(conn_id, message["bytes"])
                elif "text" in message and message["text"]:
                    try:
                        parsed = json.loads(message["text"])
                        await self._handle_text_message(conn_id, parsed)
                    except json.JSONDecodeError:
                        self._logger.warning("Invalid JSON from %s", conn_id)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            self._logger.error("Connection %s error: %s", conn_id, exc)
        finally:
            self._connections.pop(conn_id, None)
            if conn_id == self._active_connection:
                self._active_connection = next(iter(self._connections), None)
                if self._active_connection:
                    self._logger.info("Active connection transferred to %s", self._active_connection)
            self._logger.info("Client %s disconnected", conn_id)

    # ── incoming message handlers ────────────────────────────────────────────

    async def _handle_audio(self, conn_id: str, audio_bytes: bytes) -> None:
        if conn_id != self._active_connection:
            return  # Only active connection sends audio

        # Inject directly into MicStream's consumer queues so it flows
        # through VAD → STT → LLM pipeline (same path as hardware mic)
        if self._mic_stream is not None and hasattr(self._mic_stream, "inject_chunk"):
            self._mic_stream.inject_chunk(audio_bytes)
        else:
            # Fallback: publish event (nothing subscribes to this yet, but log it)
            self._logger.warning("No MicStream reference — WebSocket audio dropped")
            await self._bus.publish(
                EventType.AUDIO_CHUNK,
                {
                    "audio": audio_bytes,
                    "source": "websocket",
                    "conn_id": conn_id,
                },
            )

    async def _handle_text_message(self, conn_id: str, message: dict) -> None:
        msg_type = message.get("type")

        if msg_type == MSG_TYPE_TEXT:
            text = message.get("text", "").strip()
            if text:
                await self._bus.publish(
                    EventType.TRANSCRIPT_READY,
                    {
                        "text": text,
                        "is_final": True,
                        "source": "websocket",
                    },
                )

        elif msg_type == MSG_TYPE_CONTROL:
            action = message.get("action", "")
            if action == "interrupt":
                await self._bus.publish(EventType.INTERRUPT_DETECTED, {"source": "websocket", "conn_id": conn_id})
            elif action == "mute":
                if conn_id == self._active_connection:
                    self._active_connection = None
                    self._logger.info("Client %s muted (no active connection)", conn_id)
            elif action == "unmute":
                self._active_connection = conn_id
                self._logger.info("Client %s unmuted (now active)", conn_id)
            else:
                self._logger.warning("Unknown control action '%s' from %s", action, conn_id)

        elif msg_type == MSG_TYPE_CONFIG:
            self._logger.info("Config message from %s: %s", conn_id, message)

        else:
            self._logger.warning("Unknown message type '%s' from %s", msg_type, conn_id)

    # ── broadcast handlers (EventBus → all clients) ─────────────────────────

    async def _broadcast_state(self, event) -> None:
        new_state = event.data.get("new_state")
        state_val = new_state.value if hasattr(new_state, "value") else str(new_state)
        await self._broadcast_json({"type": "state", "state": state_val})

    async def _broadcast_transcript(self, event) -> None:
        await self._broadcast_json({
            "type": "transcript",
            "text": event.data.get("text", ""),
            "is_final": event.data.get("is_final", False),
        })

    async def _broadcast_llm_sentence(self, event) -> None:
        await self._broadcast_json({
            "type": "ai_text",
            "text": event.data.get("text", ""),
        })

    async def _broadcast_audio(self, event) -> None:
        audio = event.data.get("audio") if isinstance(event.data, dict) else event.data
        if not audio:
            return
        audio_bytes = audio if isinstance(audio, (bytes, bytearray)) else bytes(audio)
        for conn_id, ws in list(self._connections.items()):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_bytes(audio_bytes)
            except Exception:
                self._connections.pop(conn_id, None)
                if conn_id == self._active_connection:
                    self._active_connection = next(iter(self._connections), None)

    async def _broadcast_safety(self, event) -> None:
        await self._broadcast_json({
            "type": "safety",
            "direction": event.data.get("direction", ""),
            "categories": event.data.get("categories", []),
        })

    async def _broadcast_interrupt(self, event) -> None:
        await self._broadcast_json({"type": "interrupt"})

    # ── broadcast utility ────────────────────────────────────────────────────

    async def _broadcast_json(self, data: dict) -> None:
        for conn_id, ws in list(self._connections.items()):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(data)
            except Exception:
                self._logger.debug("Removing dead connection %s", conn_id)
                self._connections.pop(conn_id, None)
                if conn_id == self._active_connection:
                    self._active_connection = next(iter(self._connections), None)