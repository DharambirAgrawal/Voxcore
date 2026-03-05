"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE v4 — websocket/server.py                           ║
║    WEBSOCKET ACCEPT, AUTH, PER-USER SESSION MANAGER, FRAME ROUTING             ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Accepts WebSocket connections, authenticates via API key, creates a
    PipelineInstance per user using shared resources, routes binary/JSON
    frames, and tears down cleanly on disconnect.

    Enforces max concurrent sessions (default 8).
"""

import asyncio
import json
import logging
import urllib.parse

import websockets
from websockets.exceptions import ConnectionClosed

from websocket.pipeline_instance import PipelineInstance

logger = logging.getLogger("VoxCoreServer")


class VoxCoreServer:
    """WebSocket server that manages per-user pipeline instances."""

    def __init__(self, shared_resources):
        self.shared = shared_resources
        ws_cfg = shared_resources.config.get("websocket", {})
        self.api_key = ws_cfg.get("api_key", "voxcore-dev-key-change-me")
        self.max_sessions = ws_cfg.get("max_sessions", 8)
        self.active: dict[str, PipelineInstance] = {}  # user_id → instance

    async def start(self) -> None:
        """Start the WebSocket server and accept connections forever."""
        ws_cfg = self.shared.config.get("websocket", {})
        host = ws_cfg.get("host", "0.0.0.0")
        port = ws_cfg.get("port", 8765)
        ping_interval = ws_cfg.get("ping_interval", 10)
        ping_timeout = ws_cfg.get("ping_timeout", 30)
        max_size = ws_cfg.get("max_frame_size", 65536)

        logger.info("VoxCore WebSocket server starting on ws://%s:%d", host, port)

        async with websockets.serve(
            self._handler,
            host,
            port,
            max_size=max_size,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        ):
            logger.info(
                "Ready. Max %d concurrent sessions. Waiting for connections...",
                self.max_sessions,
            )
            await asyncio.Future()  # run forever

    def _extract_auth(self, websocket) -> str:
        """Extract API key from headers or query params (works with websockets v12+).

        Supports three auth methods:
        1. Authorization: Bearer <api_key>   (preferred)
        2. X-API-Key: <api_key>              (ESP32 simplicity)
        3. ?token=<api_key>                  (browser / Python client fallback)
        """
        # websockets v12+ exposes the HTTP request on websocket.request
        request = getattr(websocket, "request", None)
        if request is None:
            return ""

        headers = request.headers

        key = headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not key:
            key = headers.get("X-API-Key", "").strip()

        # Fallback: query parameter
        if not key:
            path = getattr(request, "path", "")
            parsed = urllib.parse.urlparse(path)
            params = urllib.parse.parse_qs(parsed.query)
            key = params.get("token", [""])[0]

        return key

    async def _handler(self, websocket) -> None:
        """Handle a single WebSocket connection lifecycle.

        1. Authenticate via key in header/query-param
        2. Extract user_id from query params
        3. Create PipelineInstance with shared resources
        4. Start pipeline, send 'ready'
        5. Route frames: binary → mic pipeline, JSON → handle control
        6. Clean up on disconnect
        """
        # ── Auth (done here because process_request API changed in websockets v12+)
        key = self._extract_auth(websocket)
        if key != self.api_key:
            logger.warning("[!] auth_failed (key=%s)", key[:8] if key else "empty")
            await websocket.close(1008, "Unauthorized")
            return

        if len(self.active) >= self.max_sessions:
            logger.warning("[!] Server at capacity (%d/%d)", len(self.active), self.max_sessions)
            await websocket.close(1013, "Server at capacity")
            return

        # ── Extract user_id from query params
        request_path = getattr(websocket.request, "path", "")
        parsed = urllib.parse.urlparse(request_path)
        params = urllib.parse.parse_qs(parsed.query)
        user_id = params.get("user_id", [f"user_{id(websocket)}"])[0]

        remote = websocket.remote_address
        logger.info("[+] %s connected from %s", user_id, remote)

        # Create per-user pipeline
        instance = PipelineInstance(websocket, user_id, self.shared)
        self.active[user_id] = instance

        try:
            # Start the full pipeline
            await instance.start()

            # Signal client: warmup done, start sending mic audio
            await websocket.send(json.dumps({"type": "ready"}))
            logger.info("[%s] Pipeline ready — sent 'ready' to client", user_id)

            # Main message loop
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary frame = raw PCM mic audio → pipeline
                    instance.bridge.push_mic_audio(message)
                else:
                    # JSON control message
                    try:
                        msg = json.loads(message)
                        await self._handle_json(websocket, user_id, msg)
                    except json.JSONDecodeError:
                        logger.warning("[%s] Invalid JSON: %s", user_id, message[:100])

        except ConnectionClosed as e:
            code = getattr(e.rcvd, "code", None) if getattr(e, "rcvd", None) else None
            logger.info("[-] %s disconnected (code=%s)", user_id, code)
        except Exception as e:
            logger.error("[!] %s error: %s", user_id, e, exc_info=True)
        finally:
            await instance.stop()
            self.active.pop(user_id, None)
            logger.info("[%s] cleaned up. Active sessions: %d", user_id, len(self.active))

    async def _handle_json(self, websocket, user_id: str, msg: dict) -> None:
        """Handle JSON control messages from client."""
        msg_type = msg.get("type", "")

        if msg_type == "ping":
            await websocket.send(json.dumps({
                "type": "pong",
                "ts": msg.get("ts"),
            }))

        elif msg_type == "session_start":
            client_id = msg.get("client_id", "unknown")
            logger.info("[%s] Session start from client: %s", user_id, client_id)

        else:
            logger.debug("[%s] Unhandled JSON type: %s", user_id, msg_type)
