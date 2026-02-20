"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         VOXCORE — agent/text_out.py                             ║
║     EMITS VALIDATED AGENT JSON TO EXTERNAL CONSUMERS (WEBHOOK / WEBSOCKET)     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Subscribes to LLM_AGENT_TAG events from ResponseParser. Validates the JSON
    structure, adds metadata (timestamp, session_id, turn_id), and emits it to
    all registered text-out consumers.

    Consumers can be:
    - WebSocket clients (real-time bi-directional)
    - REST webhook endpoints (HTTP POST)
    - Internal tool_router (for tool execution)
    - File log (for debugging)

    This is the AGENTIC OUTPUT CHANNEL — external systems connect here to
    receive structured task requests from VoxCore.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import json                             # JSON serialization
import time                             # Timestamps

import aiohttp                          # Async HTTP for webhook delivery

from core.session import Session
from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: TextOut
──────────────────────────────────────────────────────────────────────────────────
    Validates and distributes agent JSON output to external consumers.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — For session_id and turn_count metadata
            - event_bus: EventBus — Subscribe to LLM_AGENT_TAG, publish AGENT_JSON_OUT
            - config: dict — The "agent" section from config.yaml:
                - text_out_webhook: Optional[str] — URL to POST agent JSON
        
        INITIALIZES:
            self.session: Session                = session
            self.event_bus: EventBus             = event_bus
            self.webhook_url: Optional[str]      = config.get("text_out_webhook")
            
            self._agent_tag_queue: asyncio.Queue = event_bus.subscribe(EventType.LLM_AGENT_TAG)
            self._websocket_clients: list        = []   # Connected WebSocket clients
            self._output_log: list[dict]         = []   # In-memory log of all agent outputs
            
            self._logger: logging.Logger         = logging.getLogger("TextOut")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Infinite loop:
            1. event = await self._agent_tag_queue.get()
            2. agent_data = event.data  # {"action": str, "params": dict, "raw": str}
            3. validated = self._validate_and_enrich(agent_data)
            4. If validated is None: continue (invalid JSON, logged warning)
            5. Store in log: self._output_log.append(validated)
            6. Publish AGENT_JSON_OUT event with data = validated
            7. await self._distribute(validated)

    def _validate_and_enrich(self, agent_data: dict) -> Optional[dict]
        INPUTS:
            - agent_data: dict — Raw parsed agent tag data
        OUTPUT:
            - dict | None — Enriched agent JSON, or None if invalid
        WHAT IT DOES:
            1. Validate required fields:
               If "action" not in agent_data:
                   Log warning: "Agent tag missing 'action' field"
                   Return None
            2. Enrich with metadata:
               enriched = {
                   "action": agent_data["action"],
                   "params": agent_data.get("params", {}),
                   "session_id": self.session.session_id,
                   "turn_id": self.session.turn_count,
                   "timestamp": time.time(),
                   "timestamp_iso": datetime.utcnow().isoformat() + "Z",
               }
            3. Return enriched

    async def _distribute(self, agent_json: dict) -> None
        INPUTS:
            - agent_json: dict — Validated and enriched agent output
        OUTPUT: None
        WHAT IT DOES:
            1. Send to webhook (if configured):
               If self.webhook_url:
                   asyncio.create_task(self._send_webhook(agent_json))
            2. Send to all connected WebSocket clients:
               For each ws in self._websocket_clients:
                   asyncio.create_task(ws.send_json(agent_json))
            3. Log: "Agent output distributed: action={agent_json['action']}"

    async def _send_webhook(self, agent_json: dict) -> None
        INPUTS:
            - agent_json: dict — The agent output to POST
        OUTPUT: None
        WHAT IT DOES:
            1. async with aiohttp.ClientSession() as http_session:
                   async with http_session.post(
                       self.webhook_url,
                       json=agent_json,
                       headers={"Content-Type": "application/json"},
                       timeout=aiohttp.ClientTimeout(total=10)
                   ) as response:
                       If response.status != 200:
                           Log warning: "Webhook returned {status}"
            2. On any error: log and continue (don't block pipeline)

    def register_websocket(self, ws) -> None
        INPUTS:
            - ws — A WebSocket connection object
        OUTPUT: None
        WHAT IT DOES:
            Appends ws to self._websocket_clients

    def unregister_websocket(self, ws) -> None
        INPUTS:
            - ws — A WebSocket connection to remove
        OUTPUT: None
        WHAT IT DOES:
            Removes ws from self._websocket_clients (if present)

    def get_output_log(self) -> list[dict]
        INPUTS: None
        OUTPUT: list[dict] — All agent outputs from this session
        WHAT IT DOES:
            Returns a copy of self._output_log

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - TextOut   (class)
═══════════════════════════════════════════════════════════════════════════════════

EXAMPLE AGENT JSON OUTPUT:
    {
        "action": "web_search",
        "params": {"query": "London weather today"},
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "turn_id": 5,
        "timestamp": 1739980800.123,
        "timestamp_iso": "2026-02-19T12:00:00.123Z"
    }
"""



import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from core.event_bus import EventBus, EventType
from core.session import Session


class TextOut:
    """Validates and distributes agent JSON output to external consumers."""

    def __init__(self, session: Session, event_bus: EventBus, config: dict) -> None:
        self.session: Session = session
        self.event_bus: EventBus = event_bus
        self.webhook_url: Optional[str] = config.get("text_out_webhook")

        self._agent_tag_queue: asyncio.Queue = event_bus.subscribe(EventType.LLM_AGENT_TAG)
        self._websocket_clients: list = []
        self._output_log: list[dict] = []

        self._logger: logging.Logger = logging.getLogger("TextOut")

    async def run(self) -> None:
        """Main loop — consumes agent tag events and distributes them."""
        while True:
            event = await self._agent_tag_queue.get()
            agent_data = event.data  # {"action": str, "params": dict, "raw": str}

            validated = self._validate_and_enrich(agent_data)
            if validated is None:
                continue

            self._output_log.append(validated)
            await self.event_bus.publish(EventType.AGENT_JSON_OUT, data=validated)
            await self._distribute(validated)

    def _validate_and_enrich(self, agent_data: dict) -> Optional[dict]:
        """Validate required fields and enrich with session metadata."""
        if "action" not in agent_data:
            self._logger.warning("Agent tag missing 'action' field")
            return None

        enriched = {
            "action": agent_data["action"],
            "params": agent_data.get("params", {}),
            "session_id": self.session.session_id,
            "turn_id": self.session.turn_count,
            "timestamp": time.time(),
            "timestamp_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        return enriched

    async def _distribute(self, agent_json: dict) -> None:
        """Send validated agent JSON to all registered consumers."""
        # Webhook delivery
        if self.webhook_url:
            asyncio.create_task(self._send_webhook(agent_json))

        # WebSocket delivery
        for ws in list(self._websocket_clients):
            asyncio.create_task(self._send_ws(ws, agent_json))

        self._logger.info("Agent output distributed: action=%s", agent_json["action"])

    async def _send_webhook(self, agent_json: dict) -> None:
        """POST agent JSON to the configured webhook URL."""
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(
                    self.webhook_url,
                    json=agent_json,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status != 200:
                        self._logger.warning(
                            "Webhook returned %s", response.status
                        )
        except Exception as exc:
            self._logger.error("Webhook delivery failed: %s", exc)

    async def _send_ws(self, ws, agent_json: dict) -> None:
        """Send agent JSON to a single WebSocket client."""
        try:
            await ws.send_json(agent_json)
        except Exception as exc:
            self._logger.warning("WebSocket send failed: %s", exc)

    def register_websocket(self, ws) -> None:
        """Register a WebSocket connection to receive agent outputs."""
        self._websocket_clients.append(ws)

    def unregister_websocket(self, ws) -> None:
        """Remove a WebSocket connection from the client list."""
        try:
            self._websocket_clients.remove(ws)
        except ValueError:
            pass

    def get_output_log(self) -> list[dict]:
        """Return a copy of all agent outputs from this session."""
        return list(self._output_log)