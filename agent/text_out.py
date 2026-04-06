"""
VoxCore - agent/text_out.py
Validates and enriches agent JSON from LLM_AGENT_TAG events (adds session_id,
turn_id, timestamps), then distributes to WebSocket, webhooks, and ToolRouter.
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