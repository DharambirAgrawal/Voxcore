"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  VOXCORE v4 — websocket/client_controller.py                    ║
║          MAPS PIPELINE STATE → JSON COMMANDS SENT TO CONNECTED CLIENT          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Translates pipeline state changes and events into JSON WebSocket commands
    that any client (ESP32, browser, mobile app) can act on.

    The client does whatever it wants with these — LED colour on ESP32,
    CSS state on browser, nothing on a client that doesn't implement them.
"""

import json
import logging

logger = logging.getLogger("ClientController")

# Pipeline state → LED command mapping
STATE_COMMANDS = {
    "LISTENING":   "led_listening",
    "listening":   "led_listening",
    "THINKING":    "led_thinking",
    "thinking":    "led_thinking",
    "SPEAKING":    "led_speaking",
    "speaking":    "led_speaking",
    "PAUSED":      "led_thinking",
    "paused":      "led_thinking",
    "INTERRUPTED": "led_thinking",
    "interrupted": "led_thinking",
    "PENDING":     "led_thinking",
    "pending":     "led_thinking",
    "SOFT_INJECT": "led_speaking",
    "soft_inject": "led_speaking",
}


class ClientController:
    """Sends pipeline state/command JSON messages to a connected client."""

    def __init__(self, websocket, user_id: str = "unknown"):
        self.ws = websocket
        self.user_id = user_id

    async def send(self, action: str, extra: dict = None) -> None:
        """Send a command message to the client."""
        msg = {"type": "command", "action": action}
        if extra:
            msg.update(extra)
        try:
            await self.ws.send(json.dumps(msg))
            logger.debug("[%s] CMD → %s", self.user_id, action)
        except Exception as e:
            logger.warning("[%s] Failed to send command '%s': %s", self.user_id, action, e)

    async def on_state_change(self, event) -> None:
        """Called by event_bus when STATE_CHANGED fires.

        Sends both a LED command and a state_change notification.
        """
        # event.data contains {"old_state": str, "new_state": str}
        new_state = event.data.get("new_state", "") if hasattr(event, "data") and event.data else str(event)
        cmd = STATE_COMMANDS.get(new_state)
        if cmd:
            await self.send(cmd)
        # Also send state_change so client knows pipeline state
        try:
            await self.ws.send(json.dumps({
                "type": "state_change",
                "state": new_state,
            }))
        except Exception as e:
            logger.warning("[%s] Failed to send state_change: %s", self.user_id, e)

    # ── Speaker control commands ──────────────────────────────────────────

    async def speaker_pause(self, event=None) -> None:
        """TTS playback paused — client should stop playing and hold buffer."""
        await self.send("speaker_pause")

    async def speaker_resume(self, event=None) -> None:
        """TTS playback resumed — client should continue from buffer."""
        await self.send("speaker_resume")

    async def speaker_flush(self, event=None) -> None:
        """TTS buffer cleared — client should discard its local audio buffer."""
        await self.send("speaker_flush")

    # ── Mic control commands ──────────────────────────────────────────────

    async def mute_mic(self) -> None:
        """Optional: client can stop sending mic audio to save bandwidth."""
        await self.send("mute_mic")

    async def unmute_mic(self) -> None:
        """Optional: client resumes sending mic audio."""
        await self.send("unmute_mic")
