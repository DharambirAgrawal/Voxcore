"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — input/text_injector.py                           ║
║          TEXT-IN API — INJECTS EXTERNAL CONTEXT INTO THE CONVERSATION           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import logging

from core.session import Session
from core.event_bus import EventBus, EventType


class TextInjector:
    """Manages text-in injection into the VoxCore conversation."""

    def __init__(self, session: Session, event_bus: EventBus) -> None:
        self.session: Session = session
        self.event_bus: EventBus = event_bus
        self._logger: logging.Logger = logging.getLogger("TextInjector")

    async def inject(
        self,
        content: str,
        priority: str = "normal",
        source: str = "unknown",
    ) -> None:
        """Inject arbitrary text into the conversation context."""
        if not content or not content.strip():
            self._logger.warning("Ignoring empty text injection from %s", source)
            return

        await self.session.inject_text(content, priority, source)

        await self.event_bus.publish(
            EventType.TEXT_INJECTED,
            {
                "content": content,
                "priority": priority,
                "source": source,
            },
        )

        preview = content[:80] + "..." if len(content) > 80 else content
        self._logger.info(
            "Text injected (%s) from %s: '%s'", priority, source, preview
        )

    async def inject_tool_result(
        self, action: str, result: str, success: bool = True
    ) -> None:
        """Convenience method for injecting tool execution results."""
        if success:
            content = f"Result from {action}: {result}"
        else:
            content = f"Error from {action}: {result}"

        await self.inject(content, priority="high", source="tool_result")