"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — input/text_injector.py                           ║
║          TEXT-IN API — INJECTS EXTERNAL CONTEXT INTO THE CONVERSATION           ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Provides a simple interface for injecting text into the conversation context
    without the user speaking it. This is how external systems, tool results,
    RAG data, notifications, and any other text-based context gets into VoxCore.

    The injected text is queued in the Session's text_in_queue and gets flushed
    into the prompt by PromptBuilder on the next LLM call as a [CONTEXT] block.

    High-priority injections (tool results, alerts) appear before normal context.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import time                             # Timestamps

from core.session import Session
from core.event_bus import EventBus, EventType

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: TextInjector
──────────────────────────────────────────────────────────────────────────────────
    Manages text-in injection into the VoxCore conversation.

    CONSTRUCTOR: __init__(self, session: Session, event_bus: EventBus)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - session: Session — The master session state (where text_in_queue lives)
            - event_bus: EventBus — For publishing TEXT_INJECTED events
        
        INITIALIZES:
            self.session: Session        = session
            self.event_bus: EventBus     = event_bus
            self._logger: logging.Logger = logging.getLogger("TextInjector")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def inject(self, content: str, priority: str = "normal", source: str = "unknown") -> None
        INPUTS:
            - content: str — The text to inject into the conversation context
                Examples:
                    - Tool result: "London weather: 12°C, cloudy, light rain"
                    - RAG context: "User's previous meeting notes: ..."
                    - Notification: "New email from John: Subject 'Meeting update'"
                    - System: "Current time: 2:45 PM EST"
            - priority: str — "high" or "normal"
                - "high": Tool results, urgent alerts — prepended first in [CONTEXT] block
                - "normal": Background context, RAG data — appended after high priority
            - source: str — Identifier of what injected this text
                Examples: "tool_result", "rag", "notification", "websocket", "api", "memory"
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Validates content is non-empty string
            2. Calls await session.inject_text(content, priority, source)
            3. Publishes TEXT_INJECTED event with data:
               {"content": content, "priority": priority, "source": source}
            4. Logs: "Text injected ({priority}) from {source}: '{content[:80]}...'"

    async def inject_tool_result(self, action: str, result: str, success: bool = True) -> None
        INPUTS:
            - action: str — The tool action name (e.g. "web_search", "calendar")
            - result: str — The tool's output text
            - success: bool — Whether the tool executed successfully
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Formats the injection:
               If success:
                   content = f"[TOOL RESULT — {action}]: {result}"
               Else:
                   content = f"[TOOL ERROR — {action}]: {result}"
            2. Calls await self.inject(content, priority="high", source="tool_result")
        
        This is a convenience method specifically for tool_router to inject
        results back into the conversation.

    async def inject_memory(self, memories: list[str]) -> None
        INPUTS:
            - memories: list[str] — List of relevant past memory snippets
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Formats: content = "[PAST CONTEXT]:\n" + "\n".join(f"- {m}" for m in memories)
            2. Calls await self.inject(content, priority="normal", source="memory")
        
        Used at session start when LongTermMemory retrieves relevant past context.

    async def inject_system_context(self, context: str) -> None
        INPUTS:
            - context: str — System-level context (time, date, location, etc.)
        OUTPUT:
            - None
        WHAT IT DOES:
            1. Calls await self.inject(context, priority="normal", source="system")

    def get_pending_count(self) -> int
        INPUTS: None
        OUTPUT: int — Number of pending text-in entries waiting to be flushed
        WHAT IT DOES:
            Returns len(self.session.text_in_queue)

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - TextInjector   (class)
═══════════════════════════════════════════════════════════════════════════════════

USAGE EXAMPLES:
    # From tool_router after web search completes:
    await text_injector.inject_tool_result("web_search", "London: 12°C, cloudy")

    # From API endpoint when external system sends data:
    await text_injector.inject("User just booked flight to NYC", priority="high", source="api")

    # From memory module at session start:
    await text_injector.inject_memory(["User prefers morning meetings", "User is in EST timezone"])

    # From WebSocket when a notification arrives:
    await text_injector.inject("New Slack message from @jane: 'Ready for standup?'", source="websocket")
"""


"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — input/text_injector.py                           ║
║          TEXT-IN API — INJECTS EXTERNAL CONTEXT INTO THE CONVERSATION           ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
from typing import List

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
            content = f"[TOOL RESULT — {action}]: {result}"
        else:
            content = f"[TOOL ERROR — {action}]: {result}"

        await self.inject(content, priority="high", source="tool_result")

    async def inject_memory(self, memories: List[str]) -> None:
        """Inject retrieved long-term memory snippets as context."""
        content = "[PAST CONTEXT]:\n" + "\n".join(f"- {m}" for m in memories)
        await self.inject(content, priority="normal", source="memory")

    async def inject_system_context(self, context: str) -> None:
        """Inject system-level context such as time, date, or location."""
        await self.inject(context, priority="normal", source="system")

    def get_pending_count(self) -> int:
        """Return the number of pending text-in entries awaiting flush."""
        return len(self.session.text_in_queue)