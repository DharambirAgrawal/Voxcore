"""
VoxCore - agent/tool_router.py
Routes agent tool-call JSON to the matching handler in agent/tools/,
executes with timeout, and injects results back via TextInjector.
"""

import time
import asyncio
import logging

from core.event_bus import EventBus, EventType
from input.text_injector import TextInjector

from agent.tools.base_tool import BaseTool
from agent.tools.web_search import WebSearchTool
from agent.tools.article_fetch import ArticleFetchTool
from agent.tools.memory_recall import MemoryRecallTool
from agent.tools.session_cache_qa import SessionCacheQATool


class ToolRouter:
    """Routes agent tool calls to registered tool handlers."""

    def __init__(self, event_bus: EventBus, text_injector: TextInjector, config: dict) -> None:
        self.event_bus: EventBus = event_bus
        self.text_injector: TextInjector = text_injector
        self.tool_timeout: int = config.get("tool_timeout_s", 30)
        self.max_concurrent: int = config.get("max_concurrent_tools", 3)

        self._tools: dict[str, BaseTool] = {}
        self._active_tasks: list[asyncio.Task] = []

        self._agent_json_queue: asyncio.Queue = event_bus.subscribe(EventType.AGENT_JSON_OUT)

        self._logger: logging.Logger = logging.getLogger("ToolRouter")

        self._config: dict = config  # Full agent config — passed to tool constructors
        self._register_tools(config.get("tools", []))

    def _register_tools(self, enabled_tools: list[str]) -> None:
        """Register enabled tools from config into the internal registry."""
        TOOL_MAP: dict[str, type[BaseTool]] = {
            "web_search":       WebSearchTool,
            "article_fetch":    ArticleFetchTool,
            "memory_recall":    MemoryRecallTool,
            "session_cache_qa": SessionCacheQATool,
        }

        for name in enabled_tools:
            if name in TOOL_MAP:
                self._tools[name] = TOOL_MAP[name](config=self._config)
                self._logger.info("Registered tool: %s", name)
            else:
                self._logger.warning("Unknown tool: %s", name)

        self._logger.info("Registered %d tools", len(self._tools))

    def set_memory_components(self, *, session_cache=None, retriever=None, config=None) -> None:
        """V5: Re-register memory-aware tools with their dependencies injected.

        Called from main.py after memory components are created, because
        ToolRouter is instantiated before memory components exist.
        """
        full_config = config or {}

        # Re-register article_fetch with session_cache
        if "article_fetch" in self._tools and session_cache is not None:
            self._tools["article_fetch"] = ArticleFetchTool(
                config=self._config, session_cache=session_cache,
            )
            self._logger.info("Re-registered article_fetch with session_cache")

        # Re-register memory_recall with retriever
        if "memory_recall" in self._tools and retriever is not None:
            self._tools["memory_recall"] = MemoryRecallTool(
                config=self._config, retriever=retriever,
            )
            self._logger.info("Re-registered memory_recall with retriever")

        # Re-register session_cache_qa with session_cache
        if "session_cache_qa" in self._tools and session_cache is not None:
            self._tools["session_cache_qa"] = SessionCacheQATool(
                config=full_config, session_cache=session_cache,
            )
            self._logger.info("Re-registered session_cache_qa with session_cache")

    async def run(self) -> None:
        """Main loop — consumes AGENT_JSON_OUT events and dispatches tool execution."""
        while True:
            event = await self._agent_json_queue.get()
            action: str = event.data["action"]
            params: dict = event.data.get("params", {})

            if action not in self._tools:
                self._logger.warning("No tool registered for action '%s'", action)
                await self.text_injector.inject_tool_result(
                    action, f"Unknown tool: {action}", success=False
                )
                continue

            if len(self._active_tasks) >= self.max_concurrent:
                self._logger.warning("Too many concurrent tools — queueing")
                await asyncio.sleep(1)

            task = asyncio.create_task(self._execute_tool(action, params))
            self._active_tasks.append(task)
            task.add_done_callback(lambda t: self._active_tasks.remove(t))

    async def _execute_tool(self, action: str, params: dict) -> None:
        """Execute a single tool call with timeout.

        Routing after execution:
        - produces_spoken_output=True  → publish SPOKEN_TOOL_OUTPUT (TTS directly)
        - produces_spoken_output=False → inject result for main LLM to synthesize
        """
        tool = self._tools[action]
        start = time.time()
        self._logger.info("Executing tool '%s' with params: %s", action, params)

        try:
            result = await asyncio.wait_for(
                tool.execute(params),
                timeout=self.tool_timeout,
            )
            duration = time.time() - start
            self._logger.info("Tool '%s' completed in %.1fs", action, duration)

            if tool.produces_spoken_output:
                # Result IS the answer — send straight to TTS, skip main LLM
                await self.event_bus.publish(
                    EventType.SPOKEN_TOOL_OUTPUT,
                    {"text": str(result), "action": action},
                    source="ToolRouter",
                )
            else:
                # Result is context — inject for main LLM to read on next turn
                await self.text_injector.inject_tool_result(action, str(result), success=True)

            await self.event_bus.publish(
                EventType.TOOL_RESULT_READY,
                {
                    "action": action,
                    "result": result,
                    "success": True,
                    "spoken_output": tool.produces_spoken_output,
                },
                source="ToolRouter",
            )

        except asyncio.TimeoutError:
            self._logger.error("Tool '%s' timed out after %ds", action, self.tool_timeout)
            await self.text_injector.inject_tool_result(action, "Tool timed out", success=False)
            await self.event_bus.publish(
                EventType.TOOL_RESULT_READY,
                {"action": action, "result": "Tool timed out", "success": False},
                source="ToolRouter",
            )

        except Exception as e:
            self._logger.error("Tool '%s' failed: %s", action, e)
            await self.text_injector.inject_tool_result(
                action, f"Error: {str(e)}", success=False
            )
            await self.event_bus.publish(
                EventType.TOOL_RESULT_READY,
                {"action": action, "result": f"Error: {str(e)}", "success": False},
                source="ToolRouter",
            )