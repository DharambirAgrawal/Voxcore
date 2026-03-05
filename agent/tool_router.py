"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                       VOXCORE — agent/tool_router.py                            ║
║        ROUTES AGENT TOOL CALLS TO HANDLERS — INJECTS RESULTS BACK IN          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Receives validated agent JSON from TextOut (via AGENT_JSON_OUT event).
    Reads the "action" field and routes to the appropriate tool handler in
    agent/tools/. After tool execution, injects the result back into the
    conversation via TextInjector so the fast brain picks it up in the next turn.

    Key flow:
    1. Agent JSON arrives with action="web_search"
    2. ToolRouter finds WebSearchTool in registry
    3. Calls tool.execute(params) asynchronously
    4. Injects result via text_injector.inject_tool_result(action, result)
    5. The fast brain sees it in [CONTEXT] on the next turn

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger
import time                             # Execution timing

from typing import Optional
from core.event_bus import EventBus, EventType
from input.text_injector import TextInjector

# Tool imports
from agent.tools.base_tool import BaseTool
from agent.tools.web_search import WebSearchTool
from agent.tools.article_fetch import ArticleFetchTool

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: ToolRouter
──────────────────────────────────────────────────────────────────────────────────
    Routes agent tool calls to registered tool handlers.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, text_injector: TextInjector, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — Subscribe to AGENT_JSON_OUT, publish TOOL_RESULT_READY
            - text_injector: TextInjector — For injecting tool results back into conversation
            - config: dict — The "agent" section from config.yaml:
                - tools: list[str] — Enabled tool names
                - tool_timeout_s: int (30)
                - max_concurrent_tools: int (3)
        
        INITIALIZES:
            self.event_bus: EventBus             = event_bus
            self.text_injector: TextInjector     = text_injector
            self.tool_timeout: int               = config.get("tool_timeout_s", 30)
            self.max_concurrent: int             = config.get("max_concurrent_tools", 3)
            
            # Tool registry — maps action names to tool instances
            self._tools: dict[str, BaseTool]     = {}
            self._active_tasks: list[asyncio.Task] = []  # Running tool tasks
            
            self._agent_json_queue: asyncio.Queue = event_bus.subscribe(EventType.AGENT_JSON_OUT)
            
            self._logger: logging.Logger         = logging.getLogger("ToolRouter")
            
            # Register enabled tools
            self._register_tools(config.get("tools", []))

    METHODS:
    ─────────────────────────────────────────────────────────────

    def _register_tools(self, enabled_tools: list[str]) -> None
        INPUTS:
            - enabled_tools: list[str] — Tool names from config (e.g. ["web_search", "article_fetch"])
        OUTPUT: None (populates self._tools)
        WHAT IT DOES:
            1. Available tool mapping:
               TOOL_MAP = {
                   "web_search":    WebSearchTool,
                   "article_fetch": ArticleFetchTool,
               }
            2. For each name in enabled_tools:
               If name in TOOL_MAP:
                   self._tools[name] = TOOL_MAP[name]()
                   Log: "Registered tool: {name}"
               Else:
                   Log warning: "Unknown tool: {name}"
            3. Log: "Registered {len(self._tools)} tools"

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            Infinite loop:
            1. event = await self._agent_json_queue.get()
            2. action = event.data["action"]
            3. params = event.data.get("params", {})
            4. If action not in self._tools:
               Log warning: "No tool registered for action '{action}'"
               await text_injector.inject_tool_result(action, f"Unknown tool: {action}", success=False)
               Continue
            5. If len(self._active_tasks) >= self.max_concurrent:
               Log warning: "Too many concurrent tools — queueing"
               await asyncio.sleep(1)
            6. task = asyncio.create_task(self._execute_tool(action, params))
            7. self._active_tasks.append(task)
            8. task.add_done_callback(lambda t: self._active_tasks.remove(t))

    async def _execute_tool(self, action: str, params: dict) -> None
        INPUTS:
            - action: str — Tool action name
            - params: dict — Tool parameters
        OUTPUT: None (injects result via text_injector)
        WHAT IT DOES:
            1. tool = self._tools[action]
            2. start = time.time()
            3. Log: "Executing tool '{action}' with params: {params}"
            4. Try with timeout:
               result = await asyncio.wait_for(
                   tool.execute(params),
                   timeout=self.tool_timeout
               )
               duration = time.time() - start
               Log: "Tool '{action}' completed in {duration:.1f}s"
               
               # Inject result back into conversation
               await self.text_injector.inject_tool_result(action, str(result), success=True)
               
               # Publish event for monitoring
               await self.event_bus.publish(EventType.TOOL_RESULT_READY, {
                   "action": action,
                   "result": result,
                   "success": True,
               }, source="ToolRouter")
            
            5. Except asyncio.TimeoutError:
               Log error: "Tool '{action}' timed out after {self.tool_timeout}s"
               await self.text_injector.inject_tool_result(action, "Tool timed out", success=False)
            
            6. Except Exception as e:
               Log error: "Tool '{action}' failed: {e}"
               await self.text_injector.inject_tool_result(action, f"Error: {str(e)}", success=False)

    def register_tool(self, name: str, tool: BaseTool) -> None
        INPUTS:
            - name: str — Action name to register
            - tool: BaseTool — Tool instance
        OUTPUT: None
        WHAT IT DOES:
            1. self._tools[name] = tool
            2. Log: "Registered tool: {name}"
        
        Allows dynamic tool registration at runtime.

    def get_registered_tools(self) -> list[str]
        INPUTS: None
        OUTPUT: list[str] — Names of all registered tools
        WHAT IT DOES:
            Returns list(self._tools.keys())

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - ToolRouter   (class)
═══════════════════════════════════════════════════════════════════════════════════

ADDING A NEW TOOL:
    1. Create a new file in agent/tools/ (e.g. email_tool.py)
    2. Extend BaseTool with an execute(params) method
    3. Import it in tool_router.py
    4. Add it to TOOL_MAP
    5. Add the tool name to config.yaml agent.tools list
    That's it — the router handles the rest.
"""


import asyncio
import logging
import time

from typing import Optional
from core.event_bus import EventBus, EventType
from input.text_injector import TextInjector

from agent.tools.base_tool import BaseTool
from agent.tools.web_search import WebSearchTool
from agent.tools.article_fetch import ArticleFetchTool


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
            "web_search":    WebSearchTool,
            "article_fetch": ArticleFetchTool,
        }

        for name in enabled_tools:
            if name in TOOL_MAP:
                self._tools[name] = TOOL_MAP[name](config=self._config)
                self._logger.info("Registered tool: %s", name)
            else:
                self._logger.warning("Unknown tool: %s", name)

        self._logger.info("Registered %d tools", len(self._tools))

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

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """Dynamically register a tool at runtime."""
        self._tools[name] = tool
        self._logger.info("Registered tool: %s", name)

    def get_registered_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())