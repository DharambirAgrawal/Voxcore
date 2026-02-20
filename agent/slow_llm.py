"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         VOXCORE — agent/slow_llm.py                             ║
║     ASYNC SLOW LLM CALLS — KIMI-K2 / SCOUT-17B FOR COMPLEX TASKS              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Handles tasks that need more intelligence than llama-3.1-8b-instant.
    Called asynchronously by ToolRouter when the agent tag specifies a
    complex reasoning task, or by BrainRouter for secondary processing.

    Uses:
    - kimi-k2-instruct for tool-use tasks (best tool-calling ability)
    - llama-4-scout-17b for reasoning tasks
    - qwen3-32b for document summarization
    - gpt-oss-120b for maximum intelligence (rare)

    ALWAYS runs in the background. The fast brain already said "let me look
    into that" BEFORE slow_llm even starts. Results are injected back via
    TextInjector.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import asyncio                          # Async operations
import logging                          # Module logger

from brain.llm_client import LLMClient
from brain.router import BrainRouter
from input.text_injector import TextInjector
from core.event_bus import EventBus

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: SlowLLM
──────────────────────────────────────────────────────────────────────────────────
    Manages async calls to slower, more powerful LLMs.

    CONSTRUCTOR: __init__(self, llm_client: LLMClient, brain_router: BrainRouter,
                          text_injector: TextInjector, event_bus: EventBus)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - llm_client: LLMClient — The shared Groq LLM client
            - brain_router: BrainRouter — For determining which model to use
            - text_injector: TextInjector — For injecting results back into conversation
            - event_bus: EventBus — For publishing events
        
        INITIALIZES:
            self.llm_client: LLMClient           = llm_client
            self.brain_router: BrainRouter       = brain_router
            self.text_injector: TextInjector     = text_injector
            self.event_bus: EventBus             = event_bus
            self._active_tasks: list[asyncio.Task] = []
            self._logger: logging.Logger         = logging.getLogger("SlowLLM")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def process_complex_task(self, task_description: str, context: str,
                                    model: str = None) -> str
        INPUTS:
            - task_description: str — What the slow LLM should do
            - context: str — Relevant context (conversation history, injected data, etc.)
            - model: str — Override model (if None, brain_router decides)
        OUTPUT:
            - str — The LLM's complete response text
        WHAT IT DOES:
            1. If model is None:
               model = brain_router.route_agent_task(task_description)
            2. Build messages:
               messages = [
                   {"role": "system", "content": "You are a task execution assistant. Provide clear, concise results. No emotion tags."},
                   {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{task_description}"}
               ]
            3. Log: "SlowLLM: processing task on {model}"
            4. Collect full response (non-streaming for simplicity):
               result = await llm_client.complete(messages, model)
            5. Log: "SlowLLM: task completed ({len(result)} chars)"
            6. Return result

    async def process_and_inject(self, task_description: str, context: str,
                                  action_name: str, model: str = None) -> None
        INPUTS:
            - task_description: str — Task description
            - context: str — Relevant context
            - action_name: str — The original action name (for injection labeling)
            - model: str — Override model
        OUTPUT: None (injects result via TextInjector)
        WHAT IT DOES:
            1. result = await self.process_complex_task(task_description, context, model)
            2. await text_injector.inject_tool_result(action_name, result, success=True)
            3. Log: "SlowLLM result injected for action '{action_name}'"
        
        This is the primary method used by ToolRouter and TurnManager.
        It runs asynchronously — fire and forget.

    def launch_background_task(self, task_description: str, context: str,
                                action_name: str, model: str = None) -> asyncio.Task
        INPUTS:
            - Same as process_and_inject
        OUTPUT:
            - asyncio.Task — The running background task (can be cancelled)
        WHAT IT DOES:
            1. task = asyncio.create_task(
                   self.process_and_inject(task_description, context, action_name, model)
               )
            2. self._active_tasks.append(task)
            3. task.add_done_callback(lambda t: self._active_tasks.remove(t))
            4. Return task
        
        Convenience method for fire-and-forget background processing.

    async def cancel_all(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. For each task in self._active_tasks:
               task.cancel()
            2. self._active_tasks.clear()
            3. Log: "All slow LLM tasks cancelled"

    @property
    def active_task_count(self) -> int
        OUTPUT: int — Number of currently running slow LLM tasks

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - SlowLLM   (class)
═══════════════════════════════════════════════════════════════════════════════════

USAGE PATTERN:
    # In TurnManager or ToolRouter:
    slow_llm.launch_background_task(
        task_description="Analyze pros and cons of solar energy",
        context=session.get_recent_history_text(),
        action_name="analysis",
        model=None  # Let router decide
    )
    # Result automatically appears in the next turn's [CONTEXT] block
"""


import asyncio
import logging
from typing import Optional

from brain.llm_client import LLMClient
from brain.router import BrainRouter
from core.event_bus import EventBus
from input.text_injector import TextInjector


class SlowLLM:
    """
    Manages async calls to slower, more powerful LLMs.
    Handles tasks that need more intelligence than the fast conversational model.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        brain_router: BrainRouter,
        text_injector: TextInjector,
        event_bus: EventBus,
    ) -> None:
        self.llm_client: LLMClient = llm_client
        self.brain_router: BrainRouter = brain_router
        self.text_injector: TextInjector = text_injector
        self.event_bus: EventBus = event_bus
        self._active_tasks: list[asyncio.Task] = []
        self._logger: logging.Logger = logging.getLogger("SlowLLM")

    async def process_complex_task(
        self, task_description: str, context: str, model: Optional[str] = None
    ) -> str:
        """
        Execute a complex task using a specific or router-selected LLM.
        Returns the raw result string.
        """
        if model is None:
            model = self.brain_router.route_agent_task(task_description)

        messages = [
            {
                "role": "system",
                "content": "You are a task execution assistant. Provide clear, concise results. No emotion tags.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nTask:\n{task_description}",
            },
        ]

        self._logger.info(f"SlowLLM: processing task on {model}")

        # Assuming complete() returns the full string response
        result = await self.llm_client.complete(messages, model=model)

        self._logger.info(f"SlowLLM: task completed ({len(result)} chars)")
        return result

    async def process_and_inject(
        self,
        task_description: str,
        context: str,
        action_name: str,
        model: Optional[str] = None,
    ) -> None:
        """
        Process a task and automatically inject the result into the text injector.
        """
        try:
            result = await self.process_complex_task(task_description, context, model)
            await self.text_injector.inject_tool_result(
                action_name, result, success=True
            )
            self._logger.info(f"SlowLLM result injected for action '{action_name}'")
        except Exception as e:
            self._logger.error(
                f"SlowLLM failed for action '{action_name}': {e}", exc_info=True
            )
            # Optionally inject the failure so the agent knows it failed
            await self.text_injector.inject_tool_result(
                action_name, f"Error processing task: {str(e)}", success=False
            )

    def launch_background_task(
        self,
        task_description: str,
        context: str,
        action_name: str,
        model: Optional[str] = None,
    ) -> asyncio.Task:
        """
        Fire-and-forget method to launch a slow LLM task in the background.
        """
        task = asyncio.create_task(
            self.process_and_inject(task_description, context, action_name, model)
        )
        self._active_tasks.append(task)

        # Remove task from list when done to prevent memory leaks
        task.add_done_callback(
            lambda t: self._active_tasks.remove(t) if t in self._active_tasks else None
        )
        return task

    async def cancel_all(self) -> None:
        """Cancel all currently running slow LLM tasks."""
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        self._active_tasks.clear()
        self._logger.info("All slow LLM tasks cancelled")

    @property
    def active_task_count(self) -> int:
        """Return the number of currently running slow LLM tasks."""
        return len(self._active_tasks)