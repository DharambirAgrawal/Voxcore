"""
VoxCore - agent/slow_llm.py
Manages async background calls to slower, more powerful LLMs for complex
reasoning/tool-use, injecting results back via TextInjector.
"""

import asyncio
import logging
from typing import Optional

from brain.llm_client import LLMClient
from brain.router import BrainRouter
from input.text_injector import TextInjector
from core.event_bus import EventBus


class SlowLLM:
    """Background runner for slow/powerful LLM calls."""

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