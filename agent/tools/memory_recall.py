"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  VOXCORE — agent/tools/memory_recall.py                         ║
║       SEARCH LONG-TERM MEMORY FOR PAST SESSIONS / USER CONTEXT                ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Thin wrapper over MemoryRetriever. Searches memory for information about
    the user, relationships, preferences, or topics discussed in PREVIOUS sessions.

    produces_spoken_output = False — result is context for the main LLM to
    synthesize into a spoken response.
"""

import logging

from agent.tools.base_tool import BaseTool
from memory.retriever import MemoryRetriever


class MemoryRecallTool(BaseTool):
    """Search memory for information from past sessions."""

    name = "memory_recall"
    description = (
        "Search memory for information about the user, relationships, preferences, "
        "or topics discussed in PREVIOUS SESSIONS. Use when user references "
        "something from the past, mentions a person needing context, or asks "
        "what you remember about something. "
        "Do NOT use for current internet info — use web_search. "
        "Do NOT use for content from this session — use session_cache_qa."
    )
    required_params = ["query"]
    optional_params = ["memory_type", "top_k"]
    produces_spoken_output = False  # context for main LLM to synthesize

    def __init__(self, config: dict | None = None, retriever: MemoryRetriever | None = None) -> None:
        super().__init__(config)
        self._retriever = retriever

    async def execute(self, params: dict) -> str:
        """Search memory and return formatted context."""
        valid, err = self.validate_params(params)
        if not valid:
            return f"Memory recall error: {err}"

        if self._retriever is None:
            return "Memory system is not available right now."

        try:
            result = await self._retriever.recall(
                query=params["query"],
                memory_type=params.get("memory_type"),
                top_k=params.get("top_k", 5),
            )
            return self._retriever.format_for_injection(result)
        except Exception as e:
            self._logger.error("Memory recall failed: %s", e, exc_info=True)
            return "I couldn't access my memory right now. Can you tell me again?"
