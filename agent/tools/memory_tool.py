"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — agent/tools/memory_tool.py                        ║
║            MEMORY TOOL — READ/WRITE LONG-TERM CONVERSATIONAL MEMORY            ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Provides read and write access to VoxCore's long-term ChromaDB memory.
    Allows the AI to explicitly store facts ("Remember that I like coffee")
    and retrieve past information ("What did I tell you about my schedule?").

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging                          # Module logger
from typing import Any

from agent.tools.base_tool import BaseTool
from memory.long_term import LongTermMemory

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: MemoryTool(BaseTool)
──────────────────────────────────────────────────────────────────────────────────
    Read/write long-term memory via ChromaDB.

    CLASS ATTRIBUTES:
        name = "memory"
        description = "Store or retrieve information from long-term memory"
        required_params = ["operation"]  # "read" or "write"
        optional_params = ["content", "query", "top_k"]

    CONSTRUCTOR: __init__(self, long_term_memory: LongTermMemory = None)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - long_term_memory: LongTermMemory — The ChromaDB memory instance
              (can be set later via set_memory())
        INITIALIZES:
            super().__init__()
            self.memory: Optional[LongTermMemory] = long_term_memory
            self._logger = logging.getLogger("Tool.memory")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def set_memory(self, memory: LongTermMemory) -> None
        INPUTS:
            - memory: LongTermMemory — The ChromaDB memory instance
        OUTPUT: None
        WHAT IT DOES:
            Sets self.memory = memory
            Called during initialization since tools are created before memory module.

    async def execute(self, params: dict) -> str
        INPUTS:
            - params: dict containing:
                - operation: str — "read" or "write" (REQUIRED)
                - content: str — Text to store (REQUIRED for "write")
                - query: str — Search query (REQUIRED for "read")
                - top_k: int — Number of results for read (default: 5)
        OUTPUT:
            - str — For "read": relevant memories as formatted text
                     For "write": confirmation message
        WHAT IT DOES:
            1. If self.memory is None: return "Memory system not initialized"
            2. operation = params.get("operation")
            3. If operation == "write":
               content = params.get("content", "")
               If not content: return "Error: 'content' required for write operation"
               await self.memory.store(content, metadata={"source": "explicit_memory"})
               Return f"Stored in memory: '{content[:100]}...'"
            4. If operation == "read":
               query = params.get("query", "")
               If not query: return "Error: 'query' required for read operation"
               top_k = params.get("top_k", 5)
               results = await self.memory.query(query, top_k=top_k)
               Return self._format_memories(results)
            5. Else: return f"Unknown operation: {operation}. Use 'read' or 'write'."

    def _format_memories(self, results: list[dict]) -> str
        INPUTS:
            - results: list[dict] — ChromaDB query results, each with:
                - content: str
                - distance: float
                - metadata: dict
        OUTPUT:
            - str — Formatted memory results
        WHAT IT DOES:
            1. If not results: return "No relevant memories found."
            2. output = "Retrieved memories:\n"
            3. For i, r in enumerate(results, 1):
               output += f"{i}. {r['content']}\n"
            4. Return output

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - MemoryTool   (class)
════════════════════════════════════════════a═══════════════════════════════════════
"""

import logging
from typing import Any, Optional

from agent.tools.base_tool import BaseTool
from memory.long_term import LongTermMemory


class MemoryTool(BaseTool):
    """Read/write long-term memory via ChromaDB."""

    name = "memory"
    description = "Store or retrieve information from long-term memory"
    required_params = ["operation"]  # "read" or "write"
    optional_params = ["content", "query", "top_k"]

    def __init__(self, long_term_memory: LongTermMemory = None) -> None:
        super().__init__()
        self.memory: Optional[LongTermMemory] = long_term_memory
        self._logger = logging.getLogger("Tool.memory")

    def set_memory(self, memory: LongTermMemory) -> None:
        """Attach the ChromaDB memory instance (may be wired after construction)."""
        self.memory = memory

    async def execute(self, params: dict) -> str:
        if self.memory is None:
            return "Memory system not initialized"

        operation = params.get("operation")

        if operation == "write":
            content = params.get("content", "") or ""
            if not content:
                return "Error: 'content' required for write operation"
            await self.memory.store(content, metadata={"source": "explicit_memory"})
            truncated = content[:100]
            return f"Stored in memory: '{truncated}...'"

        if operation == "read":
            query = params.get("query", "") or ""
            if not query:
                return "Error: 'query' required for read operation"
            top_k = params.get("top_k", 5)
            try:
                top_k = int(top_k)
            except (TypeError, ValueError):
                top_k = 5
            results = await self.memory.query(query, top_k=top_k)
            return self._format_memories(results)

        return f"Unknown operation: {operation}. Use 'read' or 'write'."

    def _format_memories(self, results: list[dict]) -> str:
        """Format ChromaDB query results into a human-readable string."""
        if not results:
            return "No relevant memories found."

        output = "Retrieved memories:\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['content']}\n"
        return output