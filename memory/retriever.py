"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/retriever.py                               ║
║      PARALLEL MULTI-TIER MEMORY RECALL — NO LLM, PURE DATA RETRIEVAL          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Queries working memory, long-term (ChromaDB), and fact_store in PARALLEL
    via asyncio.gather(). Returns a combined result dict. No LLM calls —
    pure data retrieval + string formatting.

    Used by:
    - memory_recall tool (returns formatted text for main LLM to synthesize)
    - loader.py (session-start context assembly)
"""

import asyncio
import logging
import re
from typing import Optional

logger = logging.getLogger("MemoryRetriever")

# Keywords that hint the query is about a person or preference
_PERSON_SIGNALS = re.compile(
    r"\b(who is|about|remember|know about|relationship|friend|"
    r"partner|family|colleague|boss|wife|husband|brother|sister|"
    r"mother|father|mom|dad)\b",
    re.IGNORECASE,
)
_PREFERENCE_SIGNALS = re.compile(
    r"\b(prefer|like|love|hate|enjoy|favorite|favourite|dislike|"
    r"always|never|usually)\b",
    re.IGNORECASE,
)


class MemoryRetriever:
    """Parallel multi-tier memory recall — no LLM, pure data."""

    def __init__(
        self,
        short_term,
        long_term,
        fact_store,
    ) -> None:
        self._short_term = short_term
        self._long_term = long_term
        self._fact_store = fact_store

    async def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 5,
    ) -> dict:
        """Query all memory tiers in parallel and return combined result.

        Returns:
            {
                "working": [last 5 turns as strings],
                "memories": [chromadb results],
                "facts": {relevant fact_store entries},
                "token_estimate": int,
            }
        """
        # Launch all three paths in parallel
        working_task = asyncio.create_task(self._get_working_memory())
        long_term_task = asyncio.create_task(
            self._long_term.recall(query, memory_type=memory_type, top_k=top_k)
        )
        facts_task = asyncio.create_task(self._get_relevant_facts(query))

        working, memories, facts = await asyncio.gather(
            working_task, long_term_task, facts_task,
            return_exceptions=True,
        )

        # Handle exceptions gracefully — non-critical
        if isinstance(working, Exception):
            logger.warning("Working memory recall failed: %s", working)
            working = []
        if isinstance(memories, Exception):
            logger.warning("Long-term recall failed: %s", memories)
            memories = []
        if isinstance(facts, Exception):
            logger.warning("Fact store recall failed: %s", facts)
            facts = {}

        # Estimate tokens (~4 chars per token)
        total_chars = (
            sum(len(t) for t in working)
            + sum(len(m.get("content", "")) for m in memories)
            + sum(len(str(v)) for v in facts.values())
        )
        token_estimate = total_chars // 4

        return {
            "working": working,
            "memories": memories,
            "facts": facts,
            "token_estimate": token_estimate,
        }

    async def recall_for_person(self, name: str) -> dict:
        """Recall everything known about a specific person."""
        name_lower = name.lower()

        # Fact store: relationship + any prefix matches (SYNC — no create_task)
        facts_result = self._fact_store.get_by_prefix(f"relationship_{name_lower}")
        # ChromaDB: semantic + episodic search in parallel
        semantic_task = asyncio.create_task(
            self._long_term.recall(name, memory_type="semantic", top_k=3)
        )
        episodic_task = asyncio.create_task(
            self._long_term.recall(name, memory_type="episodic", top_k=3)
        )

        semantic, episodic = await asyncio.gather(
            semantic_task, episodic_task,
            return_exceptions=True,
        )

        if isinstance(facts_result, Exception):
            facts_result = {}
        if isinstance(semantic, Exception):
            semantic = []
        if isinstance(episodic, Exception):
            episodic = []

        return {
            "facts": facts_result,
            "semantic": semantic,
            "episodic": episodic,
        }

    def format_for_injection(self, result: dict) -> str:
        """Format recall result as plain text for text_injector.

        ~100-200 words max. No markdown. Goes through text_injector
        as tool result context for the main LLM.
        """
        parts: list[str] = []

        # Facts section
        facts = result.get("facts", {})
        if facts:
            fact_lines = []
            for key, value in facts.items():
                # Clean key for readability
                display_key = key.replace("_", " ").replace("pref ", "prefers ")
                fact_lines.append(f"  {display_key}: {value}")
            parts.append("Known facts:\n" + "\n".join(fact_lines[:8]))

        # Long-term memories section
        memories = result.get("memories", [])
        if memories:
            mem_lines = []
            for m in memories[:5]:
                content = m.get("content", "")
                # Truncate long entries
                if len(content) > 200:
                    content = content[:200] + "..."
                mem_lines.append(f"  - {content}")
            parts.append("Past memories:\n" + "\n".join(mem_lines))

        if not parts:
            return "No relevant memories found."

        return "\n\n".join(parts)

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _get_working_memory(self) -> list[str]:
        """Get last 5 turns from working memory as strings."""
        turns = self._short_term.get_recent(5)
        return [f"{t.role}: {t.content}" for t in turns]

    async def _get_relevant_facts(self, query: str) -> dict:
        """Check if query mentions a person or preference, return matching facts.

        NOTE: fact_store methods are SYNCHRONOUS (SQLite ~2ms).
        Do NOT await them — they return plain values, not coroutines.
        """
        results: dict = {}

        # Check for person-related queries
        if _PERSON_SIGNALS.search(query):
            # Extract potential names (capitalized words)
            words = query.split()
            for word in words:
                clean = word.strip(".,!?\"'")
                if clean and clean[0].isupper() and len(clean) > 1:
                    name_key = f"relationship_{clean.lower()}"
                    val = self._fact_store.get(name_key)
                    if val:
                        results[name_key] = val

        # Check for preference-related queries
        if _PREFERENCE_SIGNALS.search(query):
            pref_facts = self._fact_store.get_by_prefix("pref_")
            results.update(pref_facts)

        # Always check for core identity facts
        for key in ("user_name", "user_location", "user_occupation"):
            val = self._fact_store.get(key)
            if val:
                results[key] = val

        return results
