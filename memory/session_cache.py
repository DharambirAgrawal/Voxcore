"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/session_cache.py                            ║
║           TIER 2 — RAM-ONLY SESSION CACHE FOR FULL TOOL RESULTS                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Stores full tool results (article text, search results) in RAM for
    instant follow-up access during the current session.

    Working memory stores only a ~60-word summary with a [ref:cache_id] tag.
    When user asks a follow-up, session_cache_qa retrieves the full content
    from here — zero network latency.

    Wiped on session shutdown. Re-fetching next session is cheaper than
    stale permanent storage.
"""

import logging
import time
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("SessionCache")


class SessionCache:
    """RAM-only session cache for full tool output content."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        # {
        #   cache_id: {
        #     content_type: "article" | "search_results",
        #     full_content: str,
        #     summary: str,
        #     source_url: Optional[str],
        #     title: Optional[str],
        #     word_count: int,
        #     stored_at: float,
        #     access_count: int,
        #   }
        # }

    def store(
        self,
        content_type: str,
        full_content: str,
        summary: str,
        source_url: str = None,
        title: str = None,
    ) -> str:
        """Store content and return its cache_id."""
        cache_id = f"sc_{int(time.time())}_{uuid4().hex[:4]}"
        self._store[cache_id] = {
            "content_type": content_type,
            "full_content": full_content,
            "summary": summary,
            "source_url": source_url,
            "title": title,
            "word_count": len(full_content.split()),
            "stored_at": time.time(),
            "access_count": 0,
        }
        logger.info(
            "Cached %s [%s]: %d words (id=%s)",
            content_type,
            title or "untitled",
            len(full_content.split()),
            cache_id,
        )
        return cache_id

    def get_content(self, cache_id: str) -> Optional[str]:
        """Return full_content for a cache_id, incrementing access count.

        Returns None if not found — never raises.
        """
        entry = self._store.get(cache_id)
        if entry is None:
            return None
        entry["access_count"] += 1
        return entry["full_content"]

    def get(self, cache_id: str) -> Optional[dict]:
        """Return the full entry dict, or None."""
        return self._store.get(cache_id)

    def list_all(self) -> list[dict]:
        """Return summary info for all cached items (without full_content).

        Used by the AI to see what's cached this session.
        """
        result = []
        for cid, entry in self._store.items():
            result.append({
                "cache_id": cid,
                "content_type": entry["content_type"],
                "title": entry["title"],
                "source_url": entry["source_url"],
                "summary": entry["summary"],
                "word_count": entry["word_count"],
            })
        return result

    def clear(self) -> None:
        """Wipe all cached content. Called at session end."""
        count = len(self._store)
        self._store.clear()
        logger.info("Session cache cleared (%d entries)", count)
