"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/background_llm.py                           ║
║           BACKGROUND EXTRACTION — FIRE-AND-FORGET FACT MINING                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    After every turn, extracts memorable information (facts, preferences,
    instructions, relationships) from the last 2-3 turns and routes them
    to the appropriate memory store.

    Runs as asyncio.create_task() — NEVER awaited in the hot path.

    Uses models.llm_fast from config. Pre-filters with signal phrases
    so filler turns cost zero API calls.

MODEL: config["models"]["llm_fast"]
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Optional

from groq import AsyncGroq

logger = logging.getLogger("BackgroundLLM")

# ─── Signal pre-filter ────────────────────────────────────────────────────────

SIGNAL_PHRASES = [
    "my name is", "i am", "i live", "i work", "i like", "i hate",
    "i prefer", "i always", "i never", "remember", "don't forget",
    "actually", "no that's wrong", "you keep saying", "it's not",
    "my friend", "my colleague", "my boss", "next week", "last week",
    "i told you", "make sure", "always remember", "by the way",
    "remind me", "i need to", "i have to", "deadline", "meeting",
    "appointment", "don't ever", "save this", "note that", "save that",
]


def has_signal(turns: list) -> bool:
    """Check if recent turns contain any extractable signal phrases."""
    combined = " ".join(
        t["content"].lower() for t in turns if t.get("role") == "user"
    )
    return any(phrase in combined for phrase in SIGNAL_PHRASES)


# ─── Extraction prompt ────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extract memorable information from these conversation turns.
Output one JSON object per line. No preamble. No explanation.
If nothing worth storing: output exactly: NOTHING

Each JSON must have exactly these fields:
{
  "operation": "ADD" | "UPDATE" | "CONFLICT" | "IGNORE",
  "store":     "fact" | "procedural" | "episodic" | "semantic",
  "key":       "snake_case_key_if_fact_store_else_null",
  "content":   "single clean third-person sentence",
  "importance": 0.0-1.0,
  "category":  "fact|preference|relationship|instruction|relay|event|correction",
  "expires_at": "ISO timestamp or null"
}

Rules:
- Greetings, filler, one-word responses → NOTHING
- "hi" "okay" "thanks" "got it" "yeah" "sure" → NOTHING
- Explicit correction ("actually X not Y") → UPDATE, importance 1.0
- "remember"/"don't forget"/"always"/"never"/"make sure" → procedural
- User name/city/job → fact, key=user_name/user_city/user_occupation
- Preferences → fact, key=pref_*
- Relationships → fact, key=relationship_{firstname_lowercase}
- Future events, deadlines → episodic + procedural reminder
- Past events → episodic, long_term
- Do NOT store: AI's words, tool results, prices, weather, news
- Do NOT store: what already exists unchanged"""


def _parse_iso_to_unix(iso_str: Optional[str]) -> Optional[int]:
    """Parse an ISO timestamp string to unix timestamp, or return None."""
    if not iso_str or iso_str == "null":
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, AttributeError):
        return None


class BackgroundLLM:
    """Fire-and-forget extraction from recent turns into memory stores.

    Called via asyncio.create_task() after each turn. Never awaited.
    """

    def __init__(self, config: dict) -> None:
        self._model: str = config.get("models", {}).get(
            "llm_fast", "llama-3.1-8b-instant"
        )
        self._enabled: bool = config.get("memory", {}).get(
            "background_enabled", True
        )
        self._groq = AsyncGroq(max_retries=0)

    async def process_turn_background(
        self,
        turns: list,
        fact_store,
        procedural,
        long_term,
        existing_summary: str,
    ) -> None:
        """Extract and route memorable information from recent turns.

        Args:
            turns: Last 2-3 Turn objects from short_term
            fact_store: FactStore instance
            procedural: ProceduralStore instance
            long_term: LongTermMemory instance
            existing_summary: Current session summary from loader
        """
        if not self._enabled:
            return

        if not turns:
            return

        # Pre-filter: zero API calls for filler
        if not has_signal(turns):
            return

        try:
            extracted = await self._call_extraction(turns, existing_summary, fact_store)
            if not extracted:
                return

            for item in extracted:
                await self._route_item(item, fact_store, procedural, long_term)

        except Exception as e:
            logger.error("Background extraction failed: %s", e)

    async def _call_extraction(
        self,
        turns: list,
        existing_summary: str,
        fact_store,
    ) -> list[dict]:
        """Call llm_fast to extract structured items from turns."""
        # Format turns
        formatted = "\n".join(
            f"[{t.get('role', 'unknown')}]: {t.get('content', '')}" for t in turns
        )

        # Build context with existing facts for conflict detection
        existing_facts = fact_store.get_all() if fact_store else {}
        context_addendum = ""
        if existing_facts:
            fact_lines = "\n".join(
                f"  {k}: {v}" for k, v in list(existing_facts.items())[:15]
            )
            context_addendum = (
                f"\n\nEXISTING STORED FACTS (check for conflicts/updates):\n{fact_lines}\n"
                "If a new value contradicts an existing one, set operation to UPDATE or CONFLICT."
            )

        user_content = (
            f"{EXTRACTION_PROMPT}{context_addendum}\n\n"
            f"Conversation turns:\n{formatted}"
        )

        if existing_summary:
            user_content += f"\n\nSession context summary:\n{existing_summary[:500]}"

        try:
            response = await self._groq.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=500,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()

            if not raw or raw.upper() == "NOTHING":
                return []

            return self._parse_extraction(raw)

        except Exception as e:
            logger.error("Extraction LLM call failed: %s", e)
            return []

    @staticmethod
    def _parse_extraction(raw: str) -> list[dict]:
        """Parse one-JSON-per-line extraction output."""
        items = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or line.upper() == "NOTHING":
                continue
            try:
                obj = json.loads(line)
                # Validate required fields
                if all(k in obj for k in ("operation", "store", "content")):
                    items.append(obj)
            except json.JSONDecodeError:
                # Try to extract JSON from the line
                try:
                    start = line.index("{")
                    end = line.rindex("}") + 1
                    obj = json.loads(line[start:end])
                    if all(k in obj for k in ("operation", "store", "content")):
                        items.append(obj)
                except (ValueError, json.JSONDecodeError):
                    continue
        return items

    async def _route_item(
        self,
        item: dict,
        fact_store,
        procedural,
        long_term,
    ) -> None:
        """Route a single extracted item to the correct store."""
        op = item.get("operation", "IGNORE")
        store = item.get("store", "")

        if op == "IGNORE":
            return

        if store == "fact":
            key = item.get("key")
            if not key:
                return

            # On UPDATE/CONFLICT, log the change to long-term memory
            if op in ("UPDATE", "CONFLICT"):
                existing = fact_store.get(key)
                if existing:
                    await long_term.store(
                        text=(
                            f"Previous: '{existing}' → Updated: '{item['content']}'"
                        ),
                        memory_type="semantic",
                        category="correction",
                        importance=0.8,
                    )

            # Deduplicate: skip ADD if the exact same value is already stored.
            # Background runs on every turn; the same fact can be extracted
            # repeatedly across turns while it sits in the recent-10 window.
            if op == "ADD" and fact_store.get(key) == item["content"]:
                logger.debug("Fact unchanged, skipping: %s", key)
                return

            fact_store.set(
                key,
                item["content"],
                confidence=1.0 if op == "UPDATE" else item.get("importance", 0.7),
                source="user_correction" if op == "UPDATE" else "extracted",
            )
            logger.info("Fact stored: %s = %s", key, item["content"][:60])

        elif store == "procedural":
            expires = _parse_iso_to_unix(item.get("expires_at"))
            procedural.add_instruction(
                content=item["content"],
                category=item.get("category", "general"),
                importance=item.get("importance", 0.9),
                expires_at=expires,
            )
            logger.info("Procedural stored: %s", item["content"][:60])

        elif store in ("episodic", "semantic"):
            await long_term.store(
                text=item["content"],
                memory_type=store,
                category=item.get("category", "event"),
                importance=item.get("importance", 0.5),
            )
            logger.info("Long-term stored (%s): %s", store, item["content"][:60])
