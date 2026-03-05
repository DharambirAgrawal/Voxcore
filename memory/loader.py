"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/loader.py                                   ║
║           SESSION-START MEMORY LOADER — ASSEMBLES CONTEXT BLOCK                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Runs once at session start. Assembles a formatted text block from
    fact_store and procedural stores. Injected into the system prompt
    by prompt_builder.set_memory_block().

    No LLM involved — pure data assembly from SQLite stores.
    350 token hard cap. Never drops relays or standing instructions.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("MemoryLoader")

# Core identity keys — always loaded
CORE_IDENTITY_KEYS = [
    "user_name", "user_city", "user_state", "user_occupation",
    "user_employer", "user_timezone",
]

CORE_PREF_KEYS = [
    "pref_response_length", "pref_technical_depth",
    "pref_communication_style",
]

# Display labels for user context
KEY_LABELS = {
    "user_name": "Name",
    "user_city": "Location",
    "user_state": "State",
    "user_occupation": "Occupation",
    "user_employer": "Employer",
    "user_timezone": "Timezone",
    "pref_response_length": "Preferred length",
    "pref_technical_depth": "Technical depth",
    "pref_communication_style": "Communication",
}


class MemoryLoader:
    """Assembles a session-start memory block from SQLite stores."""

    def __init__(self, fact_store, procedural, long_term, config: dict) -> None:
        self._fact_store = fact_store
        self._procedural = procedural
        self._long_term = long_term

        mem_cfg = config.get("memory", {})
        self._max_tokens: int = mem_cfg.get("loader_max_tokens", 350)
        self._relationship_days: int = mem_cfg.get("relationship_recency_days", 14)
        self._reminder_days: int = mem_cfg.get("reminder_window_days", 7)

        self._cached_summary: str = ""

    async def build_session_block(self) -> str:
        """Build the formatted memory block for system prompt injection.

        Returns a string ready for prompt_builder.set_memory_block().
        """
        sections: list[str] = []

        # 1. Pending relays — NEVER dropped
        relays = self._procedural.get_pending_relays()
        if relays:
            lines = [f"- {r['content']}" for r in relays]
            sections.append("[PENDING MESSAGES]\n" + "\n".join(lines))

        # 2. Active instructions (non-relay) — NEVER dropped
        all_active = self._procedural.get_active()
        instructions = [i for i in all_active if i["category"] != "relay"]
        if instructions:
            lines = [f"- {i['content']}" for i in instructions]
            sections.append("[STANDING INSTRUCTIONS]\n" + "\n".join(lines))

        # 3. Core identity facts
        user_ctx_lines = []
        for key in CORE_IDENTITY_KEYS + CORE_PREF_KEYS:
            value = self._fact_store.get(key)
            if value:
                label = KEY_LABELS.get(key, key.replace("_", " ").title())
                user_ctx_lines.append(f"{label}: {value}")

        if user_ctx_lines:
            sections.append("[USER CONTEXT]\n" + "\n".join(user_ctx_lines))

        # 4. Upcoming reminders (within reminder_window_days)
        now = int(time.time())
        window_end = now + (self._reminder_days * 86400)
        reminders = self._procedural.get_active(category="reminder")
        upcoming = [
            r for r in reminders
            if r.get("expires_at") and r["expires_at"] <= window_end
        ]
        if upcoming:
            lines = [f"- {u['content']}" for u in upcoming[:5]]
            sections.append("[UPCOMING]\n" + "\n".join(lines))

        # 5. Recent relationships (last_mentioned within relationship_days)
        cutoff = now - (self._relationship_days * 86400)
        rel_entries = self._fact_store.get_by_prefix_with_meta("relationship_")
        recent_rels = [
            r for r in rel_entries
            if r.get("last_mentioned", 0) > cutoff
        ][:5]  # max 5
        if recent_rels:
            lines = []
            for r in recent_rels:
                name = r["key"].replace("relationship_", "").replace("_", " ").title()
                lines.append(f"- {name}: {r['value']}")
            sections.append("[RECENT RELATIONSHIPS]\n" + "\n".join(lines))

        # Assemble and enforce budget
        block = "\n\n".join(sections)
        block = self._enforce_budget(block, sections)

        self._cached_summary = block
        logger.info(
            "Session memory block built: ~%d tokens (%d chars)",
            len(block) // 4, len(block),
        )
        return block

    def _enforce_budget(self, block: str, sections: list[str]) -> str:
        """Trim block to fit within max_tokens.

        Drop order (spec-defined):
        1. Drop [RECENT RELATIONSHIPS] oldest entries first
        2. Drop [UPCOMING] entries lowest importance first
        3. Truncate low-confidence pref_* from [USER CONTEXT]
        4. NEVER drop [PENDING MESSAGES] or [STANDING INSTRUCTIONS]
        """
        est_tokens = len(block) // 4
        if est_tokens <= self._max_tokens:
            return block

        # Try dropping relationships section
        trimmed = [s for s in sections if not s.startswith("[RECENT RELATIONSHIPS]")]
        block = "\n\n".join(trimmed)
        if len(block) // 4 <= self._max_tokens:
            return block

        # Try dropping upcoming section
        trimmed = [
            s for s in trimmed
            if not s.startswith("[UPCOMING]")
        ]
        block = "\n\n".join(trimmed)
        if len(block) // 4 <= self._max_tokens:
            return block

        # Truncate user context pref_* lines
        final_sections = []
        for s in trimmed:
            if s.startswith("[USER CONTEXT]"):
                lines = s.split("\n")
                kept = [lines[0]]  # header
                for line in lines[1:]:
                    # Keep core identity, trim preferences
                    if any(line.startswith(KEY_LABELS.get(k, ""))
                           for k in CORE_IDENTITY_KEYS if self._fact_store.get(k)):
                        kept.append(line)
                final_sections.append("\n".join(kept))
            else:
                final_sections.append(s)

        return "\n\n".join(final_sections)

    def get_stored_summary(self) -> str:
        """Return the cached session memory block."""
        return self._cached_summary
