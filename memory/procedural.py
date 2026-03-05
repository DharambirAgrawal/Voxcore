"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/procedural.py                               ║
║           TIER 4 — STANDING INSTRUCTIONS & REMINDERS IN SQLITE                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Stores standing instructions, pending reminders, and relay messages.
    SQLite-backed for guaranteed retrieval — instructions cannot miss
    due to vector similarity threshold (unlike ChromaDB).

CATEGORY VALUES:
    general | communication | topic | reminder | relay

    relay = message from third party, surfaces proactively at session start.
"""

import logging
import os
import sqlite3
import time
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("Procedural")


class ProceduralStore:
    """SQLite-backed storage for standing instructions and reminders."""

    def __init__(self, config: dict) -> None:
        db_path = config.get("memory", {}).get(
            "procedural_db", "data/memory/procedural.db"
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        logger.info("ProceduralStore initialized at %s", db_path)

    async def initialize(self) -> None:
        """No-op async initializer — table creation is done in __init__."""
        pass

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS instructions (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                category    TEXT NOT NULL DEFAULT 'general',
                importance  REAL DEFAULT 0.9,
                source      TEXT DEFAULT 'user_explicit',
                active      INTEGER DEFAULT 1,
                created_at  INTEGER NOT NULL,
                expires_at  INTEGER DEFAULT NULL
            )
        """)
        self._conn.commit()

    def add_instruction(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.9,
        source: str = "user_explicit",
        expires_at: Optional[int] = None,
    ) -> str:
        """Add a new instruction. Returns its ID."""
        inst_id = f"inst_{int(time.time())}_{uuid4().hex[:6]}"
        now = int(time.time())
        self._conn.execute(
            """INSERT INTO instructions
               (id, content, category, importance, source, active, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
            (inst_id, content, category, importance, source, now, expires_at),
        )
        self._conn.commit()
        logger.info("Added instruction %s [%s]: %s", inst_id, category, content[:60])
        return inst_id

    def get_active(self, category: Optional[str] = None) -> list[dict]:
        """Return active instructions, optionally filtered by category.

        Excludes expired instructions. Ordered by importance DESC.
        """
        now = int(time.time())
        if category:
            rows = self._conn.execute(
                """SELECT * FROM instructions
                   WHERE active = 1 AND category = ?
                   AND (expires_at IS NULL OR expires_at > ?)
                   ORDER BY importance DESC""",
                (category, now),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM instructions
                   WHERE active = 1
                   AND (expires_at IS NULL OR expires_at > ?)
                   ORDER BY importance DESC""",
                (now,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_relays(self) -> list[dict]:
        """Return active relay messages (category='relay')."""
        return self.get_active(category="relay")

    def complete(self, instruction_id: str, outcome: str = "") -> bool:
        """Mark an instruction as inactive. Returns True if found."""
        cursor = self._conn.execute(
            "UPDATE instructions SET active = 0 WHERE id = ?",
            (instruction_id,),
        )
        self._conn.commit()
        if cursor.rowcount > 0:
            logger.info("Completed instruction %s: %s", instruction_id, outcome)
            return True
        return False

    def get_active_count(self) -> int:
        """Quick count of active instructions."""
        now = int(time.time())
        row = self._conn.execute(
            """SELECT COUNT(*) as cnt FROM instructions
               WHERE active = 1
               AND (expires_at IS NULL OR expires_at > ?)""",
            (now,),
        ).fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
