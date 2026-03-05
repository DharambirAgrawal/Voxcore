"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — memory/fact_store.py                               ║
║           TIER 3 — PERMANENT USER FACTS IN SQLITE                               ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Stores hard user facts: name, city, preferences, relationships.
    SQLite-backed for permanent persistence with ~2ms access.
    Facts are keyed with snake_case conventions and support
    confidence scoring and source tracking.

KEY NAMING CONVENTION:
    user_name, user_city, user_state, user_occupation, user_employer
    pref_response_length, pref_technical_depth, pref_communication_style
    research_field_primary, project_current, project_status
    relationship_{name}, schedule_{label}
"""

import logging
import os
import sqlite3
import time
from typing import Optional

logger = logging.getLogger("FactStore")


class FactStore:
    """SQLite-backed permanent storage for user facts and preferences."""

    def __init__(self, config: dict) -> None:
        db_path = config.get("memory", {}).get(
            "fact_store_db", "data/memory/facts.db"
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        logger.info("FactStore initialized at %s", db_path)

    async def initialize(self) -> None:
        """No-op async initializer — table creation is done in __init__."""
        pass

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                key             TEXT PRIMARY KEY,
                value           TEXT NOT NULL,
                confidence      REAL DEFAULT 0.7,
                source          TEXT DEFAULT 'extracted',
                last_mentioned  INTEGER DEFAULT 0,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def set(
        self,
        key: str,
        value: str,
        confidence: float = 0.7,
        source: str = "extracted",
    ) -> None:
        """Insert or update a fact. Preserves created_at on update."""
        now = int(time.time())
        existing = self._conn.execute(
            "SELECT created_at FROM facts WHERE key = ?", (key,)
        ).fetchone()

        if existing:
            self._conn.execute(
                """UPDATE facts
                   SET value = ?, confidence = ?, source = ?,
                       last_mentioned = ?, updated_at = ?
                   WHERE key = ?""",
                (value, confidence, source, now, now, key),
            )
        else:
            self._conn.execute(
                """INSERT INTO facts
                   (key, value, confidence, source, last_mentioned, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (key, value, confidence, source, now, now, now),
            )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        """Return value or None. Never raises."""
        row = self._conn.execute(
            "SELECT value FROM facts WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def get_with_meta(self, key: str) -> Optional[dict]:
        """Return full row as dict, or None."""
        row = self._conn.execute(
            "SELECT * FROM facts WHERE key = ?", (key,)
        ).fetchone()
        return dict(row) if row else None

    def get_all(self) -> dict[str, str]:
        """Return {key: value} for all facts."""
        rows = self._conn.execute("SELECT key, value FROM facts").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def get_by_prefix(self, prefix: str) -> dict[str, str]:
        """Return {key: value} for keys matching prefix%."""
        rows = self._conn.execute(
            "SELECT key, value FROM facts WHERE key LIKE ?",
            (f"{prefix}%",),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def get_by_prefix_with_meta(self, prefix: str) -> list[dict]:
        """Return full rows for keys matching prefix%."""
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE key LIKE ? ORDER BY last_mentioned DESC",
            (f"{prefix}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def touch(self, key: str) -> None:
        """Update last_mentioned timestamp for a fact."""
        now = int(time.time())
        self._conn.execute(
            "UPDATE facts SET last_mentioned = ? WHERE key = ?", (now, key)
        )
        self._conn.commit()

    def delete(self, key: str) -> bool:
        """Delete a fact. Returns True if deleted, False if not found."""
        cursor = self._conn.execute("DELETE FROM facts WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
