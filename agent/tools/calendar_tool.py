"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — agent/tools/calendar_tool.py                      ║
║              CALENDAR TOOL — CREATE/LIST/DELETE REMINDERS & EVENTS              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Provides local calendar/reminder support for the AI agent.
    Uses SQLite for local storage of events and reminders.
    The LLM can call this tool to set reminders, list upcoming events,
    or cancel previously created events.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging                              # Module logger
import aiosqlite                            # Async SQLite access
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent.tools.base_tool import BaseTool

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

DEFAULT_DB_PATH = Path("data/calendar.db")

CREATE_TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        event_time TEXT NOT NULL,   -- ISO 8601 format
        created_at TEXT NOT NULL,   -- When the event was created
        reminder_minutes INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active'   -- 'active' | 'cancelled'
    )
'''

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: CalendarTool(BaseTool)
──────────────────────────────────────────────────────────────────────────────────
    Manages local calendar events & reminders stored in SQLite.

    CLASS ATTRIBUTES:
        name = "calendar"
        description = "Create, list, or delete calendar events and reminders"
        required_params = ["action"]  # "create", "list", "delete"
        optional_params = ["title", "description", "event_time", "event_id",
                           "hours_ahead"]

    CONSTRUCTOR: __init__(self, db_path: str | Path = None)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - db_path: str | Path — Path to SQLite file (default: data/calendar.db)
        INITIALIZES:
            super().__init__()
            self.db_path: Path = Path(db_path) if db_path else DEFAULT_DB_PATH
            self._initialized: bool = False
            self._logger = logging.getLogger("Tool.calendar")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def _ensure_db(self) -> None
        INPUTS: None
        OUTPUT: None
        WHAT IT DOES:
            1. Ensure parent directory exists: self.db_path.parent.mkdir(parents=True, exist_ok=True)
            2. async with aiosqlite.connect(self.db_path) as db:
               await db.execute(CREATE_TABLE_SQL)
               await db.commit()
            3. self._initialized = True

    async def execute(self, params: dict) -> str
        INPUTS:
            - params: dict containing:
                - action: str — "create", "list", or "delete" (REQUIRED)
                - title: str — Event title (REQUIRED for "create")
                - description: str — Event description (optional for "create")
                - event_time: str — ISO 8601 datetime (REQUIRED for "create")
                - event_id: int — Event ID to delete (REQUIRED for "delete")
                - hours_ahead: int — How many hours ahead to list (default 24)
        OUTPUT:
            - str — Human-readable result
        WHAT IT DOES:
            1. If not self._initialized: await self._ensure_db()
            2. action = params.get("action")
            3. If action == "create": return await self._create_event(params)
            4. If action == "list": return await self._list_events(params)
            5. If action == "delete": return await self._delete_event(params)
            6. Else: return f"Unknown action: {action}. Use 'create', 'list', or 'delete'."

    async def _create_event(self, params: dict) -> str
        INPUTS:
            - params: dict with title, event_time, optionally description
        OUTPUT:
            - str — Confirmation with event ID
        WHAT IT DOES:
            1. title = params.get("title", "")
               If not title: return "Error: 'title' required"
            2. event_time = params.get("event_time", "")
               If not event_time: return "Error: 'event_time' required (ISO 8601)"
            3. Validate event_time is valid ISO 8601, catch ValueError
            4. description = params.get("description", "")
            5. async with aiosqlite.connect(self.db_path) as db:
               cursor = await db.execute(
                   "INSERT INTO events (title, description, event_time, created_at) VALUES (?, ?, ?, ?)",
                   (title, description, event_time, datetime.utcnow().isoformat())
               )
               await db.commit()
               event_id = cursor.lastrowid
            6. Return f"Event #{event_id} created: '{title}' at {event_time}"

    async def _list_events(self, params: dict) -> str
        INPUTS:
            - params: dict with optional hours_ahead (default 24)
        OUTPUT:
            - str — Formatted list of upcoming events or "No upcoming events"
        WHAT IT DOES:
            1. hours_ahead = params.get("hours_ahead", 24)
            2. now = datetime.utcnow().isoformat()
            3. cutoff = (datetime.utcnow() + timedelta(hours=hours_ahead)).isoformat()
            4. async with aiosqlite.connect(self.db_path) as db:
               db.row_factory = aiosqlite.Row
               cursor = await db.execute(
                   "SELECT * FROM events WHERE event_time BETWEEN ? AND ? AND status='active' ORDER BY event_time",
                   (now, cutoff)
               )
               rows = await cursor.fetchall()
            5. If not rows: return f"No upcoming events in the next {hours_ahead} hours."
            6. Format each row: "#{id} - {title} at {event_time}"
            7. Return formatted list

    async def _delete_event(self, params: dict) -> str
        INPUTS:
            - params: dict with event_id (int)
        OUTPUT:
            - str — Confirmation or error
        WHAT IT DOES:
            1. event_id = params.get("event_id")
               If not event_id: return "Error: 'event_id' required"
            2. async with aiosqlite.connect(self.db_path) as db:
               cursor = await db.execute(
                   "UPDATE events SET status='cancelled' WHERE id=? AND status='active'",
                   (event_id,)
               )
               await db.commit()
               if cursor.rowcount == 0: return f"Event #{event_id} not found or already cancelled."
            3. Return f"Event #{event_id} cancelled."

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - CalendarTool   (class)
═══════════════════════════════════════════════════════════════════════════════════
"""
