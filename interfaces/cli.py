"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — interfaces/cli.py                                 ║
║              CLI INTERFACE — TERMINAL DEBUG & DEVELOPMENT INTERFACE              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Provides a terminal-based interface for debugging and development.
    Displays real-time system state: current TurnState, transcript text,
    LLM output, TTS chunks, safety flags, and event flow.
    Subscribes to all major EventBus events and prints formatted output.

    Also supports text-only mode (no mic) where the developer can type
    messages to test the LLM pipeline without audio hardware.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging
import asyncio
import sys
from datetime import datetime
from typing import Optional

from core.event_bus import EventBus, EventType
from core.session import Session, TurnState

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

# ANSI color codes for terminal output
COLORS = {
    "RESET":   "\033[0m",
    "RED":     "\033[91m",
    "GREEN":   "\033[92m",
    "YELLOW":  "\033[93m",
    "BLUE":    "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN":    "\033[96m",
    "GRAY":    "\033[90m",
    "BOLD":    "\033[1m",
}

STATE_COLORS = {
    TurnState.LISTENING:    "GREEN",
    TurnState.THINKING:     "YELLOW",
    TurnState.SPEAKING:     "CYAN",
    TurnState.INTERRUPTED:  "RED",
}

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: CLIInterface
──────────────────────────────────────────────────────────────────────────────────
    Terminal debug interface with colored output and optional text input.

    CONSTRUCTOR: __init__(self, event_bus: EventBus, session: Session,
                          text_mode: bool = False, verbose: bool = True)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - event_bus: EventBus — For subscribing to all events
            - session: Session — For reading current state
            - text_mode: bool — If True, enables stdin text input (bypasses mic)
            - verbose: bool — If True, show all events; if False, only key events
        INITIALIZES:
            self._bus = event_bus
            self._session = session
            self._text_mode = text_mode
            self._verbose = verbose
            self._start_time = datetime.now()
            self._event_count = 0
            self._logger = logging.getLogger("CLI")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def run(self) -> None
        INPUTS: None
        OUTPUT: None (runs forever)
        WHAT IT DOES:
            1. Print startup banner with ASCII art
            2. Subscribe to key events:
               - EventType.STATE_CHANGE → self._on_state_change
               - EventType.TRANSCRIPT → self._on_transcript
               - EventType.LLM_SENTENCE → self._on_llm_sentence
               - EventType.LLM_TOKEN → self._on_llm_token (verbose only)
               - EventType.TTS_CHUNK → self._on_tts_chunk (verbose only)
               - EventType.SAFETY_FLAG → self._on_safety_flag
               - EventType.INTERRUPT → self._on_interrupt
               - EventType.BACKCHANNEL_CUE → self._on_backchannel
               - EventType.AGENT_ACTION → self._on_agent_action
               - EventType.TOOL_RESULT → self._on_tool_result
            3. If self._text_mode:
               asyncio.create_task(self._text_input_loop())
            4. Print "Ready. Listening..." (or "Text mode. Type messages:")
            5. Block with asyncio.Event().wait()

    async def _text_input_loop(self) -> None
        INPUTS: None
        OUTPUT: None (runs until EOF)
        WHAT IT DOES:
            1. Loop reading lines from stdin (using asyncio.get_event_loop().run_in_executor)
            2. For each line:
               line = line.strip()
               If line == "/quit" or line == "/exit": break
               If line == "/status": self._print_status()
               If line == "/history": self._print_history()
               If line.startswith("/"): self._handle_command(line)
               Else:
                   Publish EventType.TRANSCRIPT with data={
                       "text": line,
                       "is_final": True,
                       "source": "cli"
                   }
            3. Print "CLI input ended"

    async def _on_state_change(self, event) -> None
        INPUTS: event with data["old_state"], data["new_state"]
        OUTPUT: None
        WHAT IT DOES:
            1. old = event.data["old_state"]
            2. new = event.data["new_state"]
            3. color = COLORS[STATE_COLORS.get(new, "RESET")]
            4. Print f"{color}[STATE] {old.value} → {new.value}{COLORS['RESET']}"

    async def _on_transcript(self, event) -> None
        INPUTS: event with data["text"], data["is_final"]
        OUTPUT: None
        WHAT IT DOES:
            1. text = event.data["text"]
            2. is_final = event.data.get("is_final", False)
            3. If is_final:
               Print f"{COLORS['GREEN']}[USER] {text}{COLORS['RESET']}"
            4. Else (partial):
               Print f"{COLORS['GRAY']}[user] {text}...{COLORS['RESET']}" (overwrite line)

    async def _on_llm_sentence(self, event) -> None
        INPUTS: event with data["text"]
        OUTPUT: None
        WHAT IT DOES:
            Print f"{COLORS['CYAN']}[AI] {event.data['text']}{COLORS['RESET']}"

    async def _on_llm_token(self, event) -> None
        INPUTS: event with data["token"]
        OUTPUT: None
        WHAT IT DOES:
            (verbose only) Print token without newline: sys.stdout.write(token)

    async def _on_tts_chunk(self, event) -> None
        INPUTS: event with data (audio bytes info)
        OUTPUT: None
        WHAT IT DOES:
            (verbose only) Print f"{COLORS['GRAY']}[TTS] chunk {len(data)} bytes{COLORS['RESET']}"

    async def _on_safety_flag(self, event) -> None
        INPUTS: event with data["direction"], data["categories"]
        OUTPUT: None
        WHAT IT DOES:
            Print f"{COLORS['RED']}[SAFETY] {event.data['direction']} flagged: {event.data['categories']}{COLORS['RESET']}"

    async def _on_interrupt(self, event) -> None
        INPUTS: event
        OUTPUT: None
        WHAT IT DOES:
            Print f"{COLORS['YELLOW']}[INTERRUPT] User interrupted AI{COLORS['RESET']}"

    async def _on_backchannel(self, event) -> None
        INPUTS: event with data["category"]
        OUTPUT: None
        WHAT IT DOES:
            Print f"{COLORS['MAGENTA']}[BACKCHANNEL] {event.data['category']}{COLORS['RESET']}"

    async def _on_agent_action(self, event) -> None
        INPUTS: event with data["tool"], data["params"]
        OUTPUT: None
        WHAT IT DOES:
            Print f"{COLORS['BLUE']}[AGENT] Tool: {event.data['tool']} | Params: {event.data['params']}{COLORS['RESET']}"

    async def _on_tool_result(self, event) -> None
        INPUTS: event with data["tool"], data["result"]
        OUTPUT: None
        WHAT IT DOES:
            result_preview = event.data["result"][:200]
            Print f"{COLORS['BLUE']}[TOOL RESULT] {event.data['tool']}: {result_preview}{COLORS['RESET']}"

    def _print_status(self) -> None
        INPUTS: None
        OUTPUT: None (prints to terminal)
        WHAT IT DOES:
            Print current state, uptime, event count, short-term memory size

    def _print_history(self) -> None
        INPUTS: None
        OUTPUT: None (prints to terminal)
        WHAT IT DOES:
            Print last 10 turns from session history

    def _handle_command(self, command: str) -> None
        INPUTS: command: str (starts with /)
        OUTPUT: None
        WHAT IT DOES:
            Handle CLI commands: /status, /history, /help, /clear, /quit

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - CLIInterface   (class)
═══════════════════════════════════════════════════════════════════════════════════
"""
