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


"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — interfaces/cli.py                                 ║
║              CLI INTERFACE — TERMINAL DEBUG & DEVELOPMENT INTERFACE              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import asyncio
import sys
from datetime import datetime
from typing import Optional

from core.event_bus import EventBus, EventType
from core.session import Session, TurnState

# ─── ANSI color codes ───────────────────────────────────────────────────────────

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
    TurnState.LISTENING:   "GREEN",
    TurnState.THINKING:    "YELLOW",
    TurnState.SPEAKING:    "CYAN",
    TurnState.INTERRUPTED: "RED",
}

_BANNER = f"""{COLORS['CYAN']}{COLORS['BOLD']}
╔══════════════════════════════════════════╗
║            VOXCORE  CLI  v0.1            ║
║       Terminal Debug & Dev Interface     ║
╚══════════════════════════════════════════╝{COLORS['RESET']}
"""


class CLIInterface:
    """Terminal debug interface with colored output and optional text input."""

    def __init__(
        self,
        event_bus: EventBus,
        session: Session,
        text_mode: bool = False,
        verbose: bool = True,
    ) -> None:
        self._bus = event_bus
        self._session = session
        self._text_mode = text_mode
        self._verbose = verbose
        self._start_time = datetime.now()
        self._event_count = 0
        self._logger = logging.getLogger("CLI")

    # ── public entry point ───────────────────────────────────────────────────

    async def run(self) -> None:
        # 1. Startup banner
        print(_BANNER)

        # 2. Subscribe to key events
        self._bus.subscribe(EventType.STATE_CHANGED, self._on_state_change)
        self._bus.subscribe(EventType.TRANSCRIPT_READY, self._on_transcript)
        self._bus.subscribe(EventType.LLM_STREAM_DONE, self._on_llm_sentence)
        if self._verbose:
            self._bus.subscribe(EventType.LLM_SPEECH_TOKEN, self._on_llm_token)
            self._bus.subscribe(EventType.TTS_CHUNK_READY, self._on_tts_chunk)
        self._bus.subscribe(EventType.SAFETY_FLAGGED, self._on_safety_flag)
        self._bus.subscribe(EventType.INTERRUPT_DETECTED, self._on_interrupt)
        self._bus.subscribe(EventType.BACKCHANNEL_FIRE, self._on_backchannel)
        self._bus.subscribe(EventType.AGENT_JSON_OUT, self._on_agent_action)
        self._bus.subscribe(EventType.TOOL_RESULT_READY, self._on_tool_result)

        # 3. Optional text-input loop
        if self._text_mode:
            asyncio.create_task(self._text_input_loop())
            print(f"{COLORS['BOLD']}Text mode. Type messages:{COLORS['RESET']}")
        else:
            print(f"{COLORS['BOLD']}Ready. Listening...{COLORS['RESET']}")

        # 4. Block forever
        await asyncio.Event().wait()

    # ── text input loop ──────────────────────────────────────────────────────

    async def _text_input_loop(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:  # EOF
                    break
                line = line.strip()
                if not line:
                    continue
                if line in ("/quit", "/exit"):
                    break
                if line == "/status":
                    self._print_status()
                    continue
                if line == "/history":
                    self._print_history()
                    continue
                if line.startswith("/"):
                    self._handle_command(line)
                    continue
                # Normal text → publish as transcript
                await self._bus.publish(
                    EventType.TRANSCRIPT_READY,
                    {"text": line, "is_final": True, "source": "cli"},
                )
        except Exception as exc:
            self._logger.error("Text input loop error: %s", exc)
        print(f"{COLORS['GRAY']}CLI input ended{COLORS['RESET']}")

    # ── event handlers ───────────────────────────────────────────────────────

    async def _on_state_change(self, event) -> None:
        self._event_count += 1
        old = event.data.get("old_state", "?")
        new = event.data.get("new_state", event.data.get("state", "?"))
        color_name = STATE_COLORS.get(new, "RESET")
        color = COLORS.get(color_name, COLORS["RESET"])
        old_val = old.value if hasattr(old, "value") else str(old)
        new_val = new.value if hasattr(new, "value") else str(new)
        print(f"{color}[STATE] {old_val} → {new_val}{COLORS['RESET']}")

    async def _on_transcript(self, event) -> None:
        self._event_count += 1
        text = event.data["text"]
        is_final = event.data.get("is_final", False)
        if is_final:
            print(f"{COLORS['GREEN']}[USER] {text}{COLORS['RESET']}")
        else:
            sys.stdout.write(f"\r{COLORS['GRAY']}[user] {text}...{COLORS['RESET']}")
            sys.stdout.flush()

    async def _on_llm_sentence(self, event) -> None:
        self._event_count += 1
        print(f"{COLORS['CYAN']}[AI] {event.data['text']}{COLORS['RESET']}")

    async def _on_llm_token(self, event) -> None:
        self._event_count += 1
        sys.stdout.write(event.data["token"])
        sys.stdout.flush()

    async def _on_tts_chunk(self, event) -> None:
        self._event_count += 1
        data = event.data
        chunk_len = len(data) if isinstance(data, (bytes, bytearray)) else len(str(data))
        print(f"{COLORS['GRAY']}[TTS] chunk {chunk_len} bytes{COLORS['RESET']}")

    async def _on_safety_flag(self, event) -> None:
        self._event_count += 1
        direction = event.data["direction"]
        categories = event.data["categories"]
        print(f"{COLORS['RED']}[SAFETY] {direction} flagged: {categories}{COLORS['RESET']}")

    async def _on_interrupt(self, event) -> None:
        self._event_count += 1
        print(f"{COLORS['YELLOW']}[INTERRUPT] User interrupted AI{COLORS['RESET']}")

    async def _on_backchannel(self, event) -> None:
        self._event_count += 1
        print(f"{COLORS['MAGENTA']}[BACKCHANNEL] {event.data['category']}{COLORS['RESET']}")

    async def _on_agent_action(self, event) -> None:
        self._event_count += 1
        tool = event.data["tool"]
        params = event.data["params"]
        print(f"{COLORS['BLUE']}[AGENT] Tool: {tool} | Params: {params}{COLORS['RESET']}")

    async def _on_tool_result(self, event) -> None:
        self._event_count += 1
        tool = event.data["tool"]
        result_preview = str(event.data["result"])[:200]
        print(f"{COLORS['BLUE']}[TOOL RESULT] {tool}: {result_preview}{COLORS['RESET']}")

    # ── helper commands ──────────────────────────────────────────────────────

    def _print_status(self) -> None:
        uptime = datetime.now() - self._start_time
        state = self._session.state if hasattr(self._session, "state") else "unknown"
        stm_size = (
            len(self._session.short_term_memory)
            if hasattr(self._session, "short_term_memory")
            else 0
        )
        print(
            f"{COLORS['BOLD']}── STATUS ──────────────────{COLORS['RESET']}\n"
            f"  State:        {state}\n"
            f"  Uptime:       {uptime}\n"
            f"  Events seen:  {self._event_count}\n"
            f"  STM entries:  {stm_size}\n"
            f"{COLORS['BOLD']}────────────────────────────{COLORS['RESET']}"
        )

    def _print_history(self) -> None:
        history = (
            self._session.history[-10:]
            if hasattr(self._session, "history")
            else []
        )
        if not history:
            print(f"{COLORS['GRAY']}(no history){COLORS['RESET']}")
            return
        print(f"{COLORS['BOLD']}── LAST {len(history)} TURNS ──{COLORS['RESET']}")
        for i, turn in enumerate(history):
            role = turn.get("role", "?")
            text = turn.get("text", turn.get("content", ""))[:120]
            print(f"  {i + 1}. [{role}] {text}")
        print(f"{COLORS['BOLD']}───────────────────────────{COLORS['RESET']}")

    def _handle_command(self, command: str) -> None:
        cmd = command.lower().strip()
        if cmd == "/status":
            self._print_status()
        elif cmd == "/history":
            self._print_history()
        elif cmd == "/clear":
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
        elif cmd in ("/help", "/?"):
            print(
                f"{COLORS['BOLD']}Available commands:{COLORS['RESET']}\n"
                "  /status   — Show current system state\n"
                "  /history  — Show last 10 conversation turns\n"
                "  /clear    — Clear terminal\n"
                "  /help     — This message\n"
                "  /quit     — Exit CLI"
            )
        elif cmd in ("/quit", "/exit"):
            pass  # handled in _text_input_loop
        else:
            print(f"{COLORS['YELLOW']}Unknown command: {command}. Try /help{COLORS['RESET']}")