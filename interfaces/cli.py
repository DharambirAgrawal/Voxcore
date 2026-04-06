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
        self._bus.subscribe(EventType.LLM_SPEECH_TOKEN, self._on_llm_sentence)
        if self._verbose:
            self._bus.subscribe(EventType.TTS_CHUNK_READY, self._on_tts_chunk)
        self._bus.subscribe(EventType.SAFETY_FLAGGED, self._on_safety_flag)
        self._bus.subscribe(EventType.INTERRUPT_DETECTED, self._on_interrupt)
        self._bus.subscribe(EventType.BACKCHANNEL_FIRE, self._on_backchannel)
        self._bus.subscribe(EventType.AGENT_JSON_OUT, self._on_agent_action)
        self._bus.subscribe(EventType.TOOL_RESULT_READY, self._on_tool_result)

        # 3. Text-input loop — always active so you can type while mic runs
        asyncio.create_task(self._text_input_loop())
        if self._text_mode:
            print(f"{COLORS['BOLD']}Text mode. Type messages (Enter to send, /help for commands):{COLORS['RESET']}")
        else:
            print(f"{COLORS['BOLD']}Ready. Listening... (type text + Enter to inject, /help for commands){COLORS['RESET']}")

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
                    await self._handle_command(line)
                    continue
                # Normal text → publish as transcript (as if user spoke it)
                print(f"{COLORS['GREEN']}[CLI → user] {line}{COLORS['RESET']}")
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
        action = event.data.get("action", event.data.get("tool", "?"))
        params = event.data.get("params", {})
        print(f"{COLORS['BLUE']}[AGENT] Action: {action} | Params: {params}{COLORS['RESET']}")

    async def _on_tool_result(self, event) -> None:
        self._event_count += 1
        tool = event.data.get("tool", event.data.get("action", "?"))
        res_str = str(event.data.get("result", ""))
        result_preview = res_str[:200]
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
            role = getattr(turn, "role", "?")
            text = getattr(turn, "content", "")[:120]
            print(f"  {i + 1}. [{role}] {text}")
        print(f"{COLORS['BOLD']}───────────────────────────{COLORS['RESET']}")

    async def _handle_command(self, command: str) -> None:
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/status":
            self._print_status()
        elif cmd == "/history":
            self._print_history()
        elif cmd == "/clear":
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
        elif cmd == "/fetch":
            # Directly test article_fetch with a URL
            if not arg:
                print(f"{COLORS['YELLOW']}Usage: /fetch <url> [instruction]{COLORS['RESET']}")
                print(f"{COLORS['GRAY']}  Example: /fetch https://example.com/article{COLORS['RESET']}")
                print(f"{COLORS['GRAY']}  Example: /fetch https://example.com/article summarize briefly{COLORS['RESET']}")
                return
            # Split URL from optional instruction
            url_parts = arg.split(None, 1)
            url = url_parts[0]
            instruction = url_parts[1] if len(url_parts) > 1 else "summarize the article naturally in about 150 spoken words"
            fake_user_msg = f"Can you read this article for me? {url}"
            print(f"{COLORS['GREEN']}[CLI → user] {fake_user_msg}{COLORS['RESET']}")
            print(f"{COLORS['GRAY']}  (instruction: {instruction}){COLORS['RESET']}")
            await self._bus.publish(
                EventType.TRANSCRIPT_READY,
                {"text": fake_user_msg, "is_final": True, "source": "cli"},
            )
        elif cmd == "/search":
            # Directly test web_search with a query
            if not arg:
                print(f"{COLORS['YELLOW']}Usage: /search <query>{COLORS['RESET']}")
                print(f"{COLORS['GRAY']}  Example: /search latest news today{COLORS['RESET']}")
                return
            print(f"{COLORS['GREEN']}[CLI → user] {arg}{COLORS['RESET']}")
            await self._bus.publish(
                EventType.TRANSCRIPT_READY,
                {"text": arg, "is_final": True, "source": "cli"},
            )
        elif cmd in ("/help", "/?"):
            print(
                f"{COLORS['BOLD']}Available commands:{COLORS['RESET']}\n"
                "  /status    — Show current system state\n"
                "  /history   — Show last 10 conversation turns\n"
                "  /fetch URL [instruction] — Test article_fetch with a URL\n"
                "  /search QUERY — Send a search query as user input\n"
                "  /clear     — Clear terminal\n"
                "  /help      — This message\n"
                "  /quit      — Exit CLI"
            )
        elif cmd in ("/quit", "/exit"):
            pass  # handled in _text_input_loop
        else:
            print(f"{COLORS['YELLOW']}Unknown command: {command}. Try /help{COLORS['RESET']}")