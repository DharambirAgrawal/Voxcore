"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — brain/prompt_builder.py                          ║
║       ASSEMBLES THE FULL MESSAGE ARRAY FOR EVERY LLM CALL                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Builds the complete messages array that gets sent to the LLM on every turn.
    This is where the persona, memory, context injections, and conversation
    history all come together into a single coherent prompt.

    Message structure (in order):
    1. System message — persona definition, datetime, instructions for emotion
       tags, agent output format, response length constraints
    2. Compressed old memory summary (if history > 20 turns)
    3. Last 20 turns of conversation history
    4. Any queued text-in injections as [CONTEXT] blocks (flushed on read)
    5. Current user message

    Total token budget: ≤5,000 tokens for llama-3.1-8b to maintain speed.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging                          # Module logger
import time                             # Current datetime for system prompt
from datetime import datetime           # Formatted datetime string
from typing import Optional

from core.session import Session, Turn, TextInEntry

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

MAX_PROMPT_TOKENS = 5000               # Hard cap for fast brain context window
MAX_HISTORY_TURNS = 20                 # Rolling window of recent history
MAX_TEXT_IN_CHARS = 2000               # Max total characters for text-in injections
MAX_SUMMARY_CHARS = 1000               # Max characters for compressed memory summary

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: PromptBuilder
──────────────────────────────────────────────────────────────────────────────────
    Constructs the LLM message array from session state.

    CONSTRUCTOR: __init__(self, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - config: dict — The "persona" section from config.yaml:
                - name: str ("Aria")
                - system_prompt: str (the full persona prompt)
                - voice: str ("tara")
                - language: str ("en")
        
        INITIALIZES:
            self.persona_name: str     = config.get("name", "Aria")
            self.system_prompt: str    = config.get("system_prompt", "")
            self.language: str         = config.get("language", "en")
            self._logger: logging.Logger = logging.getLogger("PromptBuilder")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def build(self, session: Session) -> list[dict]
        INPUTS:
            - session: Session — The current session with history, text-in queue, etc.
        OUTPUT:
            - list[dict] — OpenAI-format messages array ready for LLM:
                [
                    {"role": "system", "content": "..."},
                    {"role": "assistant", "content": "[summary of old turns]"},  # if compressed
                    {"role": "user", "content": "..."},   # history turn
                    {"role": "assistant", "content": "..."},   # history turn
                    ...
                    {"role": "user", "content": "[CONTEXT]...\n\nUser message"},  # current turn
                ]
        WHAT IT DOES:
            1. system_msg = self._build_system_message()
            2. messages = [{"role": "system", "content": system_msg}]
            3. If session.compressed_summary:
               messages.append({"role": "assistant", "content": f"[MEMORY SUMMARY]: {session.compressed_summary}"})
            4. history_turns = session.get_recent_history(MAX_HISTORY_TURNS)
            5. For each turn in history_turns[:-1]:  (all except the last user message)
               messages.append({"role": turn.role, "content": self._format_turn(turn)})
            6. text_in_entries = session.flush_text_in()
            7. current_user_msg = history_turns[-1].content if history_turns else ""
            8. If text_in_entries:
               context_block = self._build_context_block(text_in_entries)
               current_user_msg = f"{context_block}\n\n{current_user_msg}"
            9. messages.append({"role": "user", "content": current_user_msg})
            10. self._truncate_to_budget(messages)
            11. Return messages

    def _build_system_message(self) -> str
        INPUTS: None
        OUTPUT: str — The complete system message
        WHAT IT DOES:
            Constructs the system prompt by combining:
            1. The persona system_prompt from config
            2. Current datetime: f"\nCurrent date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            3. Response format instructions:
               "\n\nRESPONSE FORMAT RULES:\n"
               "- Always prefix your speech with an emotion tag: [cheerful] [calm] [concerned] [excited] [empathetic] [curious] [surprised]\n"
               "- Keep spoken responses under 3 sentences unless asked for more.\n"
               "- For agentic tasks, output: <agent>{\"action\": \"tool_name\", \"params\": {...}}</agent>\n"
               "- When performing a task, say a brief acknowledgment ALONGSIDE the agent tag.\n"
               "- Never describe tool calls verbally. Just say 'Let me check' or 'On it'.\n"
            4. Returns the combined string

    def _format_turn(self, turn: Turn) -> str
        INPUTS:
            - turn: Turn — A conversation turn dataclass
        OUTPUT:
            - str — Formatted turn content
        WHAT IT DOES:
            1. If turn.was_interrupted:
               Return f"{turn.content} [INTERRUPTED]"
            2. If turn.agent_output:
               Return f"{turn.content}\n<agent>{json.dumps(turn.agent_output)}</agent>"
            3. Otherwise: return turn.content

    def _build_context_block(self, entries: list[TextInEntry]) -> str
        INPUTS:
            - entries: list[TextInEntry] — Text-in injection entries (already sorted by priority)
        OUTPUT:
            - str — Formatted context block
        WHAT IT DOES:
            1. Builds a string:
               "[CONTEXT — The following information has been injected into the conversation]:\n"
            2. For each entry:
               f"- [{entry.source.upper()}] {entry.content}\n"
            3. Truncates total to MAX_TEXT_IN_CHARS
            4. Appends: "[END CONTEXT]\n"
            5. Returns the complete context block
        
        EXAMPLE OUTPUT:
            "[CONTEXT — The following information has been injected into the conversation]:
            - [TOOL_RESULT] London weather: 12°C, cloudy, light rain
            - [MEMORY] User prefers Celsius over Fahrenheit
            [END CONTEXT]"

    def _truncate_to_budget(self, messages: list[dict]) -> None
        INPUTS:
            - messages: list[dict] — The messages array (modified in place)
        OUTPUT: None (modifies messages in place)
        WHAT IT DOES:
            1. Estimates total tokens: sum(len(m["content"]) // 4 for m in messages)
               (rough estimate: 1 token ≈ 4 characters)
            2. If over MAX_PROMPT_TOKENS:
               a. Remove oldest history messages (index 1 onward, keep system at 0)
               b. Keep removing until under budget
               c. Log warning: "Truncated {n} history messages to fit token budget"

    def build_for_model(self, session: Session, model: str) -> list[dict]
        INPUTS:
            - session: Session
            - model: str — Target model (different models have different context windows)
        OUTPUT:
            - list[dict] — Messages array tailored for the specific model
        WHAT IT DOES:
            1. For fast brain (llama-3.1-8b): use MAX_PROMPT_TOKENS = 5000
            2. For smart brain (scout-17b): use MAX_PROMPT_TOKENS = 20000
            3. For agentic brain (kimi-k2): use MAX_PROMPT_TOKENS = 30000
            4. Calls build() with adjusted budget
            5. Returns messages

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - PromptBuilder     (class)
    - MAX_PROMPT_TOKENS (int constant)
    - MAX_HISTORY_TURNS (int constant)
═══════════════════════════════════════════════════════════════════════════════════

TOKEN BUDGET RATIONALE:
    llama-3.1-8b-instant has 131K context, but we cap at 5K because:
    1. Groq free tier limits are per-minute, so shorter prompts = more turns/minute
    2. Faster inference — less input = faster time-to-first-token
    3. The fast brain only needs recent context for conversational responses
    4. Complex tasks that need more context get routed to bigger models
"""
