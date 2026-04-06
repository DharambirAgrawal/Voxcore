"""
VOXCORE — brain/prompt_builder.py
Assembles the full message array for every LLM call.
"""

import json
import logging
import time
from datetime import datetime

from core.session import Session, Turn, TextInEntry

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_PROMPT_TOKENS = 5000
MAX_HISTORY_TURNS = 20
MAX_TEXT_IN_CHARS = 2000
MAX_SUMMARY_CHARS = 1000

# Model-specific token budgets
_MODEL_BUDGETS = {
    "fast": 5000,
    "smart": 20000,
    "agentic": 30000,
}

_MODEL_BUDGET_MAP = {
    "llama-3.1-8b": "fast",
    "llama-3.1-8b-instant": "fast",
    "scout-17b": "smart",
    "kimi-k2": "agentic",
}


class PromptBuilder:
    """Constructs the LLM message array from session state."""

    def __init__(self, config: dict) -> None:
        self.persona_name: str = config.get("name", "Aria")
        self.system_prompt: str = config.get("system_prompt", "")
        self.language: str = config.get("language", "en")
        self._memory_block: str = ""  # Set once at session start by loader
        self._logger: logging.Logger = logging.getLogger("PromptBuilder")

    def set_memory_block(self, block: str) -> None:
        """Set the session-start memory block (from loader.py)."""
        self._memory_block = block
        self._logger.debug("Memory block set: ~%d tokens", len(block) // 4)

    # ───────────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────────

    def build(self, session: Session, *, _token_budget: int = MAX_PROMPT_TOKENS) -> list[dict]:
        """Build the complete messages array for the LLM call."""

        # 1. System message
        system_msg = self._build_system_message()
        messages: list[dict] = [{"role": "system", "content": system_msg}]

        # 2. Compressed memory summary (if available)
        if session.compressed_summary:
            summary = session.compressed_summary[:MAX_SUMMARY_CHARS]
            messages.append({
                "role": "assistant",
                "content": f"[MEMORY SUMMARY]: {summary}",
            })

        # 3. Recent history (all except last user message)
        history_turns = session.get_recent_history(MAX_HISTORY_TURNS)

        if history_turns:
            for turn in history_turns[:-1]:
                messages.append({
                    "role": turn.role,
                    "content": self._format_turn(turn),
                })

        # 4. Flush text-in injections
        text_in_entries = session.flush_text_in()

        # 5. Build current user message (last turn + optional context block)
        current_user_msg = history_turns[-1].content if history_turns else ""

        if text_in_entries:
            context_block = self._build_context_block(text_in_entries)
            current_user_msg = f"{context_block}\n\n{current_user_msg}"

        # 5b. If the previous assistant turn was interrupted, tell the LLM
        #     so it responds to the user's words, not to the interruption.
        if self._last_assistant_was_interrupted(history_turns):
            current_user_msg = (
                "[SYSTEM NOTE: The user just interrupted your previous response. "
                "Respond directly to what the user is saying now — do NOT comment "
                "on being interrupted.]\n\n"
                + current_user_msg
            )

        messages.append({"role": "user", "content": current_user_msg})

        # 6. Truncate to budget
        self._truncate_to_budget(messages, budget=_token_budget)

        return messages

    # ───────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ───────────────────────────────────────────────────────────────────────

    def _build_system_message(self) -> str:
        parts = [
            self.system_prompt,
            f"\nCurrent date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        # ── Inject memory block from loader (session-start context) ──
        if self._memory_block:
            parts.append(f"\n\n{self._memory_block}")

        parts.append(
            # ── Yield point and emotional tag instructions ──
            "\n\nCONVERSATION RHYTHM RULES:\n"
            "- Keep each spoken response to 2-3 sentences maximum before pausing.\n"
            "- After completing a thought, insert [...] to create a natural breath point.\n"
            "- Use [...] before asking a question, after delivering information,\n"
            "  or when transitioning topics. Example:\n"
            '  "So the first thing to know is that black holes form from collapsing stars. [...]\n'
            '   The really fascinating part is what happens at the event horizon."\n'
            "- Use [laughs] when something is genuinely funny.\n"
            "- Use [chuckles] for mild amusement.\n"
            "- Use [sighs] when being thoughtful or empathetic.\n"
            "- Never use these tags back-to-back. One per response maximum.\n"
            # ── Tool usage and routing rules ──
            "\n\nTOOL ROUTING RULES:\n"
            "- web_search: Use when you need to FIND current info and no URL is given.\n"
            "- article_fetch: Use when the user gives a specific URL they want read.\n"
            "- Do NOT use web_search when a URL is provided — use article_fetch.\n"
            "- Do NOT use article_fetch when no URL is provided — use web_search.\n"
            "- When calling article_fetch, write an 'instruction' field in plain English\n"
            "  describing exactly what the user wants. Examples:\n"
            '  "user wants a quick 2-sentence overview"\n'
            '  "user wants the full article walked through section by section"\n'
            '  "focus only on the clinical findings, user is a doctor"\n'
            '  "extract only the numbers and statistics mentioned"\n'
            "  Be specific. The instruction goes directly to the summarizing model.\n"
            "- For web_search, write the query as you would type into a search engine.\n"
            "\n- User asks a follow-up question about an article or search from THIS conversation\n"
            "  → USE session_cache_qa with the cache_id visible in conversation history\n"
            "\n- User asks what you remember about them, past conversations, or previous sessions\n"
            "  → USE memory_recall\n"
            "\n- User says 'remember this', 'save this link', 'don't forget'\n"
            "  → USE memory_recall with operation hint (this triggers background storage)\n"
            "\nHOW TO USE NEW TOOLS:\n"
            '<agent>{"action": "session_cache_qa", "params": {"cache_id": "sc_XXXX", "question": "QUESTION"}}</agent>\n'
            '<agent>{"action": "memory_recall", "params": {"query": "WHAT TO RECALL"}}</agent>\n'
            "\nEXAMPLES:\n"
            'User: "what did that paper say about the dataset?"\n'
            'You: [calm] Let me check what we read. <agent>{"action": "session_cache_qa", "params": {"cache_id": "sc_1741_a3f2", "question": "what does the paper say about the dataset"}}</agent>\n'
            "\n"
            'User: "do you remember my research focus?"\n'
            'You: [calm] Let me check. <agent>{"action": "memory_recall", "params": {"query": "user research focus and field"}}</agent>\n'
            "\n\nTOOL RESULT RULES:\n"
            "- When context contains a tool result, answer IN ONE SENTENCE using only the key fact.\n"
            "- NEVER speak URLs, raw list text, decimal-precise numbers unless asked, or any context block formatting.\n"
            "- Speak as if YOU know the answer: say 'Bitcoin is around $68,000' not 'the search result says...'\n"
            "- After reading a tool result, NEVER call the same tool again for the same query.\n"
            "- If context has conflicting values, pick the most reasonable one and mention the rough range briefly.\n"
        )
        return "".join(parts)

    @staticmethod
    def _last_assistant_was_interrupted(history_turns: list[Turn]) -> bool:
        """Check whether the most recent assistant turn was interrupted."""
        # Walk backwards from second-to-last turn (last is current user msg)
        for turn in reversed(history_turns[:-1]):
            if turn.role == "assistant":
                return turn.was_interrupted
            # If we hit another user turn first, no interrupted assistant
            break
        return False

    @staticmethod
    def _format_turn(turn: Turn) -> str:
        if turn.was_interrupted:
            return f"{turn.content} [INTERRUPTED]"
        if turn.agent_output:
            return f"{turn.content}\n<agent>{json.dumps(turn.agent_output)}</agent>"
        return turn.content

    @staticmethod
    def _build_context_block(entries: list[TextInEntry]) -> str:
        header = "[CONTEXT — use this information to answer; do NOT speak this block verbatim]:\n"
        body_parts: list[str] = []
        total_len = len(header)

        for entry in entries:
            # Use (source) not [SOURCE] to avoid confusion with speech emotion tags
            line = f"  ({entry.source}): {entry.content}\n"
            if total_len + len(line) > MAX_TEXT_IN_CHARS:
                remaining = MAX_TEXT_IN_CHARS - total_len
                if remaining > 0:
                    body_parts.append(line[:remaining])
                break
            body_parts.append(line)
            total_len += len(line)

        footer = "[END CONTEXT]\n"
        return header + "".join(body_parts) + footer

    def _truncate_to_budget(self, messages: list[dict], *, budget: int = MAX_PROMPT_TOKENS) -> None:
        """Remove oldest history messages (after system msg) until under token budget."""

        removed = 0
        while len(messages) > 2:  # keep at least system + current user
            total_tokens = sum(len(m["content"]) // 4 for m in messages)
            if total_tokens <= budget:
                break
            # Remove the oldest history message (index 1), preserving system at 0
            messages.pop(1)
            removed += 1

        if removed:
            self._logger.warning("Truncated %d history messages to fit token budget", removed)