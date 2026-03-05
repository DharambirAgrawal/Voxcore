"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                VOXCORE — agent/tools/session_cache_qa.py                        ║
║    QA OVER CACHED CONTENT (ARTICLES, SEARCH RESULTS) FROM THIS SESSION         ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Answer a question from an article, paper, or search result that was read
    or searched EARLIER IN THIS CONVERSATION. Uses session_cache to retrieve
    the full content and llm_fast to generate a spoken answer.

    produces_spoken_output = True — result goes DIRECTLY to TTS via
    SPOKEN_TOOL_OUTPUT event. The main LLM never sees it.
"""

import logging
import os

from groq import AsyncGroq

from agent.tools.base_tool import BaseTool
from memory.session_cache import SessionCache


class SessionCacheQATool(BaseTool):
    """Answer follow-up questions about cached content from this session."""

    name = "session_cache_qa"
    description = (
        "Answer a question from an article, paper, or search result that was "
        "read or searched EARLIER IN THIS CONVERSATION. Use when the user asks "
        "a follow-up question about content from the current session. "
        "Requires the cache_id from the earlier result (visible in conversation). "
        "Do NOT use for new searches. Do NOT use for past sessions."
    )
    required_params = ["cache_id", "question"]
    optional_params = []
    produces_spoken_output = True  # result → TTS directly

    def __init__(
        self,
        config: dict | None = None,
        session_cache: SessionCache | None = None,
    ) -> None:
        super().__init__(config)
        self._session_cache = session_cache
        # Model from config — never hardcoded
        self._model = (config or {}).get("models", {}).get(
            "llm_fast", "llama-3.1-8b-instant"
        )
        self._groq = AsyncGroq()

    async def execute(self, params: dict) -> str:
        """Look up cached content and answer the question via llm_fast."""
        valid, err = self.validate_params(params)
        if not valid:
            return f"Sorry, I couldn't process that. {err}"

        if self._session_cache is None:
            return (
                "I don't seem to have that content cached anymore. "
                "Would you like me to fetch it again?"
            )

        content = self._session_cache.get_content(params["cache_id"])
        if not content:
            return (
                "I don't seem to have that content cached anymore. "
                "Would you like me to fetch it again?"
            )

        try:
            response = await self._groq.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Answer this question based only on the content below.\n"
                            f"Spoken answer, 2-3 sentences max, no markdown, no bullet points.\n"
                            f"Question: {params['question']}\n\n"
                            f"Content:\n{content[:6000]}"
                        ),
                    }
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self._logger.error("session_cache_qa LLM call failed: %s", e, exc_info=True)
            return "Sorry, I ran into an issue looking that up. Could you ask again?"
