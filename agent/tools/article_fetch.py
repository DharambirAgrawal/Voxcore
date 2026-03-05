"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    VOXCORE — agent/tools/article_fetch.py                        ║
║           FETCH URL → CLEAN TEXT → SPOKEN SUMMARY VIA GROQ                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Fetches content from a URL, cleans it to plain text, and generates a
    spoken-aloud summary using an internal Groq LLM call.

    produces_spoken_output = True — the result goes DIRECTLY to TTS.
    The main LLM never sees it, so the summary must be fully self-contained
    and ready to be spoken aloud.

    The main LLM passes an `instruction` field that describes what the user
    wants from the article. This instruction is forwarded verbatim to the
    summarising model as the task line — no mode enums, no information loss.

═══════════════════════════════════════════════════════════════════════════════════
"""

import logging
import os
import re
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

from agent.tools.base_tool import BaseTool

# Must match a real http/https URL — catches hallucinated strings like "the link they sent"
_REAL_URL_RE = re.compile(r'^https?://', re.IGNORECASE)


class ArticleFetchTool(BaseTool):
    """Fetch a URL, extract text, and produce a spoken summary via Groq."""

    name = "article_fetch"
    description = (
        "Fetch and read the contents of a specific URL or article link. "
        "Use when the user has given a specific URL and wants to hear what it says. "
        "Do NOT use when no URL has been provided — use web_search to FIND information first."
    )
    required_params = ["url"]
    optional_params = ["instruction"]
    produces_spoken_output = True  # Result goes directly to TTS

    # ── Config defaults (overridden by config.yaml agent.article_fetch) ──
    DEFAULT_FETCH_TIMEOUT = 10
    DEFAULT_MAX_WORDS = 8000
    DEFAULT_LONG_THRESHOLD = 3000
    # llama-4-scout has 30K TPM vs llama-3.1-8b's 6K TPM — much less rate-limiting
    DEFAULT_SUMMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    DEFAULT_LONG_MODEL = "moonshotai/kimi-k2-instruct"
    DEFAULT_SUMMARY_MAX_WORDS = 150
    # Fallback chain tried in order when primary model returns 429
    DEFAULT_FALLBACK_MODELS = [
        "moonshotai/kimi-k2-instruct",        # 10K TPM
        "llama-3.3-70b-versatile",            # 12K TPM
        "meta-llama/llama-4-maverick-17b-128e-instruct",  # 6K TPM but different quota
    ]

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config.get("article_fetch", {}) if self.config else {}
        self.fetch_timeout: int = cfg.get("fetch_timeout_seconds", self.DEFAULT_FETCH_TIMEOUT)
        self.max_article_words: int = cfg.get("max_article_words", self.DEFAULT_MAX_WORDS)
        self.long_article_threshold: int = cfg.get("long_article_threshold", self.DEFAULT_LONG_THRESHOLD)
        self.summary_model: str = cfg.get("summary_model", self.DEFAULT_SUMMARY_MODEL)
        self.long_article_model: str = cfg.get("long_article_model", self.DEFAULT_LONG_MODEL)
        self.fallback_models: list[str] = cfg.get("fallback_models", self.DEFAULT_FALLBACK_MODELS)
        self.default_summary_max_words: int = cfg.get("default_summary_max_words", self.DEFAULT_SUMMARY_MAX_WORDS)
        self._logger = logging.getLogger("Tool.article_fetch")

    async def execute(self, params: dict) -> str:
        """Fetch URL → clean text → summarise via Groq → return spoken text."""
        try:
            valid, err = self.validate_params(params)
            if not valid:
                return f"Sorry, I couldn't fetch that article. {err}"

            url = params["url"]
            instruction = params.get(
                "instruction",
                f"Summarize the article naturally in about {self.default_summary_max_words} spoken words.",
            )

            # ── URL validation — catch hallucinated non-URL values ──
            if not _REAL_URL_RE.match(url):
                self._logger.warning("Rejected non-URL value: %r", url)
                return (
                    "It looks like I didn't get an actual link. "
                    "Go ahead and paste the URL and I'll read it for you."
                )

            # ── Fetch ──
            raw_html = await self._fetch_url(url)
            if raw_html is None:
                return "Sorry, I wasn't able to reach that page. It might be down or blocking requests."

            # ── Clean ──
            cleaned_text = self._extract_text(raw_html)
            if not cleaned_text or len(cleaned_text.split()) < 30:
                return "I fetched the page but couldn't find enough readable content on it."

            # ── Truncate if massive ──
            words = cleaned_text.split()
            if len(words) > self.max_article_words:
                cleaned_text = " ".join(words[: self.max_article_words])

            # ── Pick primary model based on article length ──
            word_count = len(words)
            primary_model = (
                self.long_article_model
                if word_count > self.long_article_threshold
                else self.summary_model
            )

            # Build fallback chain: primary first, then fallback_models (deduped)
            seen: set[str] = {primary_model}
            model_chain: list[str] = [primary_model]
            for m in self.fallback_models:
                if m not in seen:
                    seen.add(m)
                    model_chain.append(m)

            # ── Summarise with automatic fallback on 429 ──
            summary = await self._summarise_with_fallback(
                cleaned_text, instruction, model_chain
            )
            return summary

        except Exception as e:
            self._logger.error("article_fetch failed: %s", e, exc_info=True)
            return "Sorry, something went wrong while reading that article."

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch raw HTML from the given URL."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; VoxCore/1.0; +https://voxcore.ai)"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            timeout = aiohttp.ClientTimeout(total=self.fetch_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, allow_redirects=True) as resp:
                    if resp.status != 200:
                        self._logger.warning("Fetch returned %d for %s", resp.status, url)
                        return None
                    return await resp.text()
        except Exception as e:
            self._logger.error("Fetch error for %s: %s", url, e)
            return None

    def _extract_text(self, html: str) -> str:
        """Strip HTML to clean, readable plain text."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    async def _summarise_with_fallback(
        self, article_text: str, instruction: str, model_chain: list[str]
    ) -> str:
        """Try each model in model_chain; skip to next on 429, stop on first success."""
        for model in model_chain:
            result, rate_limited = await self._summarise(article_text, instruction, model)
            if not rate_limited:
                return result
            self._logger.warning(
                "Model '%s' rate-limited (429) — trying next in chain", model
            )
        return "I fetched the article but all summarisation models are rate-limited right now. Please try again in a moment."

    async def _summarise(self, article_text: str, instruction: str, model: str) -> tuple[str, bool]:
        """Call Groq to produce a spoken-aloud summary. Returns (text, rate_limited)."""
        prompt = (
            "You are Aria, a friendly voice AI assistant reading this article aloud to a friend on a call.\n\n"
            "Rules:\n"
            "- Write for speech only. No markdown, no bullet points, no headers, no asterisks.\n"
            "- Natural flowing sentences, conversational tone.\n"
            "- You are DESCRIBING the content from the outside — never speak as if you ARE the subject of the article.\n"
            "  Wrong: 'My responsibilities include...' or 'As a technician, I would...'\n"
            "  Right: 'The role involves...' or 'According to the posting, candidates would...'\n"
            "- Do NOT open with 'I'm here to tell you', 'This article says', or 'The author writes'.\n"
            "- Start directly with the most interesting or relevant content.\n"
            "- Do not reproduce any URLs.\n\n"
            f"Task: {instruction}\n\n"
            f"Article:\n{article_text}"
        )

        try:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                return (
                    "I fetched the article but couldn't summarise it because the Groq API key is not configured.",
                    False,
                )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 1024,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 429:
                        body = await resp.text()
                        self._logger.warning(
                            "Rate limit 429 on model '%s': %s", model, body[:120]
                        )
                        return "", True  # signal caller to try next model
                    if resp.status != 200:
                        body = await resp.text()
                        self._logger.error("Groq API error %d on model '%s': %s", resp.status, model, body)
                        return "I fetched the article but the summarisation service returned an error.", False
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    self._logger.info("Summarised with model '%s' (%d chars)", model, len(text))
                    return text, False

        except Exception as e:
            self._logger.error("Summarisation failed on model '%s': %s", model, e, exc_info=True)
            return "I fetched the article but ran into an error while summarising it.", False
