"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — agent/tools/web_search.py                         ║
║                    WEB SEARCH TOOL — SEARCHES THE INTERNET                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Performs web searches using a search API (e.g. Tavily, SerpAPI, or DuckDuckGo).
    Returns relevant search results as formatted text that gets injected back
    into the conversation context.

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging                          # Module logger
import os                               # API keys from environment
from typing import Any

import aiohttp                          # Async HTTP requests

from agent.tools.base_tool import BaseTool

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: WebSearchTool(BaseTool)
──────────────────────────────────────────────────────────────────────────────────
    Web search tool using a search API.

    CLASS ATTRIBUTES:
        name = "web_search"
        description = "Search the web for current information"
        required_params = ["query"]
        optional_params = ["num_results", "search_depth"]

    CONSTRUCTOR: __init__(self)
    ─────────────────────────────────────────────────────────────
        INPUTS: None
        INITIALIZES:
            super().__init__()
            self.api_key: str = os.environ.get("TAVILY_API_KEY", "")
            # Or use DuckDuckGo (no API key needed) as fallback
            self._logger: logging.Logger = logging.getLogger("Tool.web_search")

    METHODS:
    ─────────────────────────────────────────────────────────────

    async def execute(self, params: dict) -> str
        INPUTS:
            - params: dict containing:
                - query: str — The search query (REQUIRED)
                - num_results: int — Number of results to return (default: 5)
                - search_depth: str — "basic" or "advanced" (default: "basic")
        OUTPUT:
            - str — Formatted search results text
                Format:
                    "Search results for '{query}':\n"
                    "1. {title} — {snippet}\n   URL: {url}\n"
                    "2. {title} — {snippet}\n   URL: {url}\n"
                    ...
        WHAT IT DOES:
            1. Validate params: valid, err = self.validate_params(params)
               If not valid: return f"Error: {err}"
            2. query = params["query"]
            3. num_results = params.get("num_results", 5)
            4. If self.api_key:
               results = await self._search_tavily(query, num_results)
            5. Else:
               results = await self._search_duckduckgo(query, num_results)
            6. Return self._format_results(query, results)

    async def _search_tavily(self, query: str, num_results: int) -> list[dict]
        INPUTS:
            - query: str — Search query
            - num_results: int — Max results
        OUTPUT:
            - list[dict] — List of {"title": str, "snippet": str, "url": str}
        WHAT IT DOES:
            1. POST to https://api.tavily.com/search:
               json = {
                   "api_key": self.api_key,
                   "query": query,
                   "max_results": num_results,
                   "search_depth": "basic"
               }
            2. Parse response JSON
            3. Extract results into list of dicts
            4. Return results

    async def _search_duckduckgo(self, query: str, num_results: int) -> list[dict]
        INPUTS:
            - query: str — Search query
            - num_results: int — Max results
        OUTPUT:
            - list[dict] — List of {"title": str, "snippet": str, "url": str}
        WHAT IT DOES:
            1. Uses DuckDuckGo Instant Answer API (no API key needed):
               GET https://api.duckduckgo.com/?q={query}&format=json
            2. Parse response
            3. Extract results into list of dicts
            4. Return results
        
        NOTE: DuckDuckGo API is limited. For production, use Tavily or SerpAPI.

    def _format_results(self, query: str, results: list[dict]) -> str
        INPUTS:
            - query: str — Original query
            - results: list[dict] — Search results
        OUTPUT:
            - str — Human-readable formatted results
        WHAT IT DOES:
            1. If no results: return f"No results found for '{query}'"
            2. Build formatted string:
               output = f"Search results for '{query}':\n"
               For i, result in enumerate(results, 1):
                   output += f"{i}. {result['title']} — {result['snippet']}\n"
                   output += f"   URL: {result['url']}\n"
            3. Return output

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - WebSearchTool   (class)
═══════════════════════════════════════════════════════════════════════════════════
"""

import logging
import os
from typing import Any

import aiohttp

from agent.tools.base_tool import BaseTool


class WebSearchTool(BaseTool):
    """Web search tool using a search API."""

    name = "web_search"
    description = "Search the web for current information"
    required_params = ["query"]
    optional_params = ["num_results", "search_depth"]

    def __init__(self) -> None:
        super().__init__()
        self.api_key: str = os.environ.get("TAVILY_API_KEY", "")
        self._logger: logging.Logger = logging.getLogger("Tool.web_search")

    async def execute(self, params: dict) -> str:
        valid, err = self.validate_params(params)
        if not valid:
            return f"Error: {err}"

        query = params["query"]
        num_results = params.get("num_results", 5)
        try:
            num_results = int(num_results)
        except (TypeError, ValueError):
            num_results = 5

        try:
            if self.api_key:
                results = await self._search_tavily(query, num_results)
            else:
                results = await self._search_duckduckgo(query, num_results)
        except Exception as exc:
            self._logger.error("Search failed: %s", exc, exc_info=True)
            return f"Search error: {exc}"

        return self._format_results(query, results)

    async def _search_tavily(self, query: str, num_results: int) -> list[dict]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": "basic",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results: list[dict] = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url": item.get("url", ""),
                }
            )
        return results

    async def _search_duckduckgo(self, query: str, num_results: int) -> list[dict]:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

        results: list[dict] = []

        # Abstract text (top result)
        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading", "Result"),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                }
            )

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if len(results) >= num_results:
                break
            if "Text" in topic:
                results.append(
                    {
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                    }
                )

        return results[:num_results]

    def _format_results(self, query: str, results: list[dict]) -> str:
        if not results:
            return f"No results found for '{query}'"

        output = f"Search results for '{query}':\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']} — {result['snippet']}\n"
            output += f"   URL: {result['url']}\n"
        return output