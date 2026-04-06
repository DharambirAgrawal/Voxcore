"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — agent/tools/web_search.py                         ║
║                    WEB SEARCH TOOL — SEARCHES THE INTERNET                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import os

import aiohttp

from agent.tools.base_tool import BaseTool


class WebSearchTool(BaseTool):
    """Web search tool using Tavily search API."""

    name = "web_search"
    description = (
        "Search the internet for current information, news, facts, prices, "
        "or events that may have changed recently. Use when you need to FIND "
        "information and no URL has been provided. "
        "Do NOT use when the user has given a specific URL — use article_fetch for that."
    )
    required_params = ["query"]
    optional_params = ["num_results", "search_depth"]
    produces_spoken_output = False  # Result goes to main LLM for synthesis

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        cfg = self.config.get("web_search", {}) if self.config else {}
        self.default_num_results: int = cfg.get("default_num_results", 3)
        self.default_search_depth: str = cfg.get("search_depth", "basic")
        self.api_key: str = os.environ.get("TAVILY_API_KEY", "")
        self._logger: logging.Logger = logging.getLogger("Tool.web_search")
        if not self.api_key:
            self._logger.warning(
                "TAVILY_API_KEY not set — web_search tool will return errors until configured"
            )

    async def execute(self, params: dict) -> str:
        valid, err = self.validate_params(params)
        if not valid:
            return f"Error: {err}"

        if not self.api_key:
            return "Error: TAVILY_API_KEY is not configured — cannot perform web search"

        query = params["query"]

        try:
            num_results = int(params.get("num_results", self.default_num_results))
        except (TypeError, ValueError):
            num_results = self.default_num_results

        search_depth = params.get("search_depth", self.default_search_depth)
        if search_depth not in ("basic", "advanced"):
            search_depth = "basic"

        try:
            results = await self._search_tavily(query, num_results, search_depth)
        except Exception as exc:
            self._logger.error("Search failed: %s", exc, exc_info=True)
            return f"Search error: {exc}"

        return self._format_results(query, results)

    async def _search_tavily(
        self, query: str, num_results: int, search_depth: str = "basic"
    ) -> list[dict]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"Tavily API error {resp.status}: {body}")
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

    def _format_results(self, query: str, results: list[dict]) -> str:
        if not results:
            return f"No results found for '{query}'"

        output = f"Search results for '{query}':\n"
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']} — {result['snippet']}\n"
            output += f"   URL: {result['url']}\n"
        return output