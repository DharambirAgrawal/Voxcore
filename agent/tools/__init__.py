"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      VOXCORE — agent/tools/__init__.py                          ║
║                      TOOLS PACKAGE — INDIVIDUAL TOOL HANDLERS                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Package initializer for agent tools. Each tool is a class extending BaseTool.

EXPORTS:
    - BaseTool          (from agent.tools.base_tool)
    - WebSearchTool     (from agent.tools.web_search)
    - ArticleFetchTool  (from agent.tools.article_fetch)

USAGE:
    from agent.tools import BaseTool, WebSearchTool, ArticleFetchTool
"""

from agent.tools.base_tool import BaseTool
from agent.tools.web_search import WebSearchTool
from agent.tools.article_fetch import ArticleFetchTool
