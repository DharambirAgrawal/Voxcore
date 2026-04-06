"""
VOXCORE — brain/router.py
Intelligence router — decides which LLM model handles each task.
"""

import logging
import re

from core.session import Session

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

COMPLEX_REASONING_KEYWORDS = [
    "explain why", "compare", "analyze", "pros and cons", "what if",
    "step by step", "reasoning", "logic", "evaluate", "critique",
    "think through", "break down", "deep dive",
]

DOCUMENT_ANALYSIS_KEYWORDS = [
    "summarize this", "analyze this document", "read through",
    "what does this text say", "extract from", "key points from",
]

MAX_INTELLIGENCE_KEYWORDS = [
    "write a complete", "design a system", "architect",
    "comprehensive plan", "full analysis", "research paper",
]

# Agent action routing categories
_TOOL_ACTIONS = {
    "web_search", "calendar", "reminder", "timer", "weather",
    "email", "smart_home", "api_call", "file_read", "file_write",
}

_REASONING_ACTIONS = {
    "analysis", "comparison", "evaluation", "critique", "reasoning",
}

_SUMMARY_ACTIONS = {
    "summarize", "compress", "extract", "digest",
}

# Model metadata registry
_MODEL_REGISTRY = {
    "fast": {
        "role": "fast",
        "is_async": False,
        "max_tokens": 1024,
        "rate_limit_rpm": 30,
    },
    "smart": {
        "role": "smart",
        "is_async": True,
        "max_tokens": 4096,
        "rate_limit_rpm": 15,
    },
    "agentic": {
        "role": "agentic",
        "is_async": True,
        "max_tokens": 4096,
        "rate_limit_rpm": 15,
    },
    "deep": {
        "role": "deep",
        "is_async": True,
        "max_tokens": 8192,
        "rate_limit_rpm": 10,
    },
    "power": {
        "role": "power",
        "is_async": True,
        "max_tokens": 16384,
        "rate_limit_rpm": 5,
    },
}


class BrainRouter:
    """Routes user queries to the appropriate LLM model."""

    def __init__(self, config: dict) -> None:
        self.fast_model: str = config.get("llm_fast", "llama-3.1-8b-instant")
        self.smart_model: str = config.get("llm_smart", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.agentic_model: str = config.get("llm_agentic", "moonshotai/kimi-k2-instruct")
        self.deep_model: str = config.get("llm_deep", "qwen/qwen3-32b")
        self.power_model: str = config.get("llm_power", "openai/gpt-oss-120b")
        self._logger: logging.Logger = logging.getLogger("BrainRouter")

        # Build reverse lookup: model name → role
        self._model_to_role: dict[str, str] = {
            self.fast_model: "fast",
            self.smart_model: "smart",
            self.agentic_model: "agentic",
            self.deep_model: "deep",
            self.power_model: "power",
        }

    def route(self, user_text: str, session: Session) -> str:
        """Select the primary model for the spoken response (always fast brain)."""
        # Future: check rate limiting and fall back
        # if llm_client.is_rate_limited(self.fast_model):
        #     self._logger.warning("Fast model rate-limited, falling back to smart model")
        #     return self.smart_model
        return self.fast_model

    def route_agent_task(self, action: str) -> str:
        """Select the model for an agentic tool-calling task."""
        action_lower = action.lower()

        if action_lower in _REASONING_ACTIONS:
            return self.smart_model

        if action_lower in _SUMMARY_ACTIONS:
            return self.deep_model

        # Default: agentic model (kimi-k2, best at tool use)
        return self.agentic_model