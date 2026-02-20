"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          VOXCORE — brain/router.py                              ║
║          INTELLIGENCE ROUTER — DECIDES WHICH LLM MODEL HANDLES EACH TASK       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Decides which LLM model to use based on the detected task type and complexity.
    The fast brain (llama-3.1-8b) handles ALL real-time speech responses — it's
    the ONLY model in the hot path. Heavier models run asynchronously for complex
    tasks and never block the user-facing response.

    Routing strategy:
    - Conversational reply        → llama-3.1-8b-instant (always, hot path)
    - Tool execution              → kimi-k2-instruct (async, via agent/slow_llm.py)
    - Complex reasoning           → llama-4-scout-17b (async)
    - Long document analysis      → qwen3-32b (async)
    - Maximum intelligence tasks  → gpt-oss-120b (async, rare, rate-limited)

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

import logging                          # Module logger
import re                               # Regex for complexity detection

from core.session import Session

═══════════════════════════════════════════════════════════════════════════════════
CONSTANTS:
═══════════════════════════════════════════════════════════════════════════════════

# Complexity detection keywords (case-insensitive)
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

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: BrainRouter
──────────────────────────────────────────────────────────────────────────────────
    Routes user queries to the appropriate LLM model.

    CONSTRUCTOR: __init__(self, config: dict)
    ─────────────────────────────────────────────────────────────
        INPUTS:
            - config: dict — The "models" section from config.yaml:
                - llm_fast: str ("llama-3.1-8b-instant")
                - llm_smart: str ("meta-llama/llama-4-scout-17b-16e-instruct")
                - llm_agentic: str ("moonshotai/kimi-k2-instruct")
                - llm_deep: str ("qwen/qwen3-32b")
                - llm_power: str ("openai/gpt-oss-120b")
        
        INITIALIZES:
            self.fast_model: str     = config.get("llm_fast", "llama-3.1-8b-instant")
            self.smart_model: str    = config.get("llm_smart", "meta-llama/llama-4-scout-17b-16e-instruct")
            self.agentic_model: str  = config.get("llm_agentic", "moonshotai/kimi-k2-instruct")
            self.deep_model: str     = config.get("llm_deep", "qwen/qwen3-32b")
            self.power_model: str    = config.get("llm_power", "openai/gpt-oss-120b")
            self._logger: logging.Logger = logging.getLogger("BrainRouter")

    METHODS:
    ─────────────────────────────────────────────────────────────

    def route(self, user_text: str, session: Session) -> str
        INPUTS:
            - user_text: str — The user's current utterance (transcribed speech)
            - session: Session — Current session for context
        OUTPUT:
            - str — The model identifier to use for the primary response
        WHAT IT DOES:
            ALWAYS returns self.fast_model for the primary (spoken) response.
            
            The fast brain handles ALL real-time speech. The router's real job
            is to determine if a SECONDARY async call is needed for complex tasks.
            That decision is made by route_secondary().
            
            This method exists to keep the architecture clean and to allow
            future overrides (e.g., if fast model is rate-limited, fall back
            to smart model).
            
            Logic:
            1. If llm_client.is_rate_limited(self.fast_model):
               Log warning and return self.smart_model as fallback
            2. Otherwise: return self.fast_model

    def route_secondary(self, user_text: str, session: Session) -> str | None
        INPUTS:
            - user_text: str — The user's utterance
            - session: Session — Current session
        OUTPUT:
            - str | None — Model identifier for async secondary task, or None if not needed
        WHAT IT DOES:
            1. text_lower = user_text.lower()
            2. Check for max intelligence keywords:
               If any keyword in MAX_INTELLIGENCE_KEYWORDS is in text_lower:
                   Return self.power_model
            3. Check for document analysis:
               If any keyword in DOCUMENT_ANALYSIS_KEYWORDS is in text_lower:
                   Return self.deep_model
            4. Check for complex reasoning:
               If any keyword in COMPLEX_REASONING_KEYWORDS is in text_lower:
                   Return self.smart_model
            5. Default: return None (no secondary model needed)
        
        NOTE: This is called by TurnManager AFTER the fast brain has already
              started responding. The secondary model runs in parallel.

    def route_agent_task(self, action: str) -> str
        INPUTS:
            - action: str — The tool action name from <agent> tag
        OUTPUT:
            - str — Model identifier to use for this agentic task
        WHAT IT DOES:
            1. For tool-calling tasks (web_search, calendar, etc.):
               Return self.agentic_model (kimi-k2 — best at tool use)
            2. For reasoning tasks (analysis, comparison):
               Return self.smart_model (scout-17b)
            3. For summary/compression tasks:
               Return self.deep_model (qwen3-32b)
            4. Default: return self.agentic_model

    def get_model_info(self, model: str) -> dict
        INPUTS:
            - model: str — Model identifier
        OUTPUT:
            - dict — Model metadata:
                {
                    "name": str,
                    "role": str,       # "fast" / "smart" / "agentic" / "deep" / "power"
                    "is_async": bool,  # Whether this model runs asynchronously
                    "max_tokens": int,
                    "rate_limit_rpm": int,
                }
        WHAT IT DOES:
            Returns a lookup of model configurations for monitoring/debugging.

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - BrainRouter                 (class)
    - COMPLEX_REASONING_KEYWORDS  (list constant)
    - DOCUMENT_ANALYSIS_KEYWORDS  (list constant)
    - MAX_INTELLIGENCE_KEYWORDS   (list constant)
═══════════════════════════════════════════════════════════════════════════════════

KEY DESIGN PRINCIPLE:
    The user NEVER waits for a slow model. The fast brain ALWAYS responds
    immediately with a spoken acknowledgment. Heavy tasks run in the background
    and inject results via text_injector. This is non-negotiable for the
    real-time conversational UX.

    Flow:
    User: "Analyze the pros and cons of solar energy"
    Fast brain: "[calm] That's a great topic. Let me put together a thorough analysis."
    (Meanwhile, scout-17b runs in background)
    (Result injected via text_injector)
    Fast brain (next turn): "[cheerful] Here's what I found..."
"""


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

    def route_secondary(self, user_text: str, session: Session) -> str | None:
        """Determine if a secondary async model call is needed, and which one."""
        text_lower = user_text.lower()

        # Check max intelligence keywords first (most expensive → most specific)
        for keyword in MAX_INTELLIGENCE_KEYWORDS:
            if keyword in text_lower:
                self._logger.info("Routing secondary to power model for: '%s'", keyword)
                return self.power_model

        # Check document analysis
        for keyword in DOCUMENT_ANALYSIS_KEYWORDS:
            if keyword in text_lower:
                self._logger.info("Routing secondary to deep model for: '%s'", keyword)
                return self.deep_model

        # Check complex reasoning
        for keyword in COMPLEX_REASONING_KEYWORDS:
            if keyword in text_lower:
                self._logger.info("Routing secondary to smart model for: '%s'", keyword)
                return self.smart_model

        return None

    def route_agent_task(self, action: str) -> str:
        """Select the model for an agentic tool-calling task."""
        action_lower = action.lower()

        if action_lower in _REASONING_ACTIONS:
            return self.smart_model

        if action_lower in _SUMMARY_ACTIONS:
            return self.deep_model

        # Default: agentic model (kimi-k2, best at tool use)
        return self.agentic_model

    def get_model_info(self, model: str) -> dict:
        """Return metadata about a model for monitoring/debugging."""
        role = self._model_to_role.get(model, "unknown")
        registry = _MODEL_REGISTRY.get(role, {
            "role": "unknown",
            "is_async": True,
            "max_tokens": 2048,
            "rate_limit_rpm": 10,
        })

        return {
            "name": model,
            **registry,
        }