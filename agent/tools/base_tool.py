"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     VOXCORE — agent/tools/base_tool.py                          ║
║          ABSTRACT BASE CLASS — ALL TOOLS EXTEND THIS INTERFACE                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    Defines the interface that ALL VoxCore tools must implement. Each tool
    is a class that extends BaseTool and implements the execute() method.

    Adding a new tool to VoxCore:
    1. Create a new file in agent/tools/ (e.g. email_tool.py)
    2. Create a class extending BaseTool
    3. Implement the execute(params) async method
    4. Register it in agent/tool_router.py
    5. Add the tool name to config.yaml agent.tools list

═══════════════════════════════════════════════════════════════════════════════════
IMPORTS REQUIRED:
═══════════════════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod     # Abstract base class
from typing import Any                  # Type hints
import logging                          # Module logger

═══════════════════════════════════════════════════════════════════════════════════
CLASSES:
═══════════════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────────────
CLASS: BaseTool(ABC)
──────────────────────────────────────────────────────────────────────────────────
    Abstract base class for all VoxCore tools.

    CLASS ATTRIBUTES:
        name: str = ""                  # Tool name (e.g. "web_search") — MUST override
        description: str = ""           # Human-readable description — MUST override
        required_params: list[str] = [] # Required parameter names — SHOULD override
        optional_params: list[str] = [] # Optional parameter names — MAY override

    CONSTRUCTOR: __init__(self)
    ─────────────────────────────────────────────────────────────
        INPUTS: None
        INITIALIZES:
            self._logger: logging.Logger = logging.getLogger(f"Tool.{self.name}")

    METHODS:
    ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def execute(self, params: dict) -> Any
        INPUTS:
            - params: dict — Parameters from the <agent> tag's "params" field
                Example: {"query": "London weather today"}
        OUTPUT:
            - Any — The tool's result (will be converted to str for injection)
                Can be: str, dict, list — anything that str() works on
        WHAT IT DOES:
            This is the main method each tool implements. It performs the actual
            work (API call, database query, computation, etc.) and returns the result.
        
        MUST BE:
            - Async (uses await for I/O operations)
            - Idempotent if possible (safe to retry)
            - Timeout-safe (should complete within tool_timeout_s from config)

    def validate_params(self, params: dict) -> tuple[bool, str]
        INPUTS:
            - params: dict — Parameters to validate
        OUTPUT:
            - tuple[bool, str] — (is_valid, error_message)
                - is_valid: True if all required params are present
                - error_message: "" if valid, or description of what's missing
        WHAT IT DOES:
            1. For each name in self.required_params:
               If name not in params:
                   Return (False, f"Missing required parameter: {name}")
            2. Return (True, "")
        
        Called by ToolRouter before execute() to catch missing params early.

    def get_schema(self) -> dict
        INPUTS: None
        OUTPUT: dict — JSON schema describing this tool's interface
        WHAT IT DOES:
            Returns {
                "name": self.name,
                "description": self.description,
                "required_params": self.required_params,
                "optional_params": self.optional_params,
            }
        
        Used for documentation and potential future tool-use prompting.

═══════════════════════════════════════════════════════════════════════════════════
EXPORTS:
    - BaseTool   (abstract class)
═══════════════════════════════════════════════════════════════════════════════════
"""


from abc import ABC, abstractmethod
import logging


class BaseTool(ABC):
    """Abstract base class for all VoxCore tools.

    Every tool must set:
        name             — Unique action name (routing key in <agent> JSON)
        description      — LLM reads this to decide when to call the tool
        required_params  — Router validates these before calling execute()
        optional_params  — Tool handles defaults internally
        produces_spoken_output — If True, result goes straight to TTS (skips main LLM)
    """

    name: str = ""
    description: str = ""
    required_params: list[str] = []
    optional_params: list[str] = []
    produces_spoken_output: bool = False

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = config or {}
        self._logger: logging.Logger = logging.getLogger(f"Tool.{self.name}")

    @abstractmethod
    async def execute(self, params: dict) -> str:
        """Run the tool. Always returns a plain string.

        If produces_spoken_output is True: return natural spoken text ready for TTS.
        If produces_spoken_output is False: return factual result for main LLM to synthesize.
        Never raise — catch all errors internally and return a spoken-friendly error string.
        """
        ...

    def validate_params(self, params: dict) -> tuple[bool, str]:
        """Check that all required parameters are present."""
        for name in self.required_params:
            if name not in params:
                return False, f"Missing required parameter: {name}"
        return True, ""

    def get_schema(self) -> dict:
        """Return a JSON-serialisable schema describing this tool's interface."""
        return {
            "name": self.name,
            "description": self.description,
            "required_params": self.required_params,
            "optional_params": self.optional_params,
            "produces_spoken_output": self.produces_spoken_output,
        }