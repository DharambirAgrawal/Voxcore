"""
VoxCore - agent/tools/base_tool.py
Abstract base class for all VoxCore tools. Defines async execute(params),
parameter validation, and schema export.
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