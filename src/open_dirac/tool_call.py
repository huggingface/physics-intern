"""ToolCall dataclass — shared across agents, LLM layer, and tool executors."""

from dataclasses import dataclass


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    tool_input: dict
    output: str
    is_error: bool
    duration: float
