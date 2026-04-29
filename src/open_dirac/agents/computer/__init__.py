"""Computer agent subpackage."""


def __getattr__(name: str):
    if name == "ComputerAgent":
        from .agent import ComputerAgent

        return ComputerAgent
    if name == "ToolExecutor":
        from .tools import ToolExecutor

        return ToolExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
