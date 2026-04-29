"""Orchestrator agent subpackage."""


def __getattr__(name: str):
    if name == "OrchestratorAgent":
        from .agent import OrchestratorAgent

        return OrchestratorAgent
    if name == "OrchestratorToolExecutor":
        from .tools import OrchestratorToolExecutor

        return OrchestratorToolExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
