"""Agent registry — re-export all agent classes (lazy to avoid circular imports)."""


def __getattr__(name: str):
    """Lazy import agent classes on first access."""
    _registry = {
        "OrchestratorAgent": ".orchestrator",
        "ComputerAgent": ".computer",
        "ResearcherAgent": ".researcher",
        "ReviewerAgent": ".reviewer",
        "CriticAgent": ".critic",
        "FormatterAgent": ".formatter",
        "SurveyorAgent": ".surveyor",
        "PlannerAgent": ".planner",
        "AdjudicatorAgent": ".adjudicator",
    }
    if name in _registry:
        import importlib

        mod = importlib.import_module(_registry[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
