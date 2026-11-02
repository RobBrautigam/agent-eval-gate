"""Score an AI agent on accuracy, cost and latency from recorded runs, and fail the build when a
change makes it worse."""

from .core import Case, EvalError, Reply, load_cases, record, score

__version__ = "0.1.0"
__all__ = ["Case", "EvalError", "Reply", "load_cases", "record", "score", "__version__"]
