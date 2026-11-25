import os
import functools
from typing import Any, Callable, Dict, Optional
from app.core.config import settings
from app.core.logging import logger

try:
    from langsmith import traceable as _ls_traceable, Client as LangSmithClient
    _HAS_LANGSMITH = True
except ImportError:
    _ls_traceable = None
    LangSmithClient = None
    _HAS_LANGSMITH = False


def configure_observability() -> bool:
    """
    Configure LangSmith / LangChain tracing environment variables at startup.
    Returns True if tracing is enabled and configured with an API key.
    """
    if not settings.LANGSMITH_TRACING:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        logger.info("LangSmith tracing disabled via config.")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    if settings.LANGSMITH_API_KEY and "placeholder" not in settings.LANGSMITH_API_KEY.lower():
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        logger.info(f"LangSmith tracing enabled for project '{settings.LANGSMITH_PROJECT}'.")
        return True

    logger.info(
        f"LangSmith tracing initialized (project='{settings.LANGSMITH_PROJECT}'), "
        "no live LANGSMITH_API_KEY configured — spans will run locally."
    )
    return False


def get_langsmith_client() -> Optional[Any]:
    """Return a LangSmith Client when available and API key is configured."""
    if not _HAS_LANGSMITH or LangSmithClient is None:
        return None
    if not settings.LANGSMITH_API_KEY or "placeholder" in settings.LANGSMITH_API_KEY.lower():
        return None
    try:
        return LangSmithClient(
            api_key=settings.LANGSMITH_API_KEY,
            api_url=settings.LANGSMITH_ENDPOINT,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmithClient: {e}")
        return None


def traceable(
    name: Optional[str] = None,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Safe wrapper around langsmith.traceable.
    Falls back to a transparent decorator if langsmith is not installed or
    when running without live LangSmith credentials.
    """
    def decorator(func: Callable) -> Callable:
        if _HAS_LANGSMITH and _ls_traceable is not None:
            wrapped = _ls_traceable(
                name=name or func.__name__,
                run_type=run_type,
                metadata=metadata or {},
                tags=tags or [],
            )(func)
            return wrapped

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
