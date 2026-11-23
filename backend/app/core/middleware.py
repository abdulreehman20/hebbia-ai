import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("hebbia")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def setup_middleware(app: ASGIApp) -> None:
    """Register all HTTP middleware on the FastAPI app."""
    # cast/type: FastAPI has add_middleware
    app.add_middleware(RequestLoggingMiddleware)  # type: ignore[attr-defined]
