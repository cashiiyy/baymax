"""
BAYMAX AI – API Middleware
============================
FastAPI middleware for CORS, request logging, and error handling.
"""

import time
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger

log = get_logger(__name__)


def add_middleware(app) -> None:
    """Register all middleware on the FastAPI app."""

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all for local dev
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Logging & Timing
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        # Skip health check logging to reduce noise
        is_health = request.url.path == "/health"

        if not is_health:
            log.info("Request started | method={} path={}", request.method, request.url.path)

        try:
            response = await call_next(request)
            elapsed = time.time() - start_time
            if not is_health:
                log.info(
                    "Request complete | method={} path={} status={} ms={:.1f}",
                    request.method,
                    request.url.path,
                    response.status_code,
                    elapsed * 1000,
                )
            return response

        except Exception as exc:
            elapsed = time.time() - start_time
            log.exception(
                "Request failed | method={} path={} ms={:.1f} error={}",
                request.method,
                request.url.path,
                elapsed * 1000,
                exc,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error", "error": str(exc)},
            )
