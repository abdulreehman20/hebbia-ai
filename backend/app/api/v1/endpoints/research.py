"""Financial research SSE endpoint."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.ai.agents.orchestrator import FinancialResearchAgent
from app.schemas.research import ResearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


def _event_stream(request: ResearchRequest) -> Iterator[dict[str, str]]:
    agent = FinancialResearchAgent()
    try:
        yield from (
            {
                "event": str(event.get("event") or "message"),
                "data": json.dumps(event, default=str),
            }
            for event in agent.stream_events(request)
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research stream failed")
        yield {
            "event": "research_failed",
            "data": json.dumps({"event": "research_failed", "error": str(exc)}),
        }


@router.post("/stream")
async def research_stream(request: ResearchRequest) -> EventSourceResponse:
    """Run the Financial Research Agent and stream validated progress + final answer."""
    return EventSourceResponse(_event_stream(request))
