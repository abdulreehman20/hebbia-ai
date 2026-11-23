"""Financial research SSE endpoint (wired; agent code not migrated yet)."""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/stream")
async def research_stream() -> None:
    raise HTTPException(
        status_code=501,
        detail="Research agent not migrated into app.ai yet",
    )
