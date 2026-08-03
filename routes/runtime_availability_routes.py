"""Read-only installation status for selectable Odysseus runtimes."""

from fastapi import APIRouter

from src.runtime_availability import runtime_availability


router = APIRouter(tags=["runtime-availability"])


@router.get("/api/runtime/availability")
async def get_runtime_availability():
    return {"runtimes": runtime_availability()}

