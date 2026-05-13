from fastapi import APIRouter

from models.schemas import SetlistAddTransitionRequest, SetlistStateRequest

from core.setlist_state import (
    add_transition_to_setlist,
    get_setlist_state,
    clear_setlist_state
)


router = APIRouter(prefix="/setlist", tags=["Setlist"])


@router.post("/add-transition")
async def add_transition(req: SetlistAddTransitionRequest):
    return add_transition_to_setlist(
        current_track_id=req.current_track_id,
        next_track_id=req.next_track_id,
        transition_file=req.transition_file,
        transition_start_time=req.transition_start_time,
        track_b_entry_time=req.track_b_entry_time,
        strategy=req.strategy,
        setlist_path=req.setlist_path
    )


@router.get("/state")
async def state(setlist_path: str = "storage/setlist_state.json"):
    return get_setlist_state(setlist_path)


@router.post("/clear")
async def clear(req: SetlistStateRequest):
    return clear_setlist_state(req.setlist_path)