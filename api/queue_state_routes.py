from fastapi import APIRouter

from models.schemas import (
    CreateQueueRequest,
    AddTrackToQueueRequest,
    AdvanceQueueRequest
)

from core.queue_state import (
    create_queue_state,
    add_track_to_queue,
    get_queue_state,
    advance_queue,
    clear_queue
)


router = APIRouter(prefix="/queue-state", tags=["Queue State"])


@router.post("/create")
async def create_queue(req: CreateQueueRequest):
    return create_queue_state(req.queue_path)


@router.post("/add-track")
async def add_track(req: AddTrackToQueueRequest):
    return add_track_to_queue(
        track_id=req.track_id,
        queue_path=req.queue_path,
        library_path=req.library_path
    )


@router.get("/state")
async def state(
    queue_path: str = "storage/queue_state.json",
    library_path: str = "storage/library_metadata.json"
):
    return get_queue_state(
        queue_path=queue_path,
        library_path=library_path
    )


@router.post("/advance")
async def advance(req: AdvanceQueueRequest):
    return advance_queue(req.queue_path)


@router.post("/clear")
async def clear(req: CreateQueueRequest):
    return clear_queue(req.queue_path)