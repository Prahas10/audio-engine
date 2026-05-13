from fastapi import APIRouter

from models.schemas import RecommendNextRequest
from core.queue_manager import recommend_next_tracks


router = APIRouter(prefix="/queue", tags=["Queue"])


@router.post("/recommend-next")
async def recommend_next(req: RecommendNextRequest):
    return recommend_next_tracks(
        current_track_id=req.current_track_id,
        library_path=req.library_path,
        preferred_mix_duration=req.preferred_mix_duration,
        max_results=req.max_results
    )