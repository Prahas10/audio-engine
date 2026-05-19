from fastapi import APIRouter

from models.schemas import SmartRouteRequest
from core.smart_router import smart_route_transition


router = APIRouter(prefix="/smart", tags=["Smart Router"])

@router.post("/route-transition")
async def route_transition(req: SmartRouteRequest):
    return smart_route_transition(
        current_track_id=req.current_track_id,
        library_path=req.library_path,
        preferred_mix_duration=req.preferred_mix_duration,
        output_dir=req.output_dir,
        render=req.render
    )