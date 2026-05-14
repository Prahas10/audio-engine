from fastapi import APIRouter

from models.schemas import AssemblePlaybackRequest
from core.playback_assembly import assemble_playback_from_setlist


router = APIRouter(prefix="/playback", tags=["Playback"])


@router.post("/assemble")
async def assemble_playback(req: AssemblePlaybackRequest):
    return assemble_playback_from_setlist(
        setlist_path=req.setlist_path,
        library_path=req.library_path,
        output_path=req.output_path
    )