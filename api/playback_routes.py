from fastapi import APIRouter

from models.schemas import AssemblePlaybackRequest
from core.library import load_library_metadata
from core.queue_state import load_queue_state
from core.playback_assembly import render_continuous_set_from_ordered_tracks


router = APIRouter(prefix="/playback", tags=["Playback"])


@router.post("/assemble")
async def assemble_playback(req: AssemblePlaybackRequest):
    """
    Continuous DJ-set renderer.

    IMPORTANT:
    This no longer assembles transition clips from a setlist.

    It renders one continuous timeline directly so:
    - timestretching stays continuous
    - no gaps after transitions
    - Track B continues from the exact stretched timeline
    """

    library = load_library_metadata(req.library_path)
    queue = load_queue_state(req.queue_path)

    ordered_ids = []

    if queue.get("current_track_id"):
        ordered_ids.append(queue["current_track_id"])

    ordered_ids.extend(queue.get("upcoming_track_ids", []))

    if len(ordered_ids) < 2:
        return {
            "status": "error",
            "detail": "Queue needs at least 2 tracks."
        }

    ordered_tracks = []

    for tid in ordered_ids:
        if tid not in library:
            return {
                "status": "error",
                "detail": f"Track not found in library: {tid}"
            }

        ordered_tracks.append(library[tid])

    result = render_continuous_set_from_ordered_tracks(
        ordered_tracks=ordered_tracks,
        preferred_mix_duration=req.preferred_mix_duration,
        output_path=req.output_path,
    )

    return result