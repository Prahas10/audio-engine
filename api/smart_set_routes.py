from fastapi import APIRouter

from models.schemas import SmartSetBuildRequest
from core.smart_set_builder import build_and_render_smart_set
from models.schemas import QueueOrderSetRenderRequest
from core.smart_set_builder import build_and_render_smart_set, render_queue_order_set

router = APIRouter(prefix="/set", tags=["Smart Set Builder"])


@router.post("/build-and-render")
async def build_and_render(req: SmartSetBuildRequest):
    return build_and_render_smart_set(
        library_path=req.library_path,
        setlist_path=req.setlist_path,
        starting_track_id=req.starting_track_id,
        preferred_mix_duration=req.preferred_mix_duration,
        output_dir=req.output_dir,
        final_output_path=req.final_output_path,
        render=req.render
    )

@router.post("/render-queue-order")
async def render_queue_order(req: QueueOrderSetRenderRequest):
    return render_queue_order_set(
        queue_path=req.queue_path,
        library_path=req.library_path,
        setlist_path=req.setlist_path,
        preferred_mix_duration=req.preferred_mix_duration,
        output_dir=req.output_dir,
        final_output_path=req.final_output_path
    )