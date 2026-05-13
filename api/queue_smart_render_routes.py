from fastapi import APIRouter

from models.schemas import QueueSmartRenderRequest
from core.queue_smart_render import queue_smart_render


router = APIRouter(prefix="/queue", tags=["Queue Smart Render"])


@router.post("/smart-render")
async def smart_render(req: QueueSmartRenderRequest):
    return queue_smart_render(
        queue_path=req.queue_path,
        library_path=req.library_path,
        preferred_mix_duration=req.preferred_mix_duration,
        output_dir=req.output_dir,
        auto_advance=req.auto_advance
    )