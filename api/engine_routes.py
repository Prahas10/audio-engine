from fastapi import APIRouter
from models.schemas import AutoRenderRequest
from core.planner import plan_transition_logic
from core.renderer import render_dj_transition
from models.schemas import FXParameters

router = APIRouter()

# FastAPI endpoint that combines 'Plan' and 'Render' into a single automated 'One-Click' transition request.
@router.post("/v1/autodj/render-planned-transition")
async def render_planned_transition(req: AutoRenderRequest):
    plan = plan_transition_logic(
        track_a_path=req.track_a_path,
        track_b_path=req.track_b_path,
        preferred_mix_duration=req.preferred_mix_duration
    )

    payload = plan["render_payload"]

    result = render_dj_transition(
        track_a_path=payload["track_a_path"],
        track_b_path=payload["track_b_path"],
        transition_start_time=payload["transition_start_time"],
        track_b_entry_time=payload.get("track_b_entry_time"),
        mix_duration=payload["mix_duration"],
        output_dir=req.output_dir,
        transition_strategy=payload["transition_strategy"],
        fx_parameters=FXParameters(**payload["fx_parameters"])
    )

    return {
        "status": "success",
        "plan": plan,
        "render": result
    }