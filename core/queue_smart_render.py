from fastapi import HTTPException

from core.library import load_library_metadata
from core.queue_state import load_queue_state, advance_queue
from core.planner import plan_transition_logic
from core.renderer import render_dj_transition
from models.schemas import FXParameters


# Plans and renders transition from current queue track to next queued track
def queue_smart_render(
    queue_path,
    library_path,
    preferred_mix_duration=30,
    output_dir="rendered_clips",
    auto_advance=False
):
    library = load_library_metadata(library_path)
    state = load_queue_state(queue_path)

    current_track_id = state.get("current_track_id")
    upcoming_track_ids = state.get("upcoming_track_ids", [])

    if current_track_id is None:
        raise HTTPException(status_code=400, detail="No current track set in queue.")

    if not upcoming_track_ids:
        raise HTTPException(status_code=400, detail="No upcoming track available in queue.")

    next_track_id = upcoming_track_ids[0]

    if current_track_id not in library:
        raise HTTPException(status_code=404, detail=f"Current track not found in library: {current_track_id}")

    if next_track_id not in library:
        raise HTTPException(status_code=404, detail=f"Next track not found in library: {next_track_id}")

    current_track = library[current_track_id]
    next_track = library[next_track_id]

    plan = plan_transition_logic(
        track_a_path=current_track["path"],
        track_b_path=next_track["path"],
        preferred_mix_duration=preferred_mix_duration
    )

    payload = plan["render_payload"]

    render_result = render_dj_transition(
        track_a_path=payload["track_a_path"],
        track_b_path=payload["track_b_path"],
        transition_start_time=payload["transition_start_time"],
        track_b_entry_time=payload.get("track_b_entry_time"),
        mix_duration=payload["mix_duration"],
        output_dir=output_dir,
        transition_strategy=payload["transition_strategy"],
        fx_parameters=FXParameters(**payload["fx_parameters"])
    )

    updated_queue = None

    if auto_advance:
        updated_queue = advance_queue(queue_path)

    return {
        "status": "success",
        "current_track": current_track,
        "next_track": next_track,
        "plan": plan,
        "render": render_result,
        "auto_advanced": auto_advance,
        "updated_queue": updated_queue
    }