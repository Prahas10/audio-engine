from fastapi import HTTPException

from core.library import load_library_metadata
from core.queue_state import load_queue_state, advance_queue
from core.planner import plan_transition_logic
from core.renderer import render_dj_transition
from models.schemas import FXParameters
from core.setlist_state import add_transition_to_setlist


def queue_smart_render(
    queue_path,
    library_path,
    preferred_mix_duration=None,
    output_dir="rendered_clips",
    auto_advance=False,
):
    library = load_library_metadata(library_path)
    state   = load_queue_state(queue_path)

    current_track_id   = state.get("current_track_id")
    upcoming_track_ids = state.get("upcoming_track_ids", [])

    if current_track_id is None:
        raise HTTPException(status_code=400, detail="No current track set in queue.")
    if not upcoming_track_ids:
        raise HTTPException(status_code=400, detail="No upcoming track available in queue.")

    next_track_id = upcoming_track_ids[0]

    if current_track_id not in library:
        raise HTTPException(status_code=404, detail=f"Current track not found: {current_track_id}")
    if next_track_id not in library:
        raise HTTPException(status_code=404, detail=f"Next track not found: {next_track_id}")

    current_track = library[current_track_id]
    next_track    = library[next_track_id]

    plan    = plan_transition_logic(
        track_a_path=current_track["path"],
        track_b_path=next_track["path"],
        preferred_mix_duration=preferred_mix_duration,
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
        fx_parameters=FXParameters(**payload["fx_parameters"]),
    )

    updated_queue = advance_queue(queue_path) if auto_advance else None

    setlist_record = add_transition_to_setlist(
        current_track_id=current_track_id,
        next_track_id=next_track_id,
        transition_file=render_result["audio_clip_url"],
        transition_start_time=render_result["snapped_transition_start_time"],
        track_b_entry_time=render_result["track_b_entry_time"],
        strategy=render_result["transition_strategy"],
        mix_duration=render_result["duration_seconds"],   # FIX
        setlist_path="storage/setlist_state.json",
    )

    return {
        "status":         "success",
        "current_track":  current_track,
        "next_track":     next_track,
        "plan":plan,
        "render":render_result,
        "auto_advanced":  auto_advance,
        "updated_queue":  updated_queue,
        "setlist_record": setlist_record,
    }