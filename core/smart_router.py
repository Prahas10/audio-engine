from fastapi import HTTPException

from core.library import load_library_metadata
from core.queue_manager import recommend_next_tracks
from core.planner import plan_transition_logic
from core.renderer import render_dj_transition
from models.schemas import FXParameters


# Picks the best next track, plans the transition, and optionally renders it
def smart_route_transition(
    current_track_id,
    library_path,
    preferred_mix_duration=None,
    output_dir="rendered_clips",
    render=False
):
    library = load_library_metadata(library_path)

    if current_track_id not in library:
        raise HTTPException(
            status_code=404,
            detail=f"Current track_id not found in library: {current_track_id}"
        )

    current_track = library[current_track_id]

    recommendations = recommend_next_tracks(
        current_track_id=current_track_id,
        library_path=library_path,
        preferred_mix_duration=preferred_mix_duration,
        max_results=5
    )

    if not recommendations["recommendations"]:
        raise HTTPException(
            status_code=400,
            detail="No candidate tracks available in library."
        )

    next_track = recommendations["recommendations"][0]

    plan = plan_transition_logic(
        track_a_path=current_track["path"],
        track_b_path=next_track["path"],
        preferred_mix_duration=preferred_mix_duration
    )

    response = {
        "status": "success",
        "current_track": current_track,
        "selected_next_track": next_track,
        "plan": plan,
        "rendered": False,
        "render": None
    }

    if render:
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

        response["rendered"] = True
        response["render"] = render_result

    return response