"""
core/smart_set_builder.py

Changes in this version:
  - build_smart_track_order: passes set_position to score_next_track so the
    energy arc scorer knows where in the set each selection happens.
    set_position = (tracks_placed) / (total_tracks - 1), 0.0 to 1.0.
  - All other logic unchanged.
"""

from fastapi import HTTPException

from core.library import load_library_metadata
from core.queue_manager import score_next_track
from core.planner import plan_transition_logic
from core.renderer import render_dj_transition, render_continuous_set_from_ordered_tracks
from core.setlist_state import clear_setlist_state, add_transition_to_setlist
from core.playback_assembly import render_continuous_set_from_ordered_tracks
from core.queue_state import load_queue_state
from models.schemas import FXParameters


def pick_best_next_track(current_track, remaining_tracks,
                         preferred_mix_duration=None, set_position: float = 0.5):
    scored = [
        score_next_track(current_track, t, preferred_mix_duration, set_position=set_position)
        for t in remaining_tracks
    ]
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    return ranked[0], ranked


def build_smart_track_order(library, starting_track_id=None, preferred_mix_duration=None):
    if not library:
        raise HTTPException(status_code=400, detail="Library is empty.")
    tracks = list(library.values())
    if len(tracks) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tracks.")

    if starting_track_id:
        if starting_track_id not in library:
            raise HTTPException(status_code=404, detail=f"Starting track not found: {starting_track_id}")
        current_track = library[starting_track_id]
    else:
        # Pick the track with the most phrase points and moderate energy as opener
        current_track = sorted(
            tracks,
            key=lambda t: (
                len(t.get("phrase_points", [])),
                t.get("energy_summary", {}).get("avg", 0.5),
            ),
            reverse=True,
        )[0]

    total_tracks  = len(tracks)
    ordered_tracks = [current_track]
    remaining      = [t for t in tracks if t["track_id"] != current_track["track_id"]]
    routing_debug  = []

    while remaining:
        # set_position: how far through the set are we right now
        tracks_placed = len(ordered_tracks) - 1
        total_gaps    = total_tracks - 1
        set_position  = tracks_placed / max(total_gaps, 1)

        best_next, ranked = pick_best_next_track(
            current_track, remaining, preferred_mix_duration,
            set_position=set_position,
        )
        next_track = library[best_next["track_id"]]
        routing_debug.append({
            "from_track":       current_track["filename"],
            "selected_next":    next_track["filename"],
            "selected_score":   best_next["score"],
            "set_position":     round(set_position, 3),
            "energy_arc_score": best_next.get("energy_arc_score"),
            "selected_reasons": best_next["reasons"],
            "top_candidates":   ranked[:5],
        })
        ordered_tracks.append(next_track)
        remaining      = [t for t in remaining if t["track_id"] != next_track["track_id"]]
        current_track  = next_track

    return ordered_tracks, routing_debug


def seconds_to_mmss(seconds):
    seconds = float(seconds or 0)
    return f"{int(seconds // 60)}:{int(round(seconds % 60)):02d}"


def build_and_render_smart_set(
    library_path,
    setlist_path=None,
    starting_track_id=None,
    preferred_mix_duration=None,
    output_dir="outputs",
    final_output_path="outputs/final_set.wav",
    render=True,
):
    library = load_library_metadata(library_path)

    ordered_tracks, routing_debug = build_smart_track_order(
        library=library,
        starting_track_id=starting_track_id,
        preferred_mix_duration=preferred_mix_duration,
    )

    if len(ordered_tracks) < 2:
        raise HTTPException(status_code=400, detail="Smart set needs at least 2 tracks.")

    if render:
        result = render_continuous_set_from_ordered_tracks(
            ordered_tracks=ordered_tracks,
            preferred_mix_duration=preferred_mix_duration,
            output_path=final_output_path,
        )
        transition_results = result["transitions"]
        timeline           = result["timeline"]
        final_mix = {
            "status":           "success",
            "output_path":      result["output_path"],
            "duration_seconds": result["duration_seconds"],
            "transition_count": result["transition_count"],
        }
    else:
        transition_results = []
        for idx in range(len(ordered_tracks) - 1):
            track_a = ordered_tracks[idx]
            track_b = ordered_tracks[idx + 1]
            plan = plan_transition_logic(
                track_a_path=track_a["path"],
                track_b_path=track_b["path"],
                preferred_mix_duration=preferred_mix_duration,
            )
            transition_results.append({
                "index": idx + 1, "from_track": track_a,
                "to_track": track_b, "plan": plan, "render": None,
            })
        timeline  = []
        final_mix = None

    return {
        "status": "success",
        "mode":   "smart_set_continuous_timestretch" if render else "smart_set_plan_only",
        "track_count": len(ordered_tracks),
        "optimized_order": [
            {"position": i+1, "track_id": t["track_id"], "filename": t["filename"],
             "bpm": t.get("bpm"), "key": t.get("key"), "camelot": t.get("camelot")}
            for i, t in enumerate(ordered_tracks)
        ],
        "routing_debug":    routing_debug,
        "transition_count": len(transition_results),
        "transitions":      transition_results,
        "timeline":         timeline,
        "final_mix":        final_mix,
    }

def render_queue_order_set(
    queue_path, library_path,
    setlist_path=None,
    preferred_mix_duration=None,
    output_dir="outputs",
    final_output_path="outputs/final_set.wav",
):
    library = load_library_metadata(library_path)
    queue   = load_queue_state(queue_path)
    ordered_ids = []
    if queue.get("current_track_id"):
        ordered_ids.append(queue["current_track_id"])
    ordered_ids.extend(queue.get("upcoming_track_ids", []))
    if len(ordered_ids) < 2:
        raise HTTPException(status_code=400, detail="Queue needs at least 2 tracks.")
    ordered_tracks = []
    for tid in ordered_ids:
        if tid not in library:
            raise HTTPException(status_code=404, detail=f"Track not found: {tid}")
        ordered_tracks.append(library[tid])
    result = render_continuous_set_from_ordered_tracks(
        ordered_tracks=ordered_tracks,
        preferred_mix_duration=preferred_mix_duration,
        output_path=final_output_path,
    )
    return {
        "status":     "success",
        "mode":       "queue_order_continuous_timestretch",
        "track_count": len(ordered_tracks),
        "ordered_tracks": [
            {"position": i+1, "track_id": t["track_id"], "filename": t["filename"],
             "bpm": t.get("bpm"), "key": t.get("key"), "camelot": t.get("camelot")}
            for i, t in enumerate(ordered_tracks)
        ],
        "transition_count": result["transition_count"],
        "transitions":      result["transitions"],
        "timeline":         result["timeline"],
        "final_mix": {
            "status":           "success",
            "output_path":      result["output_path"],
            "duration_seconds": result["duration_seconds"],
            "transition_count": result["transition_count"],
        },
    }