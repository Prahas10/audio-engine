from fastapi import HTTPException

from core.library import load_library_metadata
from core.queue_manager import score_next_track
from core.planner import plan_transition_logic
from core.renderer import render_dj_transition
from core.setlist_state import clear_setlist_state, add_transition_to_setlist
from core.playback_assembly import *
from models.schemas import FXParameters
from core.queue_state import load_queue_state


def pick_best_next_track(current_track, remaining_tracks, preferred_mix_duration=None):
    scored = [
        score_next_track(current_track, t, preferred_mix_duration)
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
        current_track = sorted(
            tracks,
            key=lambda t: (len(t.get("phrase_points", [])), t.get("duration", 0)),
            reverse=True,
        )[0]

    ordered_tracks  = [current_track]
    remaining       = [t for t in tracks if t["track_id"] != current_track["track_id"]]
    routing_debug   = []

    while remaining:
        best_next, ranked = pick_best_next_track(current_track, remaining, preferred_mix_duration)
        next_track = library[best_next["track_id"]]
        routing_debug.append({
            "from_track":      current_track["filename"],
            "selected_next":   next_track["filename"],
            "selected_score":  best_next["score"],
            "selected_reasons": best_next["reasons"],
            "top_candidates":  ranked[:5],
        })
        ordered_tracks.append(next_track)
        remaining    = [t for t in remaining if t["track_id"] != next_track["track_id"]]
        current_track = next_track

    return ordered_tracks, routing_debug


def seconds_to_mmss(seconds):
    seconds = float(seconds or 0)
    return f"{int(seconds // 60)}:{int(round(seconds % 60)):02d}"


def build_set_timeline(ordered_tracks, transition_results):
    """
    Timeline formula:
        track[0] starts at 0
        track[N] starts at track[N-1].start + (A_out_point - A_entry_point)

    A_entry_point for track[0] is 0.
    A_entry_point for track[N>0] is the track_b_entry_time of the previous transition.
    """
    timeline            = []
    current_set_time    = 0.0

    for idx, track in enumerate(ordered_tracks):
        timeline.append({
            "position":             idx + 1,
            "filename":             track["filename"],
            "start_in_set":         seconds_to_mmss(current_set_time),
            "start_in_set_seconds": round(current_set_time, 3),
            "bpm":                  track.get("bpm"),
            "key":                  track.get("key"),
            "camelot":              track.get("camelot"),
        })

        if idx < len(transition_results):
            render       = (transition_results[idx].get("render") or {})
            a_out        = float(render.get("snapped_transition_start_time") or 0)
            a_entry      = 0.0 if idx == 0 else float(
                (transition_results[idx - 1].get("render") or {}).get("track_b_entry_time") or 0
            )
            current_set_time += (a_out - a_entry)

    return timeline


def _render_transition(track_a, track_b, preferred_mix_duration, output_dir,
                       setlist_path):
    plan    = plan_transition_logic(
        track_a_path=track_a["path"],
        track_b_path=track_b["path"],
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
        planner_metadata=plan,
    )

    setlist_record = add_transition_to_setlist(
        current_track_id=track_a["track_id"],
        next_track_id=track_b["track_id"],
        transition_file=render_result["audio_clip_url"],
        transition_start_time=render_result["snapped_transition_start_time"],
        track_b_entry_time=render_result["track_b_entry_time"],
        strategy=render_result["transition_strategy"],
        mix_duration=render_result["duration_seconds"],   # FIX: store for assembly
        setlist_path=setlist_path,track_b_suffix_file=render_result.get("track_b_suffix_file"),
track_b_suffix_start_time=render_result.get("track_b_suffix_start_time"),
track_b_suffix_duration=render_result.get("track_b_suffix_duration"),
    )

    return plan, render_result, setlist_record


def render_queue_order_set(
    queue_path,
    library_path,
    setlist_path=None,
    preferred_mix_duration=None,
    output_dir="outputs",
    final_output_path="outputs/final_set.wav",
):
    library = load_library_metadata(library_path)
    queue = load_queue_state(queue_path)

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
        "status": "success",
        "mode": "queue_order_continuous_timestretch",
        "track_count": len(ordered_tracks),
        "ordered_tracks": [
            {
                "position": i + 1,
                "track_id": t["track_id"],
                "filename": t["filename"],
                "bpm": t.get("bpm"),
                "key": t.get("key"),
                "camelot": t.get("camelot"),
            }
            for i, t in enumerate(ordered_tracks)
        ],
        "transition_count": result["transition_count"],
        "transitions": result["transitions"],
        "timeline": result["timeline"],   # FIX
        "final_mix": {
            "status": "success",
            "output_path": result["output_path"],
            "duration_seconds": result["duration_seconds"],
            "transition_count": result["transition_count"],
        },
    }

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
        timeline = result["timeline"]

        final_mix = {
            "status": "success",
            "output_path": result["output_path"],
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
                "index": idx + 1,
                "from_track": track_a,
                "to_track": track_b,
                "plan": plan,
                "render": None,
            })

        timeline = []
        final_mix = None

    return {
        "status": "success",
        "mode": "smart_set_continuous_timestretch" if render else "smart_set_plan_only",
        "track_count": len(ordered_tracks),
        "optimized_order": [
            {
                "position": i + 1,
                "track_id": t["track_id"],
                "filename": t["filename"],
                "bpm": t.get("bpm"),
                "key": t.get("key"),
                "camelot": t.get("camelot"),
            }
            for i, t in enumerate(ordered_tracks)
        ],
        "routing_debug": routing_debug,
        "transition_count": len(transition_results),
        "transitions": transition_results,
        "timeline": timeline,  # FIX
        "final_mix": final_mix,
    }