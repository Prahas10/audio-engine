from fastapi import HTTPException

from core.library import load_library_metadata
from core.queue_manager import score_next_track
from core.planner import plan_transition_logic
from core.renderer import render_dj_transition
from core.setlist_state import clear_setlist_state, add_transition_to_setlist
from core.playback_assembly import assemble_playback_from_setlist
from models.schemas import FXParameters
from core.queue_state import load_queue_state


def pick_best_next_track(current_track, remaining_tracks, preferred_mix_duration=None):
    scored = []
    for track in remaining_tracks:
        score_data = score_next_track(
            current_track=current_track,
            candidate_track=track,
            preferred_mix_duration=preferred_mix_duration
        )
        scored.append(score_data)
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
    return ranked[0], ranked


def build_smart_track_order(library, starting_track_id=None, preferred_mix_duration=None):
    if not library:
        raise HTTPException(status_code=400, detail="Library is empty. Scan tracks first.")

    tracks = list(library.values())

    if len(tracks) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tracks to build a set.")

    if starting_track_id:
        if starting_track_id not in library:
            raise HTTPException(status_code=404, detail=f"Starting track not found: {starting_track_id}")
        current_track = library[starting_track_id]
    else:
        current_track = sorted(
            tracks,
            key=lambda t: (len(t.get("phrase_points", [])), t.get("duration", 0)),
            reverse=True
        )[0]

    ordered_tracks = [current_track]
    remaining_tracks = [t for t in tracks if t["track_id"] != current_track["track_id"]]
    routing_debug = []

    while remaining_tracks:
        best_next, ranked_candidates = pick_best_next_track(
            current_track=current_track,
            remaining_tracks=remaining_tracks,
            preferred_mix_duration=preferred_mix_duration
        )
        next_track = library[best_next["track_id"]]
        routing_debug.append({
            "from_track": current_track["filename"],
            "selected_next": next_track["filename"],
            "selected_score": best_next["score"],
            "selected_reasons": best_next["reasons"],
            "top_candidates": ranked_candidates[:5]
        })
        ordered_tracks.append(next_track)
        remaining_tracks = [t for t in remaining_tracks if t["track_id"] != next_track["track_id"]]
        current_track = next_track

    return ordered_tracks, routing_debug


def seconds_to_mmss(seconds):
    seconds = float(seconds or 0)
    minutes = int(seconds // 60)
    secs = int(round(seconds % 60))
    return f"{minutes}:{secs:02d}"


def seconds_to_timecode(seconds):
    seconds = int(round(float(seconds or 0)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def build_set_timeline(ordered_tracks, transition_results):
    timeline = []
    current_set_time = 0.0

    for idx, track in enumerate(ordered_tracks):
        # 1. Record the audible start time
        timeline.append({
            "position": idx + 1,
            "filename": track["filename"],
            "start_in_set": seconds_to_mmss(current_set_time),
            "start_in_set_seconds": round(current_set_time, 3)
        })

        # 2. To find the NEXT start time, we need the CURRENT track's entry point
        if idx < len(transition_results):
            # This transition (idx) tells us how the CURRENT track (A) ends 
            # and the NEXT track (B) begins.
            this_render = transition_results[idx].get("render", {})
            a_out = float(this_render.get("snapped_transition_start_time") or 0)

            # --- THE FIX ---
            # We need the entry point used for the track currently playing (ordered_tracks[idx])
            # For the first track, entry is 0. 
            # For all others, the entry point was defined in the PREVIOUS transition.
            if idx == 0:
                current_entry = 0.0
            else:
                prev_render = transition_results[idx-1].get("render", {})
                current_entry = float(prev_render.get("track_b_entry_time") or 0)

            # Calculation for the next track's start:
            # We add the duration of the current song's "lead time" (Out Point - Entry Point)
            current_set_time += (a_out - current_entry)

    return timeline

def render_queue_order_set(
    queue_path,
    library_path,
    setlist_path,
    preferred_mix_duration=None,
    output_dir="outputs",
    final_output_path="outputs/final_set.wav"
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
    for track_id in ordered_ids:
        if track_id not in library:
            raise HTTPException(status_code=404, detail=f"Track not found in library: {track_id}")
        ordered_tracks.append(library[track_id])

    clear_setlist_state(setlist_path)
    transition_results = []

    for idx in range(len(ordered_tracks) - 1):
        track_a = ordered_tracks[idx]
        track_b = ordered_tracks[idx + 1]

        plan    = plan_transition_logic(
            track_a_path=track_a["path"],
            track_b_path=track_b["path"],
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

        setlist_record = add_transition_to_setlist(
            current_track_id=track_a["track_id"],
            next_track_id=track_b["track_id"],
            transition_file=render_result["audio_clip_url"],
            transition_start_time=render_result["snapped_transition_start_time"],
            track_b_entry_time=render_result["track_b_entry_time"],
            strategy=render_result["transition_strategy"],
            setlist_path=setlist_path
        )

        transition_results.append({
            "from_track":     track_a,
            "to_track":       track_b,
            "plan":           plan,
            "render":         render_result,
            "setlist_record": setlist_record
        })

    final_mix = assemble_playback_from_setlist(
        setlist_path=setlist_path,
        library_path=library_path,
        output_path=final_output_path
    )
    timeline = build_set_timeline(ordered_tracks, transition_results)

    return {
        "status":           "success",
        "mode":             "queue_order",
        "track_count":      len(ordered_tracks),
        "ordered_tracks": [
            {
                "position": idx + 1,
                "track_id": track["track_id"],
                "filename": track["filename"],
                "bpm":      track.get("bpm"),
                "key":      track.get("key"),
                "camelot":  track.get("camelot"),
            }
            for idx, track in enumerate(ordered_tracks)
        ],
        "transition_count": len(transition_results),
        "transitions":      transition_results,
        "final_mix":        final_mix,
        "timeline":         timeline,
    }


def build_and_render_smart_set(
    library_path,
    setlist_path,
    starting_track_id=None,
    preferred_mix_duration=None,
    output_dir="outputs",
    final_output_path="outputs/final_set.wav",
    render=True
):
    library = load_library_metadata(library_path)

    ordered_tracks, routing_debug = build_smart_track_order(
        library=library,
        starting_track_id=starting_track_id,
        preferred_mix_duration=preferred_mix_duration
    )

    clear_setlist_state(setlist_path)
    transition_results = []

    for idx in range(len(ordered_tracks) - 1):
        track_a = ordered_tracks[idx]
        track_b = ordered_tracks[idx + 1]

        plan = plan_transition_logic(
            track_a_path=track_a["path"],
            track_b_path=track_b["path"],
            preferred_mix_duration=preferred_mix_duration
        )

        transition_record = {
            "from_track":     track_a,
            "to_track":       track_b,
            "plan":           plan,
            "render":         None,
            "setlist_record": None
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

            setlist_record = add_transition_to_setlist(
                current_track_id=track_a["track_id"],
                next_track_id=track_b["track_id"],
                transition_file=render_result["audio_clip_url"],
                transition_start_time=render_result["snapped_transition_start_time"],
                track_b_entry_time=render_result["track_b_entry_time"],
                strategy=render_result["transition_strategy"],
                setlist_path=setlist_path
            )

            transition_record["render"]         = render_result
            transition_record["setlist_record"] = setlist_record

        transition_results.append(transition_record)

    final_mix = None
    if render:
        final_mix = assemble_playback_from_setlist(
            setlist_path=setlist_path,
            library_path=library_path,
            output_path=final_output_path
        )

    timeline = build_set_timeline(ordered_tracks, transition_results)

    return {
        "status":          "success",
        "track_count":     len(ordered_tracks),
        "optimized_order": [
            {
                "position": idx + 1,
                "track_id": track["track_id"],
                "filename": track["filename"],
                "bpm":      track.get("bpm"),
                "key":      track.get("key"),
                "camelot":  track.get("camelot"),
            }
            for idx, track in enumerate(ordered_tracks)
        ],
        "routing_debug":    routing_debug,
        "transition_count": len(transition_results),
        "transitions":      transition_results,
        "final_mix":        final_mix,
        "timeline":         timeline,
    }