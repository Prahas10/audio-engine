import os
import json
from fastapi import HTTPException

from core.library import load_library_metadata


# Creates a fresh queue state file
def create_queue_state(queue_path):
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)

    state = {
        "current_track_id": None,
        "upcoming_track_ids": []
    }

    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return {
        "status": "success",
        "queue_path": os.path.abspath(queue_path),
        **state
    }


# Loads queue state from disk
def load_queue_state(queue_path):
    if not os.path.exists(queue_path):
        return create_queue_state(queue_path)

    with open(queue_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Saves queue state to disk
def save_queue_state(queue_path, state):
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)

    with open(queue_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return state


# Adds a track from the library into the upcoming queue
def add_track_to_queue(track_id, queue_path, library_path):
    library = load_library_metadata(library_path)

    if track_id not in library:
        raise HTTPException(
            status_code=404,
            detail=f"Track not found in library: {track_id}"
        )

    state = load_queue_state(queue_path)

    if state["current_track_id"] is None:
        state["current_track_id"] = track_id
    else:
        state["upcoming_track_ids"].append(track_id)

    save_queue_state(queue_path, state)

    return {
        "status": "success",
        "queue_path": os.path.abspath(queue_path),
        "current_track_id": state["current_track_id"],
        "upcoming_track_ids": state["upcoming_track_ids"]
    }


# Returns queue state with basic track metadata attached
def get_queue_state(queue_path, library_path):
    library = load_library_metadata(library_path)
    state = load_queue_state(queue_path)

    current_track = None

    if state["current_track_id"] is not None:
        current_track = library.get(state["current_track_id"])

    upcoming_tracks = [
        library.get(track_id)
        for track_id in state["upcoming_track_ids"]
        if track_id in library
    ]

    return {
        "status": "success",
        "queue_path": os.path.abspath(queue_path),
        "current_track_id": state["current_track_id"],
        "current_track": current_track,
        "upcoming_track_ids": state["upcoming_track_ids"],
        "upcoming_tracks": upcoming_tracks
    }


# Advances queue so the next upcoming track becomes current
def advance_queue(queue_path):
    state = load_queue_state(queue_path)

    if not state["upcoming_track_ids"]:
        return {
            "status": "empty",
            "message": "No upcoming tracks to advance.",
            "queue_path": os.path.abspath(queue_path),
            "current_track_id": state["current_track_id"],
            "upcoming_track_ids": state["upcoming_track_ids"]
        }

    state["current_track_id"] = state["upcoming_track_ids"].pop(0)

    save_queue_state(queue_path, state)

    return {
        "status": "success",
        "queue_path": os.path.abspath(queue_path),
        "current_track_id": state["current_track_id"],
        "upcoming_track_ids": state["upcoming_track_ids"]
    }


# Clears the queue and removes current/upcoming state
def clear_queue(queue_path):
    return create_queue_state(queue_path)