import os
import json


# Creates a new empty setlist state file
def create_setlist_state(setlist_path):
    os.makedirs(os.path.dirname(setlist_path), exist_ok=True)

    state = {
        "transitions": []
    }

    with open(setlist_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return {
        "status": "success",
        "setlist_path": os.path.abspath(setlist_path),
        **state
    }


# Loads setlist state from disk
def load_setlist_state(setlist_path):
    if not os.path.exists(setlist_path):
        return create_setlist_state(setlist_path)

    with open(setlist_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Saves setlist state to disk
def save_setlist_state(setlist_path, state):
    os.makedirs(os.path.dirname(setlist_path), exist_ok=True)

    with open(setlist_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return state


# Adds a rendered transition to the setlist timeline
def add_transition_to_setlist(
    current_track_id,
    next_track_id,
    transition_file,
    transition_start_time,
    track_b_entry_time,
    strategy,
    setlist_path
):
    state = load_setlist_state(setlist_path)

    transition_record = {
        "current_track_id": current_track_id,
        "next_track_id": next_track_id,
        "transition_file": transition_file,
        "transition_start_time": transition_start_time,
        "track_b_entry_time": track_b_entry_time,
        "strategy": strategy
    }

    state["transitions"].append(transition_record)
    save_setlist_state(setlist_path, state)

    return {
        "status": "success",
        "setlist_path": os.path.abspath(setlist_path),
        "transition": transition_record,
        "transition_count": len(state["transitions"])
    }


# Returns the full setlist timeline
def get_setlist_state(setlist_path):
    state = load_setlist_state(setlist_path)

    return {
        "status": "success",
        "setlist_path": os.path.abspath(setlist_path),
        "transition_count": len(state["transitions"]),
        "transitions": state["transitions"]
    }


# Clears the setlist timeline
def clear_setlist_state(setlist_path):
    return create_setlist_state(setlist_path)