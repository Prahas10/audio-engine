import os
import json


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
        **state,
    }


def load_setlist_state(setlist_path):
    if not os.path.exists(setlist_path):
        return create_setlist_state(setlist_path)

    with open(setlist_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_setlist_state(setlist_path, state):
    os.makedirs(os.path.dirname(setlist_path), exist_ok=True)

    with open(setlist_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    return state


def add_transition_to_setlist(
    current_track_id,
    next_track_id,
    transition_file,
    transition_start_time,
    track_b_entry_time,
    strategy,
    setlist_path,
    mix_duration=None,
    track_b_suffix_file=None,
    track_b_suffix_start_time=None,
    track_b_suffix_duration=None,
):
    state = load_setlist_state(setlist_path)

    record = {
        "current_track_id": current_track_id,
        "next_track_id": next_track_id,

        "transition_file": transition_file,
        "track_b_suffix_file": track_b_suffix_file,

        "transition_start_time": transition_start_time,
        "track_b_entry_time": track_b_entry_time,
        "mix_duration": mix_duration,

        "track_b_suffix_start_time": track_b_suffix_start_time,
        "track_b_suffix_duration": track_b_suffix_duration,

        "strategy": strategy,
    }

    state["transitions"].append(record)
    save_setlist_state(setlist_path, state)

    return {
        "status": "success",
        "setlist_path": os.path.abspath(setlist_path),
        "transition": record,
        "transition_count": len(state["transitions"]),
    }


def get_setlist_state(setlist_path):
    state = load_setlist_state(setlist_path)

    return {
        "status": "success",
        "setlist_path": os.path.abspath(setlist_path),
        "transition_count": len(state["transitions"]),
        "transitions": state["transitions"],
    }


def clear_setlist_state(setlist_path):
    return create_setlist_state(setlist_path)