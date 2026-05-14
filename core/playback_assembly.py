import os

import librosa
import soundfile as sf
import numpy as np
from fastapi import HTTPException

from core.analysis import TARGET_SR
from core.library import load_library_metadata
from core.setlist_state import load_setlist_state


# Loads an audio file as mono float audio at the engine sample rate
def load_audio_mono(path):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {path}")

    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return y


# Normalizes only if audio is clipping
def clip_protect(y, ceiling=0.98):
    peak = np.max(np.abs(y))

    if peak > ceiling:
        y = y / peak * ceiling

    return y


# Builds a continuous playback mix from the saved setlist transitions
def assemble_playback_from_setlist(setlist_path, library_path, output_path):
    library = load_library_metadata(library_path)
    setlist = load_setlist_state(setlist_path)

    transitions = setlist.get("transitions", [])

    if not transitions:
        raise HTTPException(status_code=400, detail="Setlist has no transitions to assemble.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_audio_parts = []

    first_transition = transitions[0]
    first_track_id = first_transition["current_track_id"]

    if first_track_id not in library:
        raise HTTPException(status_code=404, detail=f"First track not found in library: {first_track_id}")

    current_track_audio = load_audio_mono(library[first_track_id]["path"])

    first_transition_start_sample = int(first_transition["transition_start_time"] * TARGET_SR)

    final_audio_parts.append(
        current_track_audio[:first_transition_start_sample]
    )

    for idx, transition in enumerate(transitions):
        current_track_id = transition["current_track_id"]
        next_track_id = transition["next_track_id"]

        transition_file = transition["transition_file"]
        transition_start_time = transition["transition_start_time"]
        track_b_entry_time = transition["track_b_entry_time"]

        if current_track_id not in library:
            raise HTTPException(status_code=404, detail=f"Track not found in library: {current_track_id}")

        if next_track_id not in library:
            raise HTTPException(status_code=404, detail=f"Next track not found in library: {next_track_id}")

        transition_audio = load_audio_mono(transition_file)
        final_audio_parts.append(transition_audio)

        next_track_audio = load_audio_mono(library[next_track_id]["path"])

        next_start_sample = int((track_b_entry_time + len(transition_audio) / TARGET_SR) * TARGET_SR)

        if idx + 1 < len(transitions):
            next_transition = transitions[idx + 1]
            next_transition_start_sample = int(next_transition["transition_start_time"] * TARGET_SR)

            if next_transition["current_track_id"] != next_track_id:
                raise HTTPException(
                    status_code=400,
                    detail="Setlist chain is broken: next transition current_track_id does not match previous next_track_id."
                )

            final_audio_parts.append(
                next_track_audio[next_start_sample:next_transition_start_sample]
            )

        else:
            final_audio_parts.append(
                next_track_audio[next_start_sample:]
            )

    final_audio = np.concatenate(final_audio_parts)
    final_audio = clip_protect(final_audio)

    sf.write(output_path, final_audio, TARGET_SR)

    return {
        "status": "success",
        "output_path": os.path.abspath(output_path),
        "duration_seconds": round(float(len(final_audio) / TARGET_SR), 3),
        "transition_count": len(transitions)
    }