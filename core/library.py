import os
import json
import uuid
from pathlib import Path

import numpy as np
import librosa
from fastapi import HTTPException
from models.schemas import CAMELOT_MAP
from core.analysis import (
    TARGET_SR,
    safe_bpm,
    estimate_key,
    get_energy_curve,
    phrase_boundary_candidates
)

# Creates a stable unique ID for a track path
def make_track_id(track_path):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(track_path)))


# Loads existing JSON library metadata
def load_library_metadata(library_path):
    if not os.path.exists(library_path):
        return {}

    with open(library_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Saves one scanned track into the JSON metadata library
def save_library_metadata(library_path, track_metadata):
    os.makedirs(os.path.dirname(library_path), exist_ok=True)

    library = load_library_metadata(library_path)

    library[track_metadata["track_id"]] = track_metadata

    with open(library_path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)

    return library


# Scans one audio track and returns reusable DJ metadata
def scan_track_metadata(track_path):
    if not os.path.exists(track_path):
        raise HTTPException(status_code=400, detail=f"Track not found: {track_path}")

    y, _ = librosa.load(track_path, sr=TARGET_SR, mono=True)

    duration = librosa.get_duration(y=y, sr=TARGET_SR)
    bpm, beats = safe_bpm(y, TARGET_SR)

    key, mode, key_confidence = estimate_key(y, TARGET_SR)
    camelot = CAMELOT_MAP.get((key, mode))

    energy_times, energy_values = get_energy_curve(y, TARGET_SR)

    phrase_points = phrase_boundary_candidates(
        beats=beats,
        sr=TARGET_SR,
        song_duration=duration,
        phrase_beats=32
    )

    return {
        "track_id": make_track_id(track_path),
        "path": os.path.abspath(track_path),
        "filename": os.path.basename(track_path),
        "extension": Path(track_path).suffix.lower(),
        "duration": round(float(duration), 3),
        "bpm": round(float(bpm), 3),
        "key": f"{key} {mode}",
        "key_name": key,
        "mode": mode,
        "camelot": camelot,
        "key_confidence": round(float(key_confidence), 3),
        "phrase_points": [round(float(p), 3) for p in phrase_points],
        "energy_summary": {
            "avg": round(float(np.mean(energy_values)), 3),
            "max": round(float(np.max(energy_values)), 3),
            "min": round(float(np.min(energy_values)), 3)
        }
    }


# Scans one track and saves it to the metadata library
def scan_and_save_track(track_path, library_path):
    metadata = scan_track_metadata(track_path)
    save_library_metadata(library_path, metadata)
    return metadata


# Scans folder, skipping already-scanned tracks unless force_rescan is enabled
def scan_folder_metadata(folder_path, library_path, force_rescan=False, clear_existing=False):
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder_path}")

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

    if clear_existing:
        library = {}
        os.makedirs(os.path.dirname(library_path), exist_ok=True)
        with open(library_path, "w", encoding="utf-8") as f:
            json.dump(library, f, indent=2)
    else:
        library = load_library_metadata(library_path)

    scanned_tracks = []
    skipped_tracks = []
    failed_tracks = []

    for file_path in Path(folder_path).rglob("*"):
        if file_path.suffix.lower() not in audio_extensions:
            continue

        track_path = str(file_path)
        track_id = make_track_id(track_path)

        if track_id in library and not force_rescan:
            skipped_tracks.append({
                "track_id": track_id,
                "path": os.path.abspath(track_path),
                "reason": "already_scanned"
            })
            continue

        try:
            metadata = scan_and_save_track(
                track_path=track_path,
                library_path=library_path
            )
            scanned_tracks.append(metadata)

        except Exception as e:
            failed_tracks.append({
                "path": track_path,
                "error": str(e)
            })

    return {
        "status": "success",
        "folder_path": os.path.abspath(folder_path),
        "library_path": os.path.abspath(library_path),
        "scanned_count": len(scanned_tracks),
        "skipped_count": len(skipped_tracks),
        "failed_count": len(failed_tracks),
        "scanned_tracks": scanned_tracks,
        "skipped_tracks": skipped_tracks,
        "failed_tracks": failed_tracks
    }


# Returns all tracks currently stored in the metadata library
def list_library_tracks(library_path):
    library = load_library_metadata(library_path)

    return {
        "status": "success",
        "library_path": os.path.abspath(library_path),
        "track_count": len(library),
        "tracks": list(library.values())
    }