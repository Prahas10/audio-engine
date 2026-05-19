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
    safe_bpm_from_path,
    estimate_key,
    get_energy_curve,
    phrase_boundary_candidates_from_downbeats,
    intro_phrase_candidates_from_downbeats,
    get_drop_candidates,
)


def make_track_id(track_path):
    stat = os.stat(track_path)
    stable_key = f"{os.path.abspath(track_path)}::{stat.st_size}::{stat.st_mtime}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))


def load_library_metadata(library_path):
    if not os.path.exists(library_path):
        return {}

    with open(library_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_library_metadata(library_path, track_metadata):
    os.makedirs(os.path.dirname(library_path), exist_ok=True)

    library = load_library_metadata(library_path)
    library[track_metadata["track_id"]] = track_metadata

    with open(library_path, "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)

    return library


def scan_track_metadata(track_path):
    if not os.path.exists(track_path):
        raise HTTPException(status_code=400, detail=f"Track not found: {track_path}")

    abs_path = os.path.abspath(track_path)

    y, _ = librosa.load(abs_path, sr=TARGET_SR, mono=True)

    duration = librosa.get_duration(y=y, sr=TARGET_SR)

    bpm, beats, downbeats = safe_bpm_from_path(abs_path, TARGET_SR)

    key, mode, key_confidence = estimate_key(y, TARGET_SR)
    camelot = CAMELOT_MAP.get((key, mode))

    energy_times, energy_values = get_energy_curve(y, TARGET_SR)

    phrase_points = phrase_boundary_candidates_from_downbeats(
        downbeats=downbeats,
        sr=TARGET_SR,
        song_duration=duration,
        phrase_bars=8,
        min_percent=0.55,
        max_percent=0.92,
    )

    intro_phrase_points = intro_phrase_candidates_from_downbeats(
        downbeats=downbeats,
        sr=TARGET_SR,
        song_duration=duration,
        phrase_bars=8,
        max_percent=0.35,
    )

    drop_points = get_drop_candidates(
        y=y,
        sr=TARGET_SR,
        candidate_times=intro_phrase_points,
        top_k=8,
    )

    stat = os.stat(abs_path)

    return {
        "track_id": make_track_id(abs_path),
        "path": abs_path,
        "filename": os.path.basename(abs_path),
        "extension": Path(abs_path).suffix.lower(),

        "file_size": int(stat.st_size),
        "modified_time": float(stat.st_mtime),

        "duration": round(float(duration), 3),
        "sample_rate": TARGET_SR,

        "bpm": round(float(bpm), 3),

        "key": f"{key} {mode}",
        "key_name": key,
        "mode": mode,
        "camelot": camelot,
        "key_confidence": round(float(key_confidence), 3),

        "beat_count": int(len(beats)),
        "downbeat_count": int(len(downbeats)),

        "beats": [round(float(t), 3) for t in librosa.samples_to_time(beats, sr=TARGET_SR)],
        "downbeats": [round(float(t), 3) for t in librosa.samples_to_time(downbeats, sr=TARGET_SR)],

        "phrase_points": [round(float(p), 3) for p in phrase_points],
        "intro_phrase_points": [round(float(p), 3) for p in intro_phrase_points],
        "drop_points": [round(float(p), 3) for p in drop_points],

        "energy_summary": {
            "avg": round(float(np.mean(energy_values)), 3),
            "max": round(float(np.max(energy_values)), 3),
            "min": round(float(np.min(energy_values)), 3),
            "std": round(float(np.std(energy_values)), 3),
        },

        "analysis_version": "madmom_v2",
    }


def scan_and_save_track(track_path, library_path):
    metadata = scan_track_metadata(track_path)
    save_library_metadata(library_path, metadata)
    return metadata


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

    existing_paths = {
        metadata.get("path"): metadata
        for metadata in library.values()
    }

    for file_path in Path(folder_path).rglob("*"):
        if file_path.suffix.lower() not in audio_extensions:
            continue

        track_path = os.path.abspath(str(file_path))
        track_id = make_track_id(track_path)

        if not force_rescan:
            if track_id in library:
                skipped_tracks.append({
                    "track_id": track_id,
                    "path": track_path,
                    "reason": "already_scanned_same_file_version",
                })
                continue

            old_metadata = existing_paths.get(track_path)
            if old_metadata:
                skipped_tracks.append({
                    "track_id": old_metadata.get("track_id"),
                    "path": track_path,
                    "reason": "already_scanned_same_path",
                })
                continue

        try:
            metadata = scan_and_save_track(
                track_path=track_path,
                library_path=library_path,
            )
            scanned_tracks.append(metadata)

        except Exception as e:
            failed_tracks.append({
                "path": track_path,
                "error": str(e),
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
        "failed_tracks": failed_tracks,
    }


def list_library_tracks(library_path):
    library = load_library_metadata(library_path)

    return {
        "status": "success",
        "library_path": os.path.abspath(library_path),
        "track_count": len(library),
        "tracks": list(library.values()),
    }


def get_track_metadata(track_path, library_path):
    library = load_library_metadata(library_path)
    abs_path = os.path.abspath(track_path)

    track_id = make_track_id(abs_path)

    if track_id in library:
        return library[track_id]

    for metadata in library.values():
        if metadata.get("path") == abs_path:
            return metadata

    return None