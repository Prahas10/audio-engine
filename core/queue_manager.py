from fastapi import HTTPException

from core.library import load_library_metadata
from core.analysis import camelot_compatible


# Scores how suitable one candidate track is after the current track
def score_next_track(current_track, candidate_track, preferred_mix_duration=None):
    score = 0.0
    reasons = []

    bpm_a = current_track.get("bpm")
    bpm_b = candidate_track.get("bpm")

    camelot_a = current_track.get("camelot")
    camelot_b = candidate_track.get("camelot")

    duration_b = candidate_track.get("duration", 0)
    phrase_points = candidate_track.get("phrase_points", [])

    harmonic_ok = camelot_compatible(camelot_a, camelot_b)
    bpm_delta = abs(bpm_a - bpm_b) if bpm_a is not None and bpm_b is not None else 999
    duration_for_scoring = preferred_mix_duration or 30
    if harmonic_ok:
        score += 35
        reasons.append("Camelot-compatible key")
    else:
        score -= 15
        reasons.append("Not harmonically ideal")

    if bpm_delta <= 2:
        score += 30
        reasons.append("Very close BPM")
    elif bpm_delta <= 5:
        score += 22
        reasons.append("Close BPM")
    elif bpm_delta <= 8:
        score += 10
        reasons.append("Moderate BPM difference")
    else:
        score -= 20
        reasons.append("Large BPM difference")

    if duration_b >= duration_for_scoring + 60:
        score += 15
        reasons.append("Enough track length for clean entry")
    elif duration_b >= duration_for_scoring:
        score += 8
        reasons.append("Enough duration for transition")
    else:
        score -= 25
        reasons.append("Track may be too short")

    if len(phrase_points) >= 3:
        score += 15
        reasons.append("Good phrase boundary availability")
    elif len(phrase_points) > 0:
        score += 7
        reasons.append("Some phrase boundaries detected")
    else:
        score -= 10
        reasons.append("Weak phrase boundary detection")

    key_conf = candidate_track.get("key_confidence", 0)

    if key_conf >= 0.55:
        score += 8
        reasons.append("Reliable key estimate")
    elif key_conf < 0.35:
        score -= 8
        reasons.append("Low key confidence")

    return {
        "track_id": candidate_track["track_id"],
        "filename": candidate_track["filename"],
        "path": candidate_track["path"],
        "score": round(float(score), 3),
        "bpm": bpm_b,
        "camelot": camelot_b,
        "key": candidate_track.get("key"),
        "reasons": reasons,
        "bpm_delta": round(float(bpm_delta), 3) if bpm_delta != 999 else None,
        "harmonic_compatible": harmonic_ok
    }


# Ranks all library tracks and recommends the best next tracks
def recommend_next_tracks(current_track_id, library_path, preferred_mix_duration=30, max_results=5):
    library = load_library_metadata(library_path)

    if current_track_id not in library:
        raise HTTPException(
            status_code=404,
            detail=f"Current track_id not found in library: {current_track_id}"
        )

    current_track = library[current_track_id]

    candidates = []

    for track_id, candidate_track in library.items():
        if track_id == current_track_id:
            continue

        try:
            candidate_score = score_next_track(
                current_track=current_track,
                candidate_track=candidate_track,
                preferred_mix_duration=preferred_mix_duration
            )
            candidates.append(candidate_score)

        except Exception as e:
            candidates.append({
                "track_id": track_id,
                "filename": candidate_track.get("filename"),
                "score": -999,
                "error": str(e)
            })

    ranked = sorted(
        candidates,
        key=lambda x: x.get("score", -999),
        reverse=True
    )

    return {
        "status": "success",
        "current_track": {
            "track_id": current_track["track_id"],
            "filename": current_track["filename"],
            "path": current_track["path"],
            "bpm": current_track.get("bpm"),
            "key": current_track.get("key"),
            "camelot": current_track.get("camelot")
        },
        "recommendations": ranked[:max_results]
    }