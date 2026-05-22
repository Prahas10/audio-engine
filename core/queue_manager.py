"""
core/queue_manager.py

Changes in this version:
  - score_next_track: adds energy_arc_score based on set position
    Early set (0-33%): prefer tracks with higher avg energy than current
    Peak (33-67%): prefer tracks near maximum energy
    Outro (67-100%): prefer tracks with lower energy for landing
  - recommend_next_tracks: passes set_position parameter
  - All other logic unchanged
"""

from fastapi import HTTPException
from core.library import load_library_metadata
from core.analysis import camelot_compatible


def _energy_arc_score(current_energy_avg: float, candidate_energy_avg: float,
                      set_position: float) -> float:
    """
    Scores how well the candidate's energy level fits the current arc position.

    set_position: 0.0 = first track, 1.0 = last track in planned set.
    If unknown, pass 0.5 (neutral).

    Energy arc shape:
      0.0 – 0.33 (build): candidate should be >= current energy
      0.33 – 0.67 (peak): candidate should be near or above current energy
      0.67 – 1.0 (outro): candidate should be <= current energy
    """
    delta = candidate_energy_avg - current_energy_avg  # positive = louder incoming

    if set_position <= 0.33:
        # Building — reward energy increase, penalise drops
        if delta >= 0:
            return min(1.0, 0.7 + delta * 1.5)
        else:
            return max(0.0, 0.7 + delta * 2.0)

    elif set_position <= 0.67:
        # Peak — both flat and slight increases are good
        if delta >= -0.05:
            return 1.0
        else:
            return max(0.0, 1.0 + delta * 3.0)

    else:
        # Outro — reward energy decrease, penalise increases
        if delta <= 0:
            return min(1.0, 0.7 + abs(delta) * 1.5)
        else:
            return max(0.0, 0.7 - delta * 2.0)


def score_next_track(current_track, candidate_track,
                     preferred_mix_duration=None, set_position: float = 0.5):
    score   = 0.0
    reasons = []

    bpm_a   = current_track.get("bpm")
    bpm_b   = candidate_track.get("bpm")
    camelot_a = current_track.get("camelot")
    camelot_b = candidate_track.get("camelot")
    duration_b    = candidate_track.get("duration", 0)
    phrase_points = candidate_track.get("phrase_points", [])
    duration_for_scoring = preferred_mix_duration or 30

    harmonic_ok = camelot_compatible(camelot_a, camelot_b)
    bpm_delta   = abs(bpm_a - bpm_b) if bpm_a is not None and bpm_b is not None else 999

    # Harmonic
    if harmonic_ok:
        score += 35
        reasons.append("Camelot-compatible key")
    else:
        score -= 15
        reasons.append("Not harmonically ideal")

    # BPM
    if bpm_delta <= 2:
        score += 30; reasons.append("Very close BPM")
    elif bpm_delta <= 5:
        score += 22; reasons.append("Close BPM")
    elif bpm_delta <= 8:
        score += 10; reasons.append("Moderate BPM difference")
    else:
        score -= 20; reasons.append("Large BPM difference")

    # Duration
    if duration_b >= duration_for_scoring + 60:
        score += 15; reasons.append("Enough track length for clean entry")
    elif duration_b >= duration_for_scoring:
        score += 8;  reasons.append("Enough duration for transition")
    else:
        score -= 25; reasons.append("Track may be too short")

    # Phrase boundaries
    if len(phrase_points) >= 3:
        score += 15; reasons.append("Good phrase boundary availability")
    elif len(phrase_points) > 0:
        score += 7;  reasons.append("Some phrase boundaries detected")
    else:
        score -= 10; reasons.append("Weak phrase boundary detection")

    # Key confidence
    key_conf = candidate_track.get("key_confidence", 0)
    if key_conf >= 0.55:
        score += 8; reasons.append("Reliable key estimate")
    elif key_conf < 0.35:
        score -= 8; reasons.append("Low key confidence")

    # Energy arc — up to ±20 points
    current_energy_avg   = current_track.get("energy_summary", {}).get("avg", 0.5)
    candidate_energy_avg = candidate_track.get("energy_summary", {}).get("avg", 0.5)
    arc_score = _energy_arc_score(current_energy_avg, candidate_energy_avg, set_position)
    arc_bonus = (arc_score - 0.5) * 40  # maps 0-1 → -20 to +20
    score += arc_bonus
    arc_label = (
        "Good energy build" if arc_bonus > 5 else
        "Good energy peak"  if arc_bonus > 0 else
        "Good energy drop"  if arc_bonus > -5 else
        "Energy arc mismatch"
    )
    reasons.append(f"{arc_label} (pos={set_position:.2f}, Δenergy={candidate_energy_avg - current_energy_avg:+.2f})")

    return {
        "track_id":            candidate_track["track_id"],
        "filename":            candidate_track["filename"],
        "path":                candidate_track["path"],
        "score":               round(float(score), 3),
        "bpm":                 bpm_b,
        "camelot":             camelot_b,
        "key":                 candidate_track.get("key"),
        "reasons":             reasons,
        "bpm_delta":           round(float(bpm_delta), 3) if bpm_delta != 999 else None,
        "harmonic_compatible": harmonic_ok,
        "energy_arc_score":    round(float(arc_score), 3),
        "set_position":        round(float(set_position), 3),
    }


def recommend_next_tracks(current_track_id, library_path,
                          preferred_mix_duration=30, max_results=5,
                          set_position: float = 0.5):
    library = load_library_metadata(library_path)

    if current_track_id not in library:
        raise HTTPException(
            status_code=404,
            detail=f"Current track_id not found in library: {current_track_id}"
        )

    current_track = library[current_track_id]
    candidates    = []

    for track_id, candidate_track in library.items():
        if track_id == current_track_id:
            continue
        try:
            candidate_score = score_next_track(
                current_track=current_track,
                candidate_track=candidate_track,
                preferred_mix_duration=preferred_mix_duration,
                set_position=set_position,
            )
            candidates.append(candidate_score)
        except Exception as e:
            candidates.append({
                "track_id": track_id,
                "filename": candidate_track.get("filename"),
                "score":    -999,
                "error":    str(e),
            })

    ranked = sorted(candidates, key=lambda x: x.get("score", -999), reverse=True)

    return {
        "status": "success",
        "current_track": {
            "track_id": current_track["track_id"],
            "filename": current_track["filename"],
            "path":     current_track["path"],
            "bpm":      current_track.get("bpm"),
            "key":      current_track.get("key"),
            "camelot":  current_track.get("camelot"),
        },
        "recommendations": ranked[:max_results],
    }