import os
import librosa
from fastapi import HTTPException
from core.analysis import *
from core.strategy_router import *
from models.schemas import *
from core.strategy_router import get_strategy_mix_duration


# Scores a transition candidate based on the energy change before and after the point (ideal for finding 'drops' or 'outros').
def local_energy_score(candidate_time, energy_times, energy_values, window_seconds=20):
    before_start = candidate_time - window_seconds
    before_end = candidate_time

    after_start = candidate_time
    after_end = candidate_time + window_seconds

    before_mask = (energy_times >= before_start) & (energy_times < before_end)
    after_mask = (energy_times >= after_start) & (energy_times < after_end)

    if not np.any(before_mask) or not np.any(after_mask):
        return 0.3

    before_energy = float(np.mean(energy_values[before_mask]))
    after_energy = float(np.mean(energy_values[after_mask]))

    # Good transition areas often have stable or slightly falling energy.
    drop = before_energy - after_energy

    score = 0.5 + drop
    return float(np.clip(score, 0.0, 1.0))

# Scores a candidate point based on whether there is enough remaining audio to complete the requested mix duration.
def runway_score(candidate_time, song_duration, mix_duration):
    remaining = song_duration - candidate_time

    if remaining >= mix_duration:
        return 1.0

    if remaining >= mix_duration * 0.5:
        return 0.6

    return 0.25

def fine_sync_near_phrase(
    track_a_beats,
    track_b_beats,
    transition_start_sample,
    chosen_b_phrase_sample,
    mix_samples,
    sr,
    search_window_beats=4,
    offset_samples=1200,
):
    """
    Fine-syncs Track B around the selected phrase start.
    It does NOT search the whole intro, only a small beat window around
    the chosen phrase point.
    """
    track_a_beats = np.asarray(track_a_beats)
    track_b_beats = np.asarray(track_b_beats)

    beat_duration_samples = int((60.0 / 124.0) * sr)
    search_radius = search_window_beats * beat_duration_samples

    candidate_b_beats = track_b_beats[
        (track_b_beats >= chosen_b_phrase_sample - search_radius)
        & (track_b_beats <= chosen_b_phrase_sample + search_radius)
    ]

    if len(candidate_b_beats) == 0:
        return chosen_b_phrase_sample, 0.5

    a_window_beats = track_a_beats[
        (track_a_beats >= transition_start_sample)
        & (track_a_beats <= transition_start_sample + mix_samples)
    ]

    best_b_start = int(chosen_b_phrase_sample)
    best_score = -1.0

    for b_anchor in candidate_b_beats:
        shifted_b_beats = track_b_beats - b_anchor + transition_start_sample

        shifted_b_window = shifted_b_beats[
            (shifted_b_beats >= transition_start_sample)
            & (shifted_b_beats <= transition_start_sample + mix_samples)
        ]

        if len(shifted_b_window) == 0:
            continue

        matches = 0
        for beat_a in a_window_beats:
            if np.any(np.abs(shifted_b_window - beat_a) <= offset_samples):
                matches += 1

        denominator = max(min(len(a_window_beats), len(shifted_b_window)), 1)
        score = matches / denominator

        if score > best_score:
            best_score = score
            best_b_start = int(b_anchor)

    return best_b_start, float(best_score)

# Aligns the beat grid of Track B with the transition point of Track A to ensure the tracks are 'in sync'.# Finds the best Track B entry beat inside the allowed intro window
def find_best_sync_point(
    track_a_beats,
    track_b_beats,
    transition_start_sample,
    mix_samples,
    offset_samples=1200,
    track_b_total_samples=None,
    min_b_entry_percent=0.0,
    max_b_entry_percent=0.35
):
    track_a_beats = np.asarray(track_a_beats)
    track_b_beats = np.asarray(track_b_beats)
 
    a_window_beats = track_a_beats[
        (track_a_beats >= transition_start_sample)
        & (track_a_beats <= transition_start_sample + mix_samples)
    ]
 
    if len(a_window_beats) == 0 or len(track_b_beats) == 0:
        return 0, 0.0
 
    # FIX 2: derive a sensible window cap when total_samples not provided
    if track_b_total_samples is None:
        track_b_total_samples = int(track_b_beats[-1] * 1.05)  # estimate from last beat
 
    min_b_sample = int(track_b_total_samples * min_b_entry_percent)
    max_b_sample = int(track_b_total_samples * max_b_entry_percent)
 
    candidate_b_beats = track_b_beats[
        (track_b_beats >= min_b_sample)
        & (track_b_beats <= max_b_sample)
    ]
 
    if len(candidate_b_beats) == 0:
        candidate_b_beats = track_b_beats
 
    best_b_start = int(candidate_b_beats[0])
    best_score   = -1.0
 
    for b_anchor in candidate_b_beats:
        shifted_b_beats = track_b_beats - b_anchor + transition_start_sample
 
        shifted_b_window = shifted_b_beats[
            (shifted_b_beats >= transition_start_sample)
            & (shifted_b_beats <= transition_start_sample + mix_samples)
        ]
 
        if len(shifted_b_window) == 0:
            continue
 
        matches = 0
        for beat_a in a_window_beats:
            if np.any(np.abs(shifted_b_window - beat_a) <= offset_samples):
                matches += 1
 
        # FIX 3: normalise against the smaller of the two windows
        denominator   = max(min(len(a_window_beats), len(shifted_b_window)), 1)
        sync_accuracy = matches / denominator
 
        if sync_accuracy > best_score:
            best_score   = sync_accuracy
            best_b_start = int(b_anchor)
 
    return best_b_start, float(best_score)

    
def track_b_intro_phrase_candidates(beats_b, sr, duration_b, phrase_beats=32):
    """
    Finds phrase starts in the intro/early section of Track B.
    This prevents Track B from entering on a random beat.
    """
    phrase_times = get_phrase_boundaries(beats_b, sr, phrase_beats)

    if not phrase_times:
        return []

    candidates = [
        t for t in phrase_times
        if 0 <= t <= duration_b * 0.35
    ]

    return candidates or phrase_times[:4]

def score_phrase_pair(
    candidate_a_time,
    candidate_b_time,
    energy_times_a,
    energy_values_a,
    energy_times_b,
    energy_values_b,
    duration_a,
    mix_duration,
    harmonic_ok,
):
    """
    Scores A/B phrase combinations instead of only scoring Track A.
    """
    a_energy_score = local_energy_score(
        candidate_time=candidate_a_time,
        energy_times=energy_times_a,
        energy_values=energy_values_a,
        window_seconds=20,
    )

    b_energy_score = local_energy_score(
        candidate_time=candidate_b_time,
        energy_times=energy_times_b,
        energy_values=energy_values_b,
        window_seconds=20,
    )

    r_score = runway_score(
        candidate_time=candidate_a_time,
        song_duration=duration_a,
        mix_duration=mix_duration,
    )

    harmonic_score = 1.0 if harmonic_ok else 0.45

    total_score = (
        0.30 * a_energy_score +
        0.25 * b_energy_score +
        0.20 * r_score +
        0.25 * harmonic_score
    )

    return float(np.clip(total_score, 0.0, 1.0)), {
        "a_energy_score": round(float(a_energy_score), 3),
        "b_energy_score": round(float(b_energy_score), 3),
        "runway_score": round(float(r_score), 3),
        "harmonic_score": round(float(harmonic_score), 3),
    }

# The 'Brain' of the engine: analyzes both tracks to find the mathematically best point and method for a transition.
def plan_transition_logic(track_a_path, track_b_path, preferred_mix_duration):
    if not os.path.exists(track_a_path):
        raise HTTPException(status_code=400, detail=f"Track A not found: {track_a_path}")

    if not os.path.exists(track_b_path):
        raise HTTPException(status_code=400, detail=f"Track B not found: {track_b_path}")

    print("\n--- Starting Brain V1 Transition Planner ---")

    y_a, _ = librosa.load(track_a_path, sr=TARGET_SR, mono=True)
    y_b, _ = librosa.load(track_b_path, sr=TARGET_SR, mono=True)

    duration_a = librosa.get_duration(y=y_a, sr=TARGET_SR)
    duration_b = librosa.get_duration(y=y_b, sr=TARGET_SR)

    bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)
    bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)

    key_a, mode_a, key_conf_a = estimate_key(y_a, TARGET_SR)
    key_b, mode_b, key_conf_b = estimate_key(y_b, TARGET_SR)

    camelot_a = CAMELOT_MAP.get((key_a, mode_a))
    camelot_b = CAMELOT_MAP.get((key_b, mode_b))

    harmonic_ok = camelot_compatible(camelot_a, camelot_b)

    energy_times_a, energy_values_a = get_energy_curve(y_a, TARGET_SR)
    energy_times_b, energy_values_b = get_energy_curve(y_b, TARGET_SR)
    
    planning_mix_duration = preferred_mix_duration or round((60.0 / bpm_a) * 32 * 2, 1)
    
    candidates = phrase_boundary_candidates(
        beats=beats_a,
        sr=TARGET_SR,
        song_duration=duration_a,
        phrase_beats=32
    )
# Keep phrase candidates in the DJ-friendly outro zone
    candidates = [
        p for p in candidates
        if duration_a * 0.55 <= p <= duration_a * 0.88
    ]
    
    if not candidates:
        beat_times = librosa.samples_to_time(beats_a, sr=TARGET_SR)
        candidates = [float(t) for t in beat_times if duration_a * 0.55 <= t <= duration_a * 0.92]
 
    if not candidates:
        raise HTTPException(status_code=400, detail="Could not find candidate transition points.")
    
    b_phrase_candidates = track_b_intro_phrase_candidates(
        beats_b=beats_b,
        sr=TARGET_SR,
        duration_b=duration_b,
        phrase_beats=32
    )

    if not b_phrase_candidates:
        raise HTTPException(
            status_code=400,
            detail="Could not find Track B phrase candidates."
        )

    # --- Score Track A / Track B phrase pairs ---
    scored_candidates = []
    best_score = -1.0
    best_time = candidates[0]
    best_track_b_entry_time = b_phrase_candidates[0]
    best_energy_score = 0.5
    best_runway_score = 1.0

    for candidate_a_time in candidates:
        for candidate_b_time in b_phrase_candidates:
            total_score, score_parts = score_phrase_pair(
                candidate_a_time=candidate_a_time,
                candidate_b_time=candidate_b_time,
                energy_times_a=energy_times_a,
                energy_values_a=energy_values_a,
                energy_times_b=energy_times_b,
                energy_values_b=energy_values_b,
                duration_a=duration_a,
                mix_duration=planning_mix_duration,
                harmonic_ok=harmonic_ok,
            )

            scored_candidates.append({
                "track_a_time": round(float(candidate_a_time), 3),
                "track_b_time": round(float(candidate_b_time), 3),
                "score": round(float(total_score), 3),
                **score_parts,
            })

            if total_score > best_score:
                best_score = total_score
                best_time = candidate_a_time
                best_track_b_entry_time = candidate_b_time
                best_energy_score = score_parts["a_energy_score"]
                best_runway_score = score_parts["runway_score"]

    # --- Choose strategy based on best candidate ---
    strategy, strategy_scores = choose_strategy_with_scores(
        harmonic_ok=harmonic_ok,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        best_time=best_time,
        song_duration_a=duration_a,
        mix_duration=planning_mix_duration,
        energy_score=best_energy_score,
        runway_score_value=best_runway_score,
        sync_accuracy=0.5,  # placeholder; updated after sync search below
        key_confidence_a=key_conf_a,
        key_confidence_b=key_conf_b
    )
 
    # --- Determine final mix_duration ---
    if preferred_mix_duration is None:
        mix_duration = get_strategy_mix_duration(strategy, bpm_a)
    else:
        mix_duration = preferred_mix_duration
 
    mix_samples = int(mix_duration * TARGET_SR)
    best_start_sample = int(best_time * TARGET_SR)
 
    # FIX 1: sync search uses the actual chosen start + final mix_samples
    chosen_b_phrase_sample = int(best_track_b_entry_time * TARGET_SR)

    start_sample_b, sync_accuracy = fine_sync_near_phrase(
        track_a_beats=beats_a,
        track_b_beats=beats_b,
        transition_start_sample=best_start_sample,
        chosen_b_phrase_sample=chosen_b_phrase_sample,
        mix_samples=mix_samples,
        sr=TARGET_SR,
        search_window_beats=4,
        offset_samples=1200,
    )

    track_b_entry_time = start_sample_b / TARGET_SR
    reason = (
        f"Selected {strategy} because Track A is {camelot_a}, "
        f"Track B is {camelot_b}, harmonic compatibility is {harmonic_ok}, "
        f"BPM delta is {abs(bpm_a - bpm_b):.2f}, and the best phrase-energy point is {best_time:.2f}s."
    )
    print(reason)
    
    return {
        "status": "success",
        "recommended_transition_start_time": round(float(best_time), 3),
        "recommended_track_b_entry_time": round(float(track_b_entry_time), 3),
        "recommended_track_b_entry_sample": int(start_sample_b),
        "sync_accuracy": round(float(sync_accuracy), 3),
        "recommended_strategy": strategy,
        "strategy_scores": strategy_scores,
        "mix_duration": mix_duration,
        "reason": reason,
        "track_a": {
            "duration": round(float(duration_a), 2),
            "bpm": round(float(bpm_a), 2),
            "key": f"{key_a} {mode_a}",
            "camelot": camelot_a,
            "key_confidence": round(float(key_conf_a), 3)
        },
        "track_b": {
            "duration": round(float(duration_b), 2),
            "bpm": round(float(bpm_b), 2),
            "key": f"{key_b} {mode_b}",
            "camelot": camelot_b,
            "key_confidence": round(float(key_conf_b), 3)
        },
        "harmonic_compatible": harmonic_ok,
        "top_phrase_pairs": sorted(
            scored_candidates,
            key=lambda x: x["score"],
            reverse=True
        )[:5],
        "render_payload": {
            "track_a_path": track_a_path,
            "track_b_path": track_b_path,
            "transition_start_time": round(float(best_time), 3),
            "track_b_entry_time": round(float(track_b_entry_time), 3),
            "mix_duration": mix_duration,
            "output_dir": "outputs",
            "transition_strategy": strategy,
            "fx_parameters": {
                "apply_reverb_tail": strategy == "reverb_wash",
                "loop_track_a": strategy == "auto_loop",
                # FIX 2: use corrected default end freqs matching transitions.py
                "hpf_sweep_end_freq":   3500.0 if strategy == "hpf_sweep" else None,
                "lpf_sweep_end_freq":   400.0  if strategy == "lpf_sweep"  else None,
            }
        }
    }
