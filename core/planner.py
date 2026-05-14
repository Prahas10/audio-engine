import os
import librosa
from fastapi import HTTPException
from core.analysis import *
from core.strategy_router import *
from models.schemas import *


# Scores a transition candidate based on the energy change before and after the point (ideal for finding 'drops' or 'outros').
def local_energy_score(candidate_time, energy_times, energy_values, window_seconds=20):
    before_start = candidate_time - window_seconds
    before_end = candidate_time

    after_start = candidate_time
    after_end = candidate_time + window_seconds

    before_mask = (energy_times >= before_start) & (energy_times < before_end)
    after_mask = (energy_times >= after_start) & (energy_times < after_end)

    if not np.any(before_mask) or not np.any(after_mask):
        return 0.5

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

# Aligns the beat grid of Track B with the transition point of Track A to ensure the tracks are 'in sync'.
def find_best_sync_point(track_a_beats, track_b_beats, transition_start_sample, mix_samples, offset_samples=1200):
    track_a_beats = np.asarray(track_a_beats)
    track_b_beats = np.asarray(track_b_beats)

    a_window_beats = track_a_beats[
        (track_a_beats >= transition_start_sample)
        & (track_a_beats <= transition_start_sample + mix_samples)
    ]

    if len(a_window_beats) == 0 or len(track_b_beats) == 0:
        return 0, 0.0

    best_b_start = 0
    best_score = -1

    for b_idx in range(len(track_b_beats)):
        b_anchor = track_b_beats[b_idx]
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

        score = matches / max(len(a_window_beats), 1)

        if score > best_score:
            best_score = score
            best_b_start = b_anchor

    return int(best_b_start), float(best_score)


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

    candidates = phrase_boundary_candidates(
        beats=beats_a,
        sr=TARGET_SR,
        song_duration=duration_a,
        phrase_beats=32
    )

    if not candidates:
        beat_times = librosa.samples_to_time(beats_a, sr=TARGET_SR)
        candidates = [
            float(t)
            for t in beat_times
            if duration_a * 0.55 <= t <= duration_a * 0.92
        ]

    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="Could not find candidate transition points."
        )

    best_score = -1
    best_time = candidates[0]

    best_start_sample_a = int(best_time * TARGET_SR)
    mix_samples = int(preferred_mix_duration * TARGET_SR)

    start_sample_b, sync_accuracy = find_best_sync_point(
        track_a_beats=beats_a,
        track_b_beats=beats_b,
        transition_start_sample=best_start_sample_a,
        mix_samples=mix_samples,
        offset_samples=int(0.027 * TARGET_SR)
    )

    track_b_entry_time = start_sample_b / TARGET_SR
    scored_candidates = []

    for candidate_time in candidates:
        energy_score = local_energy_score(
            candidate_time=candidate_time,
            energy_times=energy_times_a,
            energy_values=energy_values_a,
            window_seconds=20
        )

        r_score = runway_score(
            candidate_time=candidate_time,
            song_duration=duration_a,
            mix_duration=preferred_mix_duration
        )

        # Phrase candidates are already phrase-aligned, so phrase score is strong.
        phrase_score = 1.0

        total_score = (
            phrase_score * 0.35
            + energy_score * 0.40
            + r_score * 0.25
        )

        scored_candidates.append({
            "time": round(float(candidate_time), 3),
            "score": round(float(total_score), 3),
            "energy_score": round(float(energy_score), 3),
            "runway_score": round(float(r_score), 3)
        })

        if total_score > best_score:
            best_score = total_score
            best_time = candidate_time
            best_energy_score = energy_score
            best_runway_score = r_score

    strategy, strategy_scores = choose_strategy_with_scores(
        harmonic_ok=harmonic_ok,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        best_time=best_time,
        song_duration_a=duration_a,
        mix_duration=preferred_mix_duration,
        energy_score=best_energy_score,
        runway_score_value=best_runway_score,
        sync_accuracy=sync_accuracy,
        key_confidence_a=key_conf_a,
        key_confidence_b=key_conf_b
    )

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
        "mix_duration": preferred_mix_duration,
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
        "top_candidate_points": sorted(
            scored_candidates,
            key=lambda x: x["score"],
            reverse=True
        )[:5],
        "render_payload": {
            "track_a_path": track_a_path,
            "track_b_path": track_b_path,
            "transition_start_time": round(float(best_time), 3),
            "track_b_entry_time": round(float(track_b_entry_time), 3),
            "mix_duration": preferred_mix_duration,
            "output_dir": "outputs",
            "transition_strategy": strategy,
            "fx_parameters": {
                "apply_reverb_tail": strategy == "reverb_wash",
                "loop_track_a": strategy == "auto_loop",
                "hpf_sweep_end_freq": 5000 if strategy == "hpf_sweep" else None
            }
        }
    }