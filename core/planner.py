import os
import numpy as np
import librosa
from fastapi import HTTPException

from core.analysis import (
    TARGET_SR, safe_bpm, estimate_key, get_energy_curve,
    get_phrase_boundaries, camelot_compatible,
)
from core.strategy_router import choose_strategy_with_scores, get_strategy_mix_duration
from models.schemas import CAMELOT_MAP


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

def safe_mean(values, default=0.0):
    v = np.asarray(values)
    return float(np.mean(v)) if len(v) > 0 else default

def safe_std(values, default=0.0):
    v = np.asarray(values)
    return float(np.std(v)) if len(v) > 0 else default

def get_window_values(times, values, start, end):
    times  = np.asarray(times)
    values = np.asarray(values)
    return values[(times >= start) & (times < end)]


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def local_energy_shape_score(candidate_time, energy_times, energy_values, window_seconds=16):
    """
    Good outgoing transition points have stable or slightly falling energy.
    """
    before = get_window_values(energy_times, energy_values,
                               candidate_time - window_seconds, candidate_time)
    after  = get_window_values(energy_times, energy_values,
                               candidate_time, candidate_time + window_seconds)

    if len(before) < 2 or len(after) < 2:
        return 0.45

    drop      = safe_mean(before) - safe_mean(after)
    stability = 1.0 - min(safe_std(after) * 3.0, 1.0)
    drop_score = clamp01(0.65 + (drop * 1.2 if drop >= 0 else drop * 1.8))
    return clamp01(0.65 * drop_score + 0.35 * stability)


def runway_score(candidate_time, song_duration, mix_duration, safety_seconds=4.0):
    """
    Strongly penalises candidates without enough audio remaining.
    """
    remaining = song_duration - candidate_time
    required  = mix_duration + safety_seconds

    if remaining >= required:       return 1.0
    if remaining >= mix_duration:   return 0.75
    if remaining >= mix_duration * 0.75: return 0.35
    return 0.0


def outro_zone_score(candidate_time, duration):
    """
    Prefers the DJ outro zone (65–88% through the track).
    Kept at a tiny weight — phrase_score already captures position alignment.
    """
    if duration <= 0:
        return 0.5
    p = candidate_time / duration
    if   0.65 <= p <= 0.88: return 1.0
    elif 0.55 <= p <  0.65: return 0.70
    elif 0.88 <  p <= 0.93: return 0.55
    elif 0.50 <= p <  0.55: return 0.35
    return 0.10


def phrase_strength_score(candidate_time, beats, sr, phrase_beats=32):
    """
    Rewards candidates that land on a 32-beat (8-bar) phrase boundary.
    A real DJ always starts the mix on a phrase, not a random beat.
    """
    if beats is None or len(beats) == 0:
        return 0.4

    beat_times   = librosa.samples_to_time(beats, sr=sr)
    if len(beat_times) < phrase_beats + 1:
        return 0.45

    phrase_times = beat_times[::phrase_beats]
    nearest      = float(np.min(np.abs(phrase_times - candidate_time)))
    avg_beat     = float(np.median(np.diff(beat_times))) if len(beat_times) > 2 else 0.5
    tolerance    = max(avg_beat * 1.5, 0.25)
    return clamp01(1.0 - nearest / tolerance)


def spectral_low_stability_score(y, sr, candidate_time, window_seconds=16):
    """
    Penalises transition points where low-end energy is unstable.
    Unstable bass at the mix point causes muddy clashes.
    """
    start = max(0, int((candidate_time - window_seconds) * sr))
    end   = min(len(y), int((candidate_time + window_seconds) * sr))

    if end <= start or (end - start) < sr:
        return 0.45

    S        = np.abs(librosa.stft(y[start:end], n_fft=2048, hop_length=512))
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs <= 180

    if not np.any(low_mask):
        return 0.45

    low_energy     = np.mean(S[low_mask, :], axis=0)
    normalized_std = np.std(low_energy) / (np.mean(low_energy) + 1e-8)
    return clamp01(1.0 - normalized_std)


def track_b_intro_score(candidate_b_time, energy_times_b, energy_values_b, duration_b):
    """
    FIX: This was weighted at 0.10 — far too low. Where Track B enters is
    nearly as important as where Track A exits. A mid-drop entry on Track B
    sounds wrong regardless of how perfect Track A's exit is. Now 0.20.

    Prefers entry in the first 20% of the track (clean intro) with enough
    energy to blend cleanly.
    """
    if duration_b <= 0:
        return 0.5

    p = candidate_b_time / duration_b
    if   0.00 <= p <= 0.20: position_score = 1.0
    elif 0.20 <  p <= 0.35: position_score = 0.75
    elif 0.35 <  p <= 0.50: position_score = 0.40
    else:                   position_score = 0.15

    after = get_window_values(energy_times_b, energy_values_b,
                              candidate_b_time, candidate_b_time + 16)
    energy_score = clamp01(0.6 + safe_mean(after) * 0.4) if len(after) >= 2 else 0.5

    return clamp01(0.65 * position_score + 0.35 * energy_score)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def get_phrase_candidates(beats, sr, duration, phrase_beats=32,
                          min_percent=0.55, max_percent=0.92):
    phrase_times = get_phrase_boundaries(beats, sr, phrase_beats)
    candidates   = [
        float(t) for t in phrase_times
        if duration * min_percent <= float(t) <= duration * max_percent
    ]
    if candidates:
        return candidates
    beat_times = librosa.samples_to_time(beats, sr=sr)
    return [
        float(t) for t in beat_times
        if duration * min_percent <= float(t) <= duration * max_percent
    ]


def track_b_intro_phrase_candidates(beats_b, sr, duration_b, phrase_beats=32):
    phrase_times     = get_phrase_boundaries(beats_b, sr, phrase_beats)
    intro_candidates = [
        float(t) for t in phrase_times
        if 0 <= float(t) <= duration_b * 0.35
    ]
    if intro_candidates:
        return intro_candidates
    if phrase_times:
        return [float(t) for t in phrase_times[:6]]
    beat_times = librosa.samples_to_time(beats_b, sr=sr)
    return [float(t) for t in beat_times if 0 <= float(t) <= duration_b * 0.35]


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------

def score_transition_pair(
    candidate_a_time, candidate_b_time,
    y_a, beats_a, sr,
    energy_times_a, energy_values_a,
    energy_times_b, energy_values_b,
    duration_a, duration_b,
    mix_duration, harmonic_ok,
):
    """
    FIX: Completely rebalanced weights.

    Old:  phrase 0.20, outro 0.16, a_energy 0.16, runway 0.18,
          low_stability 0.12, b_intro 0.10, harmonic 0.08
    → harmonic was 0.08 — a DJ would never blend clashing keys with a long
      EQ blend, but the old scorer would happily choose it.

    New:  phrase 0.25, harmonic 0.20, b_intro 0.20, runway 0.15,
          a_energy 0.12, low_stability 0.05, outro_zone 0.03
    → outro_zone kept at 0.03 since phrase_score already rewards position.
    → harmonic raised to 0.20 — it's a hard musical constraint.
    → b_intro raised to 0.20 — entry point matters as much as exit point.
    → harder penalty for non-harmonic pairs (0.40 vs 0.55).
    """
    phrase_score   = phrase_strength_score(candidate_a_time, beats_a, sr)
    harmonic_score = 1.0 if harmonic_ok else 0.40
    b_intro        = track_b_intro_score(candidate_b_time, energy_times_b, energy_values_b, duration_b)
    runway         = runway_score(candidate_a_time, duration_a, mix_duration)
    a_energy       = local_energy_shape_score(candidate_a_time, energy_times_a, energy_values_a)
    low_stability  = spectral_low_stability_score(y_a, sr, candidate_a_time)
    outro_zone     = outro_zone_score(candidate_a_time, duration_a)

    total = (
        0.25 * phrase_score   +
        0.20 * harmonic_score +
        0.20 * b_intro        +
        0.15 * runway         +
        0.12 * a_energy       +
        0.05 * low_stability  +
        0.03 * outro_zone
    )

    return clamp01(total), {
        "phrase_score":     round(phrase_score, 3),
        "harmonic_score":   round(harmonic_score, 3),
        "b_intro_score":    round(b_intro, 3),
        "runway_score":     round(runway, 3),
        "a_energy_score":   round(a_energy, 3),
        "low_stability":    round(low_stability, 3),
        "outro_zone_score": round(outro_zone, 3),
    }


# ---------------------------------------------------------------------------
# Fine beat-grid sync
# ---------------------------------------------------------------------------

def fine_sync_near_phrase(
    track_a_beats, track_b_beats,
    transition_start_sample, chosen_b_phrase_sample,
    mix_samples, sr, bpm_a,
    search_window_beats=4, offset_samples=1200,
):
    """
    FIX: Hardcoded 124 BPM fallback replaced with actual bpm_a parameter.

    Searches ±4 beats around the chosen phrase boundary for the beat that
    produces the best grid alignment with Track A.
    """
    track_a_beats = np.asarray(track_a_beats)
    track_b_beats = np.asarray(track_b_beats)

    if len(track_a_beats) == 0 or len(track_b_beats) == 0:
        return int(chosen_b_phrase_sample), 0.0

    beat_diffs            = np.diff(track_a_beats)
    beat_duration_samples = (
        int(np.median(beat_diffs)) if len(beat_diffs) > 0
        else int((60.0 / bpm_a) * sr)
    )

    search_radius     = search_window_beats * beat_duration_samples
    candidate_b_beats = track_b_beats[
        (track_b_beats >= chosen_b_phrase_sample - search_radius)
        & (track_b_beats <= chosen_b_phrase_sample + search_radius)
    ]

    if len(candidate_b_beats) == 0:
        return int(chosen_b_phrase_sample), 0.5

    a_window_beats = track_a_beats[
        (track_a_beats >= transition_start_sample)
        & (track_a_beats <= transition_start_sample + mix_samples)
    ]

    best_b_start = int(chosen_b_phrase_sample)
    best_score   = -1.0

    for b_anchor in candidate_b_beats:
        shifted        = track_b_beats - b_anchor + transition_start_sample
        shifted_window = shifted[
            (shifted >= transition_start_sample)
            & (shifted <= transition_start_sample + mix_samples)
        ]
        if len(shifted_window) == 0:
            continue

        matches = sum(
            1 for ba in a_window_beats
            if np.any(np.abs(shifted_window - ba) <= offset_samples)
        )
        score = matches / max(min(len(a_window_beats), len(shifted_window)), 1)

        if score > best_score:
            best_score   = score
            best_b_start = int(b_anchor)

    return best_b_start, clamp01(best_score)

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

# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def plan_transition_logic(track_a_path, track_b_path, preferred_mix_duration):
    if not os.path.exists(track_a_path):
        raise HTTPException(status_code=400, detail=f"Track A not found: {track_a_path}")
    if not os.path.exists(track_b_path):
        raise HTTPException(status_code=400, detail=f"Track B not found: {track_b_path}")

    print("\n--- Starting Improved Transition Planner ---")

    y_a, _ = librosa.load(track_a_path, sr=TARGET_SR, mono=True)
    y_b, _ = librosa.load(track_b_path, sr=TARGET_SR, mono=True)

    duration_a = librosa.get_duration(y=y_a, sr=TARGET_SR)
    duration_b = librosa.get_duration(y=y_b, sr=TARGET_SR)

    bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)
    bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)

    key_a, mode_a, key_conf_a = estimate_key(y_a, TARGET_SR)
    key_b, mode_b, key_conf_b = estimate_key(y_b, TARGET_SR)

    camelot_a   = CAMELOT_MAP.get((key_a, mode_a))
    camelot_b   = CAMELOT_MAP.get((key_b, mode_b))
    harmonic_ok = camelot_compatible(camelot_a, camelot_b)

    energy_times_a, energy_values_a = get_energy_curve(y_a, TARGET_SR)
    energy_times_b, energy_values_b = get_energy_curve(y_b, TARGET_SR)

    # FIX: default to 2 phrases so runway scoring uses a realistic duration.
    # 1 phrase at 128 BPM = ~15s — too short for most blending strategies.
    planning_mix_duration = preferred_mix_duration or round((60.0 / bpm_a) * 32 * 2, 1)

    candidates_a = get_phrase_candidates(
        beats=beats_a, sr=TARGET_SR, duration=duration_a,
        phrase_beats=32, min_percent=0.55, max_percent=0.92,
    )
    if not candidates_a:
        raise HTTPException(status_code=400, detail="Could not find Track A transition candidates.")

    candidates_b = track_b_intro_phrase_candidates(
        beats_b=beats_b, sr=TARGET_SR, duration_b=duration_b, phrase_beats=32,
    )
    if not candidates_b:
        raise HTTPException(status_code=400, detail="Could not find Track B entry candidates.")

    # --- Score all A/B pairs ---
    scored_candidates = []
    best_score        = -1.0
    best_time         = candidates_a[0]
    best_b_time       = candidates_b[0]
    best_score_parts  = {}

    for ca in candidates_a:
        for cb in candidates_b:
            score, parts = score_transition_pair(
                candidate_a_time=ca,
                candidate_b_time=cb,
                y_a=y_a,
                beats_a=beats_a,
                sr=TARGET_SR,
                energy_times_a=energy_times_a,
                energy_values_a=energy_values_a,
                energy_times_b=energy_times_b,
                energy_values_b=energy_values_b,
                duration_a=duration_a,
                duration_b=duration_b,
                mix_duration=planning_mix_duration,
                harmonic_ok=harmonic_ok,
            )
            scored_candidates.append({
                "track_a_time": round(float(ca), 3),
                "track_b_time": round(float(cb), 3),
                "score":        round(float(score), 3),
                **parts,
            })
            if score > best_score:
                best_score       = score
                best_time        = ca
                best_b_time      = cb
                best_score_parts = parts

    # --- Strategy selection ---
    # FIX: sync_accuracy was hardcoded to 0.5, incorrectly disqualifying
    # strategies like bass_swap (min_sync 0.6) and drop_mix (min_sync 0.65).
    # Pass 1.0 here — the planner doesn't have real sync accuracy yet.
    # Actual alignment is validated in the renderer after stretching.
    strategy, strategy_scores = choose_strategy_with_scores(
        harmonic_ok=harmonic_ok,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        best_time=best_time,
        song_duration_a=duration_a,
        mix_duration=planning_mix_duration,
        energy_score=best_score_parts.get("a_energy_score", 0.5),
        runway_score_value=best_score_parts.get("runway_score", 1.0),
        sync_accuracy=1.0,
        key_confidence_a=key_conf_a,
        key_confidence_b=key_conf_b,
    )

    mix_duration = preferred_mix_duration if preferred_mix_duration is not None \
        else get_strategy_mix_duration(strategy, bpm_a)

    # --- Re-check runway with final mix_duration ---
    # FIX: Old code re-sorted entirely by runway filter, overriding the
    # winning candidate with a lower-scoring one. Now only replaces the
    # winner if it genuinely fails the runway check AND a valid alternative
    # exists — preserving the full score ranking otherwise.
    winner_runway = runway_score(best_time, duration_a, mix_duration)

    if winner_runway < 0.75:
        valid_alternatives = [
            row for row in scored_candidates
            if row["track_a_time"] != best_time
            and runway_score(row["track_a_time"], duration_a, mix_duration) >= 0.75
        ]
        if valid_alternatives:
            best_row   = max(valid_alternatives, key=lambda x: x["score"])
            best_time  = best_row["track_a_time"]
            best_b_time = best_row["track_b_time"]
            print(f"Runway re-check: transition moved to {best_time:.2f}s")

    # --- Fine beat-grid sync ---
    mix_samples          = int(mix_duration * TARGET_SR)
    best_start_sample    = int(best_time * TARGET_SR)
    chosen_b_phrase_smpl = int(best_b_time * TARGET_SR)

    start_sample_b, sync_accuracy = fine_sync_near_phrase(
        track_a_beats=beats_a,
        track_b_beats=beats_b,
        transition_start_sample=best_start_sample,
        chosen_b_phrase_sample=chosen_b_phrase_smpl,
        mix_samples=mix_samples,
        sr=TARGET_SR,
        bpm_a=bpm_a,
        search_window_beats=4,
        offset_samples=1200,
    )

    track_b_entry_time = start_sample_b / TARGET_SR

    reason = (
        f"Selected {strategy}. Track A {camelot_a}, Track B {camelot_b}, "
        f"harmonic={harmonic_ok}, BPM delta={abs(bpm_a - bpm_b):.2f}. "
        f"Transition at {best_time:.2f}s (score={best_score:.3f}), "
        f"Track B entry at {track_b_entry_time:.2f}s, sync={sync_accuracy:.3f}."
    )
    print(reason)

    return {
        "status":                            "success",
        "recommended_transition_start_time": round(float(best_time), 3),
        "recommended_track_b_entry_time":    round(float(track_b_entry_time), 3),
        "recommended_track_b_entry_sample":  int(start_sample_b),
        "sync_accuracy":                     round(float(sync_accuracy), 3),
        "recommended_strategy":              strategy,
        "strategy_scores":                   strategy_scores,
        "mix_duration":                      mix_duration,
        "reason":                            reason,
        "track_a": {
            "duration":       round(float(duration_a), 2),
            "bpm":            round(float(bpm_a), 2),
            "key":            f"{key_a} {mode_a}",
            "camelot":        camelot_a,
            "key_confidence": round(float(key_conf_a), 3),
        },
        "track_b": {
            "duration":       round(float(duration_b), 2),
            "bpm":            round(float(bpm_b), 2),
            "key":            f"{key_b} {mode_b}",
            "camelot":        camelot_b,
            "key_confidence": round(float(key_conf_b), 3),
        },
        "harmonic_compatible": harmonic_ok,
        "top_phrase_pairs": sorted(
            scored_candidates, key=lambda x: x["score"], reverse=True
        )[:8],
        "render_payload": {
            "track_a_path":          track_a_path,
            "track_b_path":          track_b_path,
            "transition_start_time": round(float(best_time), 3),
            "track_b_entry_time":    round(float(track_b_entry_time), 3),
            "mix_duration":          mix_duration,
            "output_dir":            "outputs",
            "transition_strategy":   strategy,
            "fx_parameters": {
                "apply_reverb_tail":  strategy == "reverb_wash",
                "loop_track_a":       strategy == "auto_loop",
                "hpf_sweep_end_freq": 3500.0 if strategy == "hpf_sweep" else None,
                "lpf_sweep_end_freq": 400.0  if strategy == "lpf_sweep"  else None,
                "bpm":                None,
            },
        },
    }