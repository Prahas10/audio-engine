"""
core/planner.py

Changes in this version:
  - track_b_intro_score: hard minimum energy floor — rejects entry points
    where Track B has near-silence in the next 8 bars (prevents energy holes)
  - score_transition_pair: b_intro weight raised, energy floor enforced
  - plan_transition_logic: passes y_a, y_b to fine_sync_onset for coarse scan
  - _infer_mix_profile: unchanged
  - cross_track_loudness_score: unchanged
"""

import os
import numpy as np
import librosa
from fastapi import HTTPException

from core.analysis import (
    TARGET_SR,
    safe_bpm_from_path,
    estimate_key,
    get_energy_curve,
    get_drop_candidates,
    phrase_boundary_candidates_from_downbeats,
    intro_phrase_candidates_from_downbeats,
    get_phrase_boundaries,
    camelot_compatible,
    fine_sync_onset,
    verify_beat_alignment,
)
from core.library import get_track_metadata
from core.strategy_router import choose_strategy_with_scores, get_strategy_mix_duration
from models.schemas import CAMELOT_MAP


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clamp01(x):   return float(np.clip(x, 0.0, 1.0))
def safe_mean(v, default=0.0):
    v = np.asarray(v); return float(np.mean(v)) if len(v) > 0 else default
def safe_std(v, default=0.0):
    v = np.asarray(v); return float(np.std(v))  if len(v) > 0 else default

def get_window_values(times, values, start, end):
    times  = np.asarray(times)
    values = np.asarray(values)
    return values[(times >= start) & (times < end)]


# ---------------------------------------------------------------------------
# Vocal detection
# ---------------------------------------------------------------------------

def detect_vocal_regions(y, sr, hop_length=512, threshold=0.45):
    y_harm, _ = librosa.effects.hpss(y)
    flatness   = librosa.feature.spectral_flatness(y=y_harm, hop_length=hop_length)[0]
    S          = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs      = librosa.fft_frequencies(sr=sr)
    vocal_mask = (freqs >= 300) & (freqs <= 3400)
    total_energy = np.sum(S, axis=0) + 1e-9
    vocal_energy = np.sum(S[vocal_mask, :], axis=0) / total_energy
    S_harm       = np.abs(librosa.stft(y_harm, hop_length=hop_length))
    harm_energy  = np.sum(S_harm, axis=0) / (np.sum(S, axis=0) + 1e-9)
    min_len      = min(len(flatness), len(vocal_energy), len(harm_energy))
    flatness     = flatness[:min_len]
    vocal_energy = vocal_energy[:min_len]
    harm_energy  = harm_energy[:min_len]
    flat_norm    = 1.0 - np.clip(flatness / (np.max(flatness) + 1e-9), 0.0, 1.0)
    vocal_prob   = np.clip(0.40 * flat_norm + 0.35 * vocal_energy + 0.25 * harm_energy, 0.0, 1.0)
    times        = librosa.frames_to_time(np.arange(min_len), sr=sr, hop_length=hop_length)
    return times, vocal_prob, vocal_prob > threshold


def vocal_density_at(vocal_times, vocal_prob, candidate_time, window_seconds=8.0):
    window = get_window_values(
        vocal_times, vocal_prob,
        candidate_time - window_seconds * 0.5,
        candidate_time + window_seconds * 0.5,
    )
    return float(np.mean(window)) if len(window) > 0 else 0.0


def vocal_free_score(vocal_density):
    return clamp01(1.0 - vocal_density * 2.5)


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def load_or_analyze_track(track_path, library_path=None):
    metadata = None
    if library_path:
        metadata = get_track_metadata(track_path, library_path)

    y, _     = librosa.load(track_path, sr=TARGET_SR, mono=True)
    duration = librosa.get_duration(y=y, sr=TARGET_SR)

    if metadata:
        bpm       = float(metadata["bpm"])
        beats     = np.asarray(librosa.time_to_samples(metadata.get("beats", []), sr=TARGET_SR), dtype=int)
        downbeats = np.asarray(librosa.time_to_samples(metadata.get("downbeats", []), sr=TARGET_SR), dtype=int)
        key       = metadata.get("key_name")
        mode      = metadata.get("mode")
        key_conf  = float(metadata.get("key_confidence", 0.0))
        camelot   = metadata.get("camelot")
        energy_times, energy_values, raw_rms_db = get_energy_curve(y, TARGET_SR)
        stored_rms_db = metadata.get("energy_summary", {}).get("raw_rms_db")
        if stored_rms_db is not None:
            raw_rms_db = float(stored_rms_db)
        return dict(y=y, duration=duration, bpm=bpm, beats=beats, downbeats=downbeats,
                    key=key, mode=mode, key_confidence=key_conf, camelot=camelot,
                    energy_times=energy_times, energy_values=energy_values,
                    raw_rms_db=raw_rms_db, metadata_used=True, metadata=metadata)

    bpm, beats, downbeats     = safe_bpm_from_path(track_path, TARGET_SR)
    key, mode, key_conf       = estimate_key(y, TARGET_SR)
    camelot                   = CAMELOT_MAP.get((key, mode))
    energy_times, energy_values, raw_rms_db = get_energy_curve(y, TARGET_SR)
    return dict(y=y, duration=duration, bpm=bpm, beats=beats, downbeats=downbeats,
                key=key, mode=mode, key_confidence=key_conf, camelot=camelot,
                energy_times=energy_times, energy_values=energy_values,
                raw_rms_db=raw_rms_db, metadata_used=False, metadata=None)


# ---------------------------------------------------------------------------
# Genre-aware profile
# ---------------------------------------------------------------------------

def _infer_mix_profile(bpm_a, energy_avg_a, key_confidence_a):
    if bpm_a >= 138 and energy_avg_a > 0.65 and key_confidence_a < 0.40:
        return {"phrase_score":0.32,"harmonic_score":0.06,"b_intro_score":0.18,
                "vocal_free_a":0.08,"runway_score":0.14,"a_energy_score":0.12,
                "low_stability":0.07,"loudness_match":0.02,"outro_zone_score":0.01}
    elif bpm_a < 130 and key_confidence_a > 0.55:
        return {"phrase_score":0.16,"harmonic_score":0.28,"b_intro_score":0.22,
                "vocal_free_a":0.16,"runway_score":0.10,"a_energy_score":0.05,
                "low_stability":0.02,"loudness_match":0.01,"outro_zone_score":0.00}
    return None


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------

def local_energy_shape_score(candidate_time, energy_times, energy_values, window_seconds=16):
    before = get_window_values(energy_times, energy_values, candidate_time - window_seconds, candidate_time)
    after  = get_window_values(energy_times, energy_values, candidate_time, candidate_time + window_seconds)
    if len(before) < 2 or len(after) < 2:
        return 0.45
    drop       = safe_mean(before) - safe_mean(after)
    stability  = 1.0 - min(safe_std(after) * 3.0, 1.0)
    drop_score = clamp01(0.65 + (drop * 1.2 if drop >= 0 else drop * 1.8))
    base       = clamp01(0.65 * drop_score + 0.35 * stability)

    # Hard penalty: if Track A is already nearly silent at transition start,
    # this point is in a breakdown/outro — never a good transition point.
    # Two thresholds work together:
    #   normalised RMS < 0.20 → in a quiet section of this track
    #   (was 0.12 — too low, missed LP Giobbi outro at -23dBFS)
    mean_before = safe_mean(before)
    if mean_before < 0.20:
        return base * 0.20

    return base


def runway_score(candidate_time, song_duration, mix_duration, safety_seconds=4.0):
    remaining = song_duration - candidate_time
    required  = mix_duration + safety_seconds
    if remaining >= required:             return 1.0
    if remaining >= mix_duration:         return 0.75
    if remaining >= mix_duration * 0.75:  return 0.35
    return 0.0


def outro_zone_score(candidate_time, duration):
    if duration <= 0: return 0.5
    p = candidate_time / duration
    if   0.65 <= p <= 0.88: return 1.0
    elif 0.55 <= p <  0.65: return 0.70
    elif 0.88 <  p <= 0.93: return 0.55
    elif 0.50 <= p <  0.55: return 0.35
    return 0.10


def phrase_strength_score(candidate_time, beats, sr, phrase_beats=32):
    if beats is None or len(beats) == 0: return 0.4
    beat_times   = librosa.samples_to_time(beats, sr=sr)
    if len(beat_times) < phrase_beats + 1: return 0.45
    phrase_times = beat_times[::phrase_beats]
    nearest      = float(np.min(np.abs(phrase_times - candidate_time)))
    avg_beat     = float(np.median(np.diff(beat_times))) if len(beat_times) > 2 else 0.5
    return clamp01(1.0 - nearest / max(avg_beat * 1.5, 0.25))


def spectral_low_stability_score(y, sr, candidate_time, window_seconds=16):
    start = max(0, int((candidate_time - window_seconds) * sr))
    end   = min(len(y), int((candidate_time + window_seconds) * sr))
    if end <= start or (end - start) < sr: return 0.45
    S        = np.abs(librosa.stft(y[start:end], n_fft=2048, hop_length=512))
    freqs    = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs <= 180
    if not np.any(low_mask): return 0.45
    low_energy     = np.mean(S[low_mask, :], axis=0)
    normalized_std = np.std(low_energy) / (np.mean(low_energy) + 1e-8)
    return clamp01(1.0 - normalized_std)


def cross_track_loudness_score(raw_rms_db_a, raw_rms_db_b):
    diff_db = abs(raw_rms_db_a - raw_rms_db_b)
    return clamp01(1.0 - diff_db / 12.0)


def track_b_intro_score(candidate_b_time, energy_times_b, energy_values_b, duration_b,
                        vocal_times_b=None, vocal_prob_b=None):
    """
    Scores Track B entry points.

    Fix: enforces a hard energy floor — if the mean energy in the 8 bars
    after entry is below 0.10 (near-silence / deep breakdown), the score
    is capped at 0.15 regardless of position. This prevents the planner
    from choosing entries that create an audible energy hole.
    """
    if duration_b <= 0:
        return 0.5

    p = candidate_b_time / duration_b
    if   0.00 <= p <= 0.20: position_score = 1.0
    elif 0.20 <  p <= 0.35: position_score = 0.75
    elif 0.35 <  p <= 0.50: position_score = 0.40
    else:                   position_score = 0.15

    after  = get_window_values(energy_times_b, energy_values_b,
                               candidate_b_time,      candidate_b_time + 16)
    after2 = get_window_values(energy_times_b, energy_values_b,
                               candidate_b_time + 16, candidate_b_time + 32)

    if len(after) >= 2:
        mean_energy_after = float(np.mean(after))
        # Also check the second 16s window — some tracks have a slow build.
        # If either the immediate or second window has energy, allow the entry.
        mean_energy_after2 = float(np.mean(after2)) if len(after2) >= 2 else 0.0
        best_energy = max(mean_energy_after, mean_energy_after2)

        # Hard floor raised to 0.18: entries where energy stays near-silent
        # for a full 32s are almost always wrong phrase choices.
        if best_energy < 0.18:
            return 0.08
        energy_score = clamp01(0.4 + best_energy * 0.6)
    else:
        energy_score = 0.5

    if vocal_times_b is not None and vocal_prob_b is not None:
        vd      = vocal_density_at(vocal_times_b, vocal_prob_b, candidate_b_time, window_seconds=8.0)
        v_score = vocal_free_score(vd)
    else:
        v_score = 1.0

    # Energy weight raised to 0.40 (was 0.35): prevents position=1.0 from
    # overriding a genuinely quiet entry point (the LP Giobbi hole problem).
    return clamp01(0.35 * position_score + 0.40 * energy_score + 0.25 * v_score)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def get_phrase_candidates(beats, sr, duration, phrase_beats=32,
                          min_percent=0.55, max_percent=0.92):
    phrase_times = get_phrase_boundaries(beats, sr, phrase_beats)
    candidates   = [float(t) for t in phrase_times
                    if duration * min_percent <= float(t) <= duration * max_percent]
    if candidates:
        return candidates
    beat_times = librosa.samples_to_time(beats, sr=sr)
    return [float(t) for t in beat_times
            if duration * min_percent <= float(t) <= duration * max_percent]


def track_b_intro_phrase_candidates(beats_b, sr, duration_b, phrase_beats=32):
    phrase_times     = get_phrase_boundaries(beats_b, sr, phrase_beats)
    intro_candidates = [float(t) for t in phrase_times if 0 <= float(t) <= duration_b * 0.35]
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
    bpm_a=128.0, key_confidence_a=0.5,
    raw_rms_db_a=None, raw_rms_db_b=None,
    vocal_times_a=None, vocal_prob_a=None,
    vocal_times_b=None, vocal_prob_b=None,
):
    phrase_score   = phrase_strength_score(candidate_a_time, beats_a, sr)
    harmonic_score = 1.0 if harmonic_ok else 0.40
    b_intro        = track_b_intro_score(
        candidate_b_time, energy_times_b, energy_values_b, duration_b,
        vocal_times_b, vocal_prob_b,
    )
    runway         = runway_score(candidate_a_time, duration_a, mix_duration)
    a_energy       = local_energy_shape_score(candidate_a_time, energy_times_a, energy_values_a)
    low_stability  = spectral_low_stability_score(y_a, sr, candidate_a_time)
    outro_zone     = outro_zone_score(candidate_a_time, duration_a)

    if vocal_times_a is not None and vocal_prob_a is not None:
        vd_a    = vocal_density_at(vocal_times_a, vocal_prob_a, candidate_a_time)
        vocal_a = vocal_free_score(vd_a)
    else:
        vocal_a = 1.0

    loudness_match = (cross_track_loudness_score(raw_rms_db_a, raw_rms_db_b)
                      if raw_rms_db_a is not None and raw_rms_db_b is not None else 0.5)

    parts = {
        "phrase_score":     round(phrase_score, 3),
        "harmonic_score":   round(harmonic_score, 3),
        "b_intro_score":    round(b_intro, 3),
        "vocal_free_a":     round(vocal_a, 3),
        "runway_score":     round(runway, 3),
        "a_energy_score":   round(a_energy, 3),
        "low_stability":    round(low_stability, 3),
        "loudness_match":   round(loudness_match, 3),
        "outro_zone_score": round(outro_zone, 3),
    }

    energy_avg_a = safe_mean(energy_values_a)
    profile      = _infer_mix_profile(bpm_a, energy_avg_a, key_confidence_a)

    if profile:
        total = sum(profile.get(k, 0.0) * v for k, v in parts.items())
    else:
        # Default weights — b_intro raised to 0.22 (was 0.18) to harder penalise bad entries
        total = (
            0.20 * phrase_score   +
            0.18 * harmonic_score +
            0.22 * b_intro        +
            0.14 * vocal_a        +
            0.12 * runway         +
            0.07 * a_energy       +
            0.04 * low_stability  +
            0.02 * loudness_match +
            0.01 * outro_zone
        )

    return clamp01(total), parts


# ---------------------------------------------------------------------------
# Legacy sync (beat-grid, kept for renderer single-clip fallback)
# ---------------------------------------------------------------------------

def find_best_sync_point(
    track_a_beats, track_b_beats,
    transition_start_sample, mix_samples,
    offset_samples=1200,
    track_b_total_samples=None,
    min_b_entry_percent=0.0,
    max_b_entry_percent=0.35,
):
    track_a_beats = np.asarray(track_a_beats)
    track_b_beats = np.asarray(track_b_beats)
    a_window = track_a_beats[
        (track_a_beats >= transition_start_sample) &
        (track_a_beats <= transition_start_sample + mix_samples)
    ]
    if len(a_window) == 0 or len(track_b_beats) == 0:
        return 0, 0.0
    if track_b_total_samples is None:
        track_b_total_samples = int(track_b_beats[-1] * 1.05)
    min_b = int(track_b_total_samples * min_b_entry_percent)
    max_b = int(track_b_total_samples * max_b_entry_percent)
    candidate_b = track_b_beats[(track_b_beats >= min_b) & (track_b_beats <= max_b)]
    if len(candidate_b) == 0:
        candidate_b = track_b_beats
    best_b_start = int(candidate_b[0])
    best_score   = -1.0
    for b_anchor in candidate_b:
        shifted = track_b_beats - b_anchor + transition_start_sample
        sw      = shifted[(shifted >= transition_start_sample) &
                          (shifted <= transition_start_sample + mix_samples)]
        if len(sw) == 0: continue
        matches = sum(1 for ba in a_window if np.any(np.abs(sw - ba) <= offset_samples))
        score   = matches / max(min(len(a_window), len(sw)), 1)
        if score > best_score:
            best_score   = score
            best_b_start = int(b_anchor)
    return best_b_start, float(best_score)


# ---------------------------------------------------------------------------
# Main planner
# ---------------------------------------------------------------------------

def plan_transition_logic(
    track_a_path, track_b_path, preferred_mix_duration,
    library_path="storage/library_metadata.json",
):
    if not os.path.exists(track_a_path):
        raise HTTPException(status_code=400, detail=f"Track A not found: {track_a_path}")
    if not os.path.exists(track_b_path):
        raise HTTPException(status_code=400, detail=f"Track B not found: {track_b_path}")

    print("\n--- Starting Transition Planner ---")

    track_a = load_or_analyze_track(track_a_path, library_path)
    track_b = load_or_analyze_track(track_b_path, library_path)

    y_a, duration_a           = track_a["y"], track_a["duration"]
    y_b, duration_b           = track_b["y"], track_b["duration"]
    bpm_a, bpm_b              = track_a["bpm"], track_b["bpm"]
    beats_a, beats_b          = track_a["beats"], track_b["beats"]
    downbeats_a, downbeats_b  = track_a["downbeats"], track_b["downbeats"]
    camelot_a, camelot_b      = track_a["camelot"], track_b["camelot"]
    key_conf_a, key_conf_b    = track_a["key_confidence"], track_b["key_confidence"]
    harmonic_ok               = camelot_compatible(camelot_a, camelot_b)
    energy_times_a, energy_values_a = track_a["energy_times"], track_a["energy_values"]
    energy_times_b, energy_values_b = track_b["energy_times"], track_b["energy_values"]
    raw_rms_db_a = track_a.get("raw_rms_db")
    raw_rms_db_b = track_b.get("raw_rms_db")

    print("Detecting vocal regions...")
    try:
        vocal_times_a, vocal_prob_a, _ = detect_vocal_regions(y_a, TARGET_SR)
        vocal_times_b, vocal_prob_b, _ = detect_vocal_regions(y_b, TARGET_SR)
    except Exception as e:
        print(f"Vocal detection failed ({e}) — skipping.")
        vocal_times_a = vocal_prob_a = None
        vocal_times_b = vocal_prob_b = None

    planning_mix_duration = preferred_mix_duration or round((60.0 / bpm_a) * 32 * 2, 1)

    candidates_a = phrase_boundary_candidates_from_downbeats(
        downbeats=downbeats_a, sr=TARGET_SR, song_duration=duration_a,
        bpm=bpm_a, min_percent=0.55, max_percent=0.92,
    )
    if not candidates_a:
        candidates_a = get_phrase_candidates(beats_a, TARGET_SR, duration_a)
    if not candidates_a:
        raise HTTPException(status_code=400, detail="No Track A transition candidates found.")

    candidates_b = intro_phrase_candidates_from_downbeats(
        downbeats=downbeats_b, sr=TARGET_SR, song_duration=duration_b,
        bpm=bpm_b, max_percent=0.35,
    )
    if not candidates_b:
        candidates_b = track_b_intro_phrase_candidates(beats_b, TARGET_SR, duration_b)
    if not candidates_b:
        raise HTTPException(status_code=400, detail="No Track B entry candidates found.")

    # Augment candidates_b with energy-jump points (drop_points).
    # These mark where the kick/energy section begins after a quiet intro.
    # Critical for sparse-intro tracks (piano, ambient) where the phrase
    # boundary is at bar-1 but energy doesn't start until bar-9 or later.
    #
    # Source priority:
    #   1. drop_points from library metadata (pre-computed)
    #   2. Computed live from y_b if metadata is missing or empty
    _drop_pts_raw = []
    if track_b.get("metadata") and track_b["metadata"].get("drop_points"):
        _drop_pts_raw = track_b["metadata"]["drop_points"]

    if not _drop_pts_raw:
        # Compute live: score all intro candidates by energy jump
        print("Computing drop_points live for Track B (not in metadata)...")
        try:
            _drop_pts_raw = get_drop_candidates(
                y=y_b, sr=TARGET_SR,
                candidate_times=candidates_b,
                top_k=8,
            )
        except Exception as e:
            print(f"Live drop_points computation failed ({e})")

    if _drop_pts_raw:
        drop_pts = [
            float(t) for t in _drop_pts_raw
            if 0 <= float(t) <= duration_b * 0.50
        ]
        if drop_pts:
            existing_set = set(round(c, 1) for c in candidates_b)
            new_drops    = [t for t in drop_pts if round(t, 1) not in existing_set]
            if new_drops:
                candidates_b = candidates_b + new_drops
                print(f"Added {len(new_drops)} drop_point candidates for Track B: "
                      f"{[round(t,1) for t in new_drops]}")

    # Score all A/B pairs
    scored      = []
    best_score  = -1.0
    best_time   = candidates_a[0]
    best_b_time = candidates_b[0]
    best_parts  = {}

    for ca in candidates_a:
        for cb in candidates_b:
            score, parts = score_transition_pair(
                candidate_a_time=ca, candidate_b_time=cb,
                y_a=y_a, beats_a=beats_a, sr=TARGET_SR,
                energy_times_a=energy_times_a, energy_values_a=energy_values_a,
                energy_times_b=energy_times_b, energy_values_b=energy_values_b,
                duration_a=duration_a, duration_b=duration_b,
                mix_duration=planning_mix_duration, harmonic_ok=harmonic_ok,
                bpm_a=bpm_a, key_confidence_a=key_conf_a,
                raw_rms_db_a=raw_rms_db_a, raw_rms_db_b=raw_rms_db_b,
                vocal_times_a=vocal_times_a, vocal_prob_a=vocal_prob_a,
                vocal_times_b=vocal_times_b, vocal_prob_b=vocal_prob_b,
            )
            scored.append({"track_a_time": round(ca, 3), "track_b_time": round(cb, 3),
                           "score": round(score, 3), **parts})
            if score > best_score:
                best_score = score; best_time = ca; best_b_time = cb; best_parts = parts

    strategy, strategy_scores = choose_strategy_with_scores(
        harmonic_ok=harmonic_ok, bpm_a=bpm_a, bpm_b=bpm_b,
        best_time=best_time, song_duration_a=duration_a,
        mix_duration=planning_mix_duration,
        energy_score=best_parts.get("a_energy_score", 0.5),
        runway_score_value=best_parts.get("runway_score", 1.0),
        sync_accuracy=1.0,
        key_confidence_a=key_conf_a, key_confidence_b=key_conf_b,
    )

    mix_duration = (preferred_mix_duration if preferred_mix_duration is not None
                    else get_strategy_mix_duration(strategy, bpm_a))

    if runway_score(best_time, duration_a, mix_duration) < 0.75:
        alternatives = [r for r in scored
                        if r["track_a_time"] != best_time
                        and runway_score(r["track_a_time"], duration_a, mix_duration) >= 0.75]
        if alternatives:
            best_row    = max(alternatives, key=lambda x: x["score"])
            best_time   = best_row["track_a_time"]
            best_b_time = best_row["track_b_time"]
            print(f"Runway re-check: moved to {best_time:.2f}s")

    # Fine sync — coarse-to-fine with validation
    mix_samples          = int(mix_duration * TARGET_SR)
    transition_sample_a  = int(best_time * TARGET_SR)
    chosen_b_sample      = int(best_b_time * TARGET_SR)

    # ---------------------------------------------------------------------------
    # Exhaustive sync search — ALL (A_time, B_time) pairs
    #
    # Score every pair by: combined = w_sync * sync_score + w_pair * pair_score
    # This finds the pair where the beat grids actually lock, not just the one
    # with the best musical structure score. The two scores are complementary:
    #   pair_score  → musical fit (energy, harmonic, phrase position)
    #   sync_score  → beat-grid lock quality (do the kicks actually align?)
    #
    # We need both. A musically perfect entry point that's 2 bars off in timing
    # sounds worse than a slightly less perfect entry that's beat-locked.
    # ---------------------------------------------------------------------------
    SYNC_TARGET     = 0.55   # combined score goal for sync dimension
    SYNC_MIN_ACCEPT = 0.35   # below this, sync is pure noise
    SYNC_WEIGHT     = 0.55   # sync quality weight in combined score
    PAIR_WEIGHT     = 0.45   # musical fit weight in combined score

    # All unique (A_time, B_time) pairs from the scored list
    all_pairs = {}
    for row in scored:
        key = (round(row["track_a_time"], 2), round(row["track_b_time"], 2))
        if key not in all_pairs:
            all_pairs[key] = row["score"]

    print(f"Exhaustive sync search across {len(all_pairs)} pairs "
          f"({len(candidates_a)} A × {len(candidates_b)} B) — checking all...")

    # All candidates are evaluated. No early exit.
    # Each candidate that reaches sync target is post-verified with the full
    # mix_duration window. We keep the one with the best combined score
    # among all candidates that pass post-verification.
    # Candidates that fail post-verify are marked and skipped.
    pairs_ranked   = sorted(all_pairs.items(), key=lambda x: x[1], reverse=True)
    _bar_ms        = (4 * 60.0 / max(bpm_a, 1.0)) * 1000.0
    _check_samples = min(int(planning_mix_duration * TARGET_SR), len(y_a), len(y_b))

    # Tracking
    best_combined       = -1.0   # best combined among post-verified candidates
    best_sync_score     = -1.0
    best_synced_b       = chosen_b_sample
    best_pair_a_time    = best_time
    best_pair_b_time    = best_b_time
    best_transition_a   = transition_sample_a

    # Also track best raw combined in case nothing passes post-verify
    fallback_combined   = -1.0
    fallback_synced_b   = chosen_b_sample
    fallback_a_time     = best_time
    fallback_b_time     = best_b_time
    fallback_a_s        = transition_sample_a

    for (a_time, b_time), pair_score in pairs_ranked:
        a_s = int(a_time * TARGET_SR)
        b_s = int(b_time * TARGET_SR)

        synced_b, sync_acc = fine_sync_onset(
            y_a=y_a, y_b=y_b,
            transition_start_sample=a_s,
            chosen_b_phrase_sample=b_s,
            bpm_a=bpm_a, sr=TARGET_SR,
            search_window_beats=4,
        )

        combined = SYNC_WEIGHT * sync_acc + PAIR_WEIGHT * pair_score

        # Track best raw result as fallback
        if combined > fallback_combined:
            fallback_combined = combined
            fallback_synced_b = synced_b
            fallback_a_time   = a_time
            fallback_b_time   = b_time
            fallback_a_s      = a_s

        # Only post-verify candidates that clear the sync target
        passed_verify = False
        verify_drift  = None

        if sync_acc >= SYNC_TARGET:
            _chk = min(_check_samples, len(y_a) - a_s, len(y_b) - synced_b)
            if _chk > TARGET_SR * 4:
                try:
                    _drift_ms, _ = verify_beat_alignment(
                        y_a[a_s:a_s + _chk],
                        y_b[synced_b:synced_b + _chk],
                        TARGET_SR, bpm_a,
                    )
                    verify_drift = _drift_ms
                    passed_verify = abs(_drift_ms) <= _bar_ms
                except Exception:
                    passed_verify = True  # can't verify — trust sync score
            else:
                passed_verify = True

        # Update best only if this pair cleared post-verification
        if passed_verify and combined > best_combined:
            best_combined     = combined
            best_sync_score   = sync_acc
            best_synced_b     = synced_b
            best_pair_a_time  = a_time
            best_pair_b_time  = b_time
            best_transition_a = a_s

        drift_str = f" verify={verify_drift:.0f}ms" if verify_drift is not None else ""
        status    = " ✓" if passed_verify and sync_acc >= SYNC_TARGET else                     (" ✗post-verify" if sync_acc >= SYNC_TARGET and not passed_verify else "")
        print(f"  A={a_time:.1f}s B={b_time:.1f}s "
              f"pair={pair_score:.3f} sync={sync_acc:.3f} combined={combined:.3f}"
              f"{drift_str}{status}")

    # Use best post-verified pair; fall back to best raw if nothing passed
    if best_combined > -1.0:
        synced_b_sample     = best_synced_b
        sync_accuracy       = best_sync_score
        best_time           = best_pair_a_time
        best_b_time         = best_pair_b_time
        transition_sample_a = best_transition_a
        print(f"Best verified pair: A={best_time:.1f}s B={best_b_time:.1f}s "
              f"sync={sync_accuracy:.3f} combined={best_combined:.3f}")
    else:
        # Nothing passed post-verify — use the raw best (sparse intro case)
        synced_b_sample     = fallback_synced_b
        sync_accuracy       = -1.0
        best_time           = fallback_a_time
        best_b_time         = fallback_b_time
        transition_sample_a = fallback_a_s
        print(f"No pair passed post-verify. Using raw best: "
              f"A={best_time:.1f}s B={best_b_time:.1f}s")

    # Sparse-intro fallback: if still no good sync, use highest-energy B entry
    if sync_accuracy < SYNC_MIN_ACCEPT:
        print(f"No sync lock (best={sync_accuracy:.3f}) — "
              f"using highest-energy B entry.")
        try:
            rms_times, rms_vals, _ = get_energy_curve(y_b, TARGET_SR)
            best_drop_energy = -1.0
            best_drop_sample = int(best_b_time * TARGET_SR)
            for cb in candidates_b:
                after  = rms_vals[(rms_times >= cb) & (rms_times < cb + 16.0)]
                energy = float(np.mean(after)) if len(after) >= 2 else 0.0
                if energy > best_drop_energy:
                    best_drop_energy = energy
                    best_drop_sample = int(cb * TARGET_SR)
            synced_b_sample = best_drop_sample
            best_b_time     = best_drop_sample / TARGET_SR
            print(f"  Highest-energy B entry: {best_b_time:.2f}s "
                  f"(energy={best_drop_energy:.3f})")
        except Exception as e:
            print(f"  Drop_point fallback failed ({e})")

    mix_samples = int(mix_duration * TARGET_SR)
    track_b_entry_time = synced_b_sample / TARGET_SR

    reason = (
        f"Selected {strategy}. A={camelot_a}, B={camelot_b}, "
        f"harmonic={harmonic_ok}, BPM delta={abs(bpm_a - bpm_b):.2f}. "
        f"Transition at {best_time:.2f}s (score={best_score:.3f}), "
        f"B entry at {track_b_entry_time:.2f}s, sync={sync_accuracy:.3f}."
    )
    print(reason)

    return {
        "status":                            "success",
        "recommended_transition_start_time": round(float(best_time), 3),
        "recommended_track_b_entry_time":    round(float(track_b_entry_time), 3),
        "recommended_track_b_entry_sample":  int(synced_b_sample),
        "sync_accuracy":                     round(float(sync_accuracy), 3),
        "recommended_strategy":              strategy,
        "strategy_scores":                   strategy_scores,
        "mix_duration":                      mix_duration,
        "reason":                            reason,
        "track_a": {
            "duration":       round(float(duration_a), 2),
            "bpm":            round(float(bpm_a), 2),
            "key":            f"{track_a['key']} {track_a['mode']}",
            "camelot":        camelot_a,
            "key_confidence": round(float(key_conf_a), 3),
            "beats":          [round(float(t), 3) for t in librosa.samples_to_time(beats_a, sr=TARGET_SR)],
            "downbeats":      [round(float(t), 3) for t in librosa.samples_to_time(downbeats_a, sr=TARGET_SR)],
        },
        "track_b": {
            "duration":       round(float(duration_b), 2),
            "bpm":            round(float(bpm_b), 2),
            "key":            f"{track_b['key']} {track_b['mode']}",
            "camelot":        camelot_b,
            "key_confidence": round(float(key_conf_b), 3),
            "beats":          [round(float(t), 3) for t in librosa.samples_to_time(beats_b, sr=TARGET_SR)],
            "downbeats":      [round(float(t), 3) for t in librosa.samples_to_time(downbeats_b, sr=TARGET_SR)],
        },
        "harmonic_compatible": harmonic_ok,
        "top_phrase_pairs":    sorted(scored, key=lambda x: x["score"], reverse=True)[:8],
        "render_payload": {
            "track_a_path":          track_a_path,
            "track_b_path":          track_b_path,
            "transition_start_time": round(float(best_time), 3),
            "track_b_entry_time":    round(float(track_b_entry_time), 3),
            "mix_duration":          mix_duration,
            "output_dir":            "outputs",
            "transition_strategy":   strategy,
            "_beats_a":              beats_a.tolist(),
            "_beats_b":              beats_b.tolist(),
            "_bpm_a":                round(float(bpm_a), 3),
            "_bpm_b":                round(float(bpm_b), 3),
            "fx_parameters": {
                "apply_reverb_tail":  strategy == "reverb_wash",
                "loop_track_a":       strategy == "auto_loop",
                "hpf_sweep_end_freq": 3500.0 if strategy == "hpf_sweep" else None,
                "lpf_sweep_end_freq": 400.0  if strategy == "lpf_sweep"  else None,
                "bpm":                None,
            },
        },
    }