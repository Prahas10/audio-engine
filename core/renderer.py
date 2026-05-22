"""
core/renderer.py

Changes in this version:
  - _render_transition_segment: drift correction NOW APPLIED (was diagnostic-only)
    Correction capped at ±1 bar to prevent wild sample shifts from bad drift readings
  - _safe_strategy: bass_swap and drop_mix banned from continuous mode (enforced)
  - Per-segment RMS normalisation retained from previous version
  - All other logic unchanged
"""

import os
import uuid
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import librosa
from fastapi import HTTPException

from core.analysis import (
    TARGET_SR,
    safe_bpm,
    estimate_key,
    rms_level,
    camelot_compatible,
    verify_beat_alignment,
    sharpen_transients,
    apply_entry_ramp,
    get_energy_curve,
)
from core.transitions import (
    pad_or_trim,
    phase_align,
    snap_to_phrase_boundary,
    auto_loop_track_a_segment,
)
from core.strategy_router import apply_transition_strategy, get_strategy_mix_duration
from core.planner import plan_transition_logic, find_best_sync_point
from core.library import load_library_metadata
from core.queue_state import load_queue_state
from models.schemas import CAMELOT_MAP, FXParameters

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_RMS    = 0.08
PEAK_CEILING  = 0.93

_CONTINUOUS_SET_BANNED   = {"bass_swap", "drop_mix"}
_CONTINUOUS_SET_FALLBACK = {"bass_swap": "energy_blend", "drop_mix": "phrase_mix"}

# Max drift correction applied inside _render_transition_segment.
# Anything larger means the sync itself is wrong — we cap and log.
MAX_DRIFT_CORRECTION_MS = 200.0


# ---------------------------------------------------------------------------
# Loudness helpers
# ---------------------------------------------------------------------------

def _rms_normalise(y: np.ndarray, target_rms: float = TARGET_RMS,
                   max_gain_db: float = 12.0):
    if len(y) == 0:
        return y, 1.0
    current_rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    if current_rms < 1e-9:
        return y, 1.0
    gain = target_rms / current_rms
    max_gain_linear = 10 ** (max_gain_db / 20.0)
    gain = float(np.clip(gain, 1.0 / max_gain_linear, max_gain_linear))
    return y * gain, gain


def _soft_limit(y, threshold=0.95):
    if len(y) == 0:
        return y
    y = y.copy()
    over = np.abs(y) > threshold
    if np.any(over):
        signs  = np.sign(y)
        excess = np.abs(y[over]) - threshold
        y[over] = signs[over] * (
            threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold))
        )
    return y


def _derive_energy_score(y, sr) -> float:
    _, energy_values, _ = get_energy_curve(y, sr)
    return float(np.clip(np.mean(energy_values), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Timeline-safe append
# ---------------------------------------------------------------------------

def _append_part(parts, part, fade_ms=35.0):
    if part is None or len(part) == 0:
        return
    if not parts:
        parts.append(part)
        return
    prev     = parts[-1]
    fade_len = min(int((fade_ms / 1000.0) * TARGET_SR), len(prev), len(part))
    if fade_len < 2:
        parts.append(part)
        return
    t        = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.cos(t * np.pi / 2.0)
    fade_in  = np.sin(t * np.pi / 2.0)
    joined   = prev[-fade_len:] * fade_out + part[:fade_len] * fade_in
    parts[-1] = np.concatenate([prev[:-fade_len], joined, part[fade_len:]])


# ---------------------------------------------------------------------------
# Stretch helpers
# ---------------------------------------------------------------------------

def _compute_stretch_rate(bpm_from, bpm_to):
    if bpm_from <= 0 or bpm_to <= 0:
        return 1.0
    rate = bpm_from / bpm_to
    if 0.80 <= rate <= 1.25:           return float(rate)
    if 0.80 <= rate / 2.0 <= 1.25:    return float(rate / 2.0)
    if 0.80 <= rate * 2.0 <= 1.25:    return float(rate * 2.0)
    raise HTTPException(
        status_code=400,
        detail=f"BPM gap too large: from={bpm_from:.2f}, to={bpm_to:.2f}, rate={rate:.4f}",
    )


def _stretch_audio(y, rate):
    if abs(rate - 1.0) <= 0.01:
        return y
    y_stretched = pyrb.time_stretch(y, TARGET_SR, rate)
    return sharpen_transients(y_stretched, TARGET_SR, strength=0.35)


# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------

def _map_original_time_to_current_sample(original_time, current_original_offset, current_rate):
    local_original_seconds = float(original_time) - float(current_original_offset)
    local_output_seconds   = local_original_seconds / max(float(current_rate), 1e-9)
    return int(local_output_seconds * TARGET_SR)


def _map_original_time_to_stretched_sample(original_time, stretch_rate):
    return int(float(original_time) / max(float(stretch_rate), 1e-9) * TARGET_SR)


def _current_sample_to_original_time(sample, current_original_offset, current_rate):
    return float(current_original_offset + (sample / TARGET_SR) * current_rate)


# ---------------------------------------------------------------------------
# Strategy guard
# ---------------------------------------------------------------------------

def _safe_strategy(plan) -> str:
    strategy     = plan.get("recommended_strategy", "energy_blend")
    harmonic     = bool(plan.get("harmonic_compatible", False))
    mix_duration = float(plan.get("mix_duration", 30.0))

    if strategy in _CONTINUOUS_SET_BANNED:
        replacement = _CONTINUOUS_SET_FALLBACK.get(strategy, "energy_blend")
        print(f"Continuous-set: {strategy} → {replacement} (banned in set mode)")
        return replacement

    if strategy == "harmonic_mix" and not harmonic:
        print("Continuous-set: harmonic_mix → energy_blend (tracks not harmonically compatible)")
        return "energy_blend"

    if strategy == "long_eq_blend" and mix_duration < 24:
        print("Continuous-set: long_eq_blend → energy_blend (mix duration too short)")
        return "energy_blend"

    return strategy


# ---------------------------------------------------------------------------
# Gain helpers (single-clip renderer)
# ---------------------------------------------------------------------------

def _measure_lufs_simple(y, sr):
    block_len = int(0.4 * sr)
    if len(y) < block_len:
        return rms_level(y)
    rms_blocks = []
    for i in range(0, len(y) - block_len, block_len // 2):
        block = y[i:i + block_len]
        rms_blocks.append(float(np.sqrt(np.mean(block ** 2) + 1e-9)))
    return float(np.mean(rms_blocks)) if rms_blocks else rms_level(y)


def _match_gain_entry(segment_b, segment_a, sr, bpm, max_gain_db=6.0):
    beat_samples = int((60.0 / bpm) * sr)
    check_len    = min(4 * beat_samples, len(segment_a), len(segment_b))
    ref_loud     = _measure_lufs_simple(segment_a[:check_len] if check_len >= 1024 else segment_a, sr)
    b_loud       = _measure_lufs_simple(segment_b[:check_len] if check_len >= 1024 else segment_b, sr)
    if b_loud < 1e-9:
        return segment_b, 1.0
    gain     = ref_loud / b_loud
    max_gain = 10 ** (max_gain_db / 20.0)
    gain     = float(np.clip(gain, 1.0 / max_gain, max_gain))
    return segment_b * gain, gain


def _apply_gain_automation(segment_b, mix_samples, sr, bpm):
    ramp_samples = min(int((60.0 / bpm) * sr * 2), mix_samples // 4)
    segment_b    = segment_b.copy()
    segment_b[:ramp_samples] *= np.linspace(0.0, 1.0, ramp_samples) ** 0.5
    return segment_b


# ---------------------------------------------------------------------------
# Transition segment renderer — DRIFT CORRECTION NOW APPLIED
# ---------------------------------------------------------------------------

def _render_transition_segment(
    y_a, y_b,
    start_sample_a, start_sample_b,
    mix_duration, bpm_a, strategy,
    target_rms: float = TARGET_RMS,
    sync_accuracy: float = 1.0,
):
    """
    Renders a single crossfade segment for the continuous set.

    Drift correction: after slicing both segments, verify_beat_alignment
    measures the remaining offset. If drift is within MAX_DRIFT_CORRECTION_MS
    we apply the shift directly. If drift exceeds this cap we log a warning
    (indicating the entry point itself was wrong, not just a sample offset).

    Both segments are RMS-normalised to target_rms before the transition
    function sees them so all strategies operate on matched loudness.
    """
    mix_samples    = int(float(mix_duration) * TARGET_SR)
    start_sample_a = int(np.clip(start_sample_a, 0, max(0, len(y_a) - 1)))
    start_sample_b = int(np.clip(start_sample_b, 0, max(0, len(y_b) - 1)))

    remaining_a = len(y_a) - start_sample_a
    if remaining_a < mix_samples:
        segment_a = auto_loop_track_a_segment(
            y_a=y_a, start_sample_a=start_sample_a,
            mix_samples=mix_samples, sr=TARGET_SR,
        )
    else:
        segment_a = pad_or_trim(y_a[start_sample_a:start_sample_a + mix_samples], mix_samples)

    segment_b = pad_or_trim(y_b[start_sample_b:start_sample_b + mix_samples], mix_samples)

    # RMS-normalise both to the same target before mixing
    segment_a, gain_a = _rms_normalise(segment_a, target_rms=target_rms)
    segment_b, gain_b = _rms_normalise(segment_b, target_rms=target_rms)

    # Phase alignment (polarity check)
    shift, polarity_flip = phase_align(segment_a, segment_b, TARGET_SR)
    if polarity_flip:
        segment_b = -segment_b

    # Beat drift measurement and correction
    drift_ms, is_aligned = verify_beat_alignment(segment_a, segment_b, TARGET_SR, bpm_a)
    drift_correction_applied = False

    # Strategies that require tight beat alignment (< 50ms)
    TIMING_SENSITIVE = {"bass_swap", "drop_mix", "loop_roll", "percussion_blend"}

    # Degrade timing-sensitive strategies when sync confidence is low.
    # sync_accuracy < 0.55 means the correlator found no reliable lock —
    # drift_ms measurement is also unreliable in this case (spectral mismatch).
    # energy_blend handles loose timing gracefully; others don't.
    if sync_accuracy < 0.55 and strategy in TIMING_SENSITIVE:
        print(f"Low sync confidence ({sync_accuracy:.3f}) + timing-sensitive strategy "
              f"({strategy}) → energy_blend.")
        strategy = "energy_blend"

    if not is_aligned:
        if abs(drift_ms) <= MAX_DRIFT_CORRECTION_MS:
            correction = int((drift_ms / 1000.0) * TARGET_SR)
            print(f"Drift correction applied: {drift_ms:.1f}ms ({correction} samples)")
            if correction > 0:
                segment_b = np.pad(segment_b, (correction, 0))[:mix_samples]
            elif correction < 0:
                segment_b = np.pad(segment_b[abs(correction):], (0, abs(correction)))[:mix_samples]
            drift_correction_applied = True
        else:
            print(
                f"WARNING: drift {drift_ms:.0f}ms > cap {MAX_DRIFT_CORRECTION_MS:.0f}ms "
                f"— entry point timing uncertain. No sample shift applied."
            )
            if strategy in TIMING_SENSITIVE:
                print(f"  Strategy {strategy} requires tight sync — degrading to energy_blend.")
                strategy = "energy_blend"

    segment_b = apply_entry_ramp(segment_b, ramp_ms=10.0, sr=TARGET_SR)

    energy_score  = _derive_energy_score(segment_a, TARGET_SR)
    fx_parameters = FXParameters(bpm=bpm_a, energy_score=energy_score)

    mixed = apply_transition_strategy(
        segment_a=segment_a, segment_b=segment_b,
        sr=TARGET_SR, transition_strategy=strategy,
        fx_parameters=fx_parameters,
    )
    mixed = _soft_limit(mixed, threshold=0.95)
    mixed = pad_or_trim(mixed, mix_samples)

    return mixed, {
        "phase_shift_samples_detected": int(shift),
        "phase_shift_ms_detected":      round(float(shift / TARGET_SR * 1000.0), 2),
        "phase_shift_applied":          bool(polarity_flip),
        "beat_alignment_drift_ms":      round(float(drift_ms), 2),
        "drift_correction_applied":     drift_correction_applied,
        "is_aligned":                   bool(is_aligned),
        "segment_a_gain":               round(float(gain_a), 3),
        "segment_b_gain":               round(float(gain_b), 3),
        "energy_score":                 round(float(energy_score), 3),
    }


# ---------------------------------------------------------------------------
# Continuous set renderer
# ---------------------------------------------------------------------------

def render_continuous_set_from_ordered_tracks(
    ordered_tracks,
    preferred_mix_duration=None,
    output_path="outputs/final_set.wav",
    target_rms: float = TARGET_RMS,
):
    if len(ordered_tracks) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tracks.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    loaded = []
    for track in ordered_tracks:
        y   = _load_audio(track["path"])
        bpm = _get_bpm(track, y)
        loaded.append({"metadata": track, "path": track["path"], "audio": y, "bpm": bpm})

    full_parts  = []
    transitions = []
    timeline    = []

    current_audio           = loaded[0]["audio"]
    current_track           = loaded[0]
    current_bpm             = loaded[0]["bpm"]
    current_original_offset = 0.0
    current_rate            = 1.0
    set_cursor_seconds      = 0.0

    for idx in range(len(loaded) - 1):
        next_track = loaded[idx + 1]
        print(
            f"\nRendering transition {idx + 1}: "
            f"{current_track['metadata'].get('filename')} → "
            f"{next_track['metadata'].get('filename')}"
        )

        plan         = plan_transition_logic(
            track_a_path=current_track["path"],
            track_b_path=next_track["path"],
            preferred_mix_duration=preferred_mix_duration,
        )
        plan_t_orig  = float(plan["recommended_transition_start_time"])
        plan_b_orig  = float(plan["recommended_track_b_entry_time"])
        mix_duration = float(plan["mix_duration"])
        strategy     = _safe_strategy(plan)
        mix_samples  = int(mix_duration * TARGET_SR)

        start_sample_a = _map_original_time_to_current_sample(
            plan_t_orig, current_original_offset, current_rate)
        start_sample_a = int(np.clip(start_sample_a, 0, max(0, len(current_audio) - mix_samples)))

        actual_a_orig = _current_sample_to_original_time(
            start_sample_a, current_original_offset, current_rate)

        next_rate            = _compute_stretch_rate(next_track["bpm"], current_bpm)
        next_audio_stretched = _stretch_audio(next_track["audio"], next_rate)

        start_sample_b = _map_original_time_to_stretched_sample(plan_b_orig, next_rate)
        start_sample_b = int(np.clip(
            start_sample_b, 0, max(0, len(next_audio_stretched) - mix_samples)))

        actual_b_orig = (start_sample_b / TARGET_SR) * next_rate

        # 1. Track body — RMS-normalised
        prefix = current_audio[:start_sample_a]
        if len(prefix) > 0:
            prefix_norm, prefix_gain = _rms_normalise(prefix, target_rms=target_rms)
            prefix_norm = _soft_limit(prefix_norm)
            _append_part(full_parts, prefix_norm)
            body_start = set_cursor_seconds
            body_end   = set_cursor_seconds + len(prefix) / TARGET_SR
            timeline.append({
                "type":                  "track_body",
                "track_id":              current_track["metadata"].get("track_id"),
                "filename":              current_track["metadata"].get("filename"),
                "set_start":             round(body_start, 3),
                "set_end":               round(body_end, 3),
                "source_original_start": round(current_original_offset, 3),
                "source_original_end":   round(actual_a_orig, 3),
                "rate":                  round(float(current_rate), 5),
                "loudness_gain":         round(float(prefix_gain), 3),
            })
            set_cursor_seconds = body_end

        # 2. Transition
        plan_sync_accuracy = float(plan.get("sync_accuracy", 1.0) or 1.0)
        mixed_segment, render_meta = _render_transition_segment(
            y_a=current_audio, y_b=next_audio_stretched,
            start_sample_a=start_sample_a, start_sample_b=start_sample_b,
            mix_duration=mix_duration, bpm_a=current_bpm, strategy=strategy,
            target_rms=target_rms, sync_accuracy=plan_sync_accuracy,
        )
        _append_part(full_parts, mixed_segment)

        transition_set_start = set_cursor_seconds
        transition_set_end   = set_cursor_seconds + mix_duration

        transition_record = {
            "index": idx + 1,
            "from_track": {
                "track_id": current_track["metadata"].get("track_id"),
                "filename": current_track["metadata"].get("filename"),
                "bpm":      current_track["metadata"].get("bpm"),
                "key":      current_track["metadata"].get("key"),
                "camelot":  current_track["metadata"].get("camelot"),
            },
            "to_track": {
                "track_id": next_track["metadata"].get("track_id"),
                "filename": next_track["metadata"].get("filename"),
                "bpm":      next_track["metadata"].get("bpm"),
                "key":      next_track["metadata"].get("key"),
                "camelot":  next_track["metadata"].get("camelot"),
            },
            "strategy":              strategy,
            "planner_strategy":      plan.get("recommended_strategy"),
            "set_start":             round(transition_set_start, 3),
            "set_end":               round(transition_set_end, 3),
            "mix_duration":          round(mix_duration, 3),
            "track_a_planned_original_time":       round(plan_t_orig, 3),
            "track_a_actual_original_time":        round(actual_a_orig, 3),
            "track_a_local_sample":                int(start_sample_a),
            "track_b_planned_original_entry_time": round(plan_b_orig, 3),
            "track_b_actual_original_entry_time":  round(actual_b_orig, 3),
            "track_b_stretched_entry_sample":      int(start_sample_b),
            "stretch_rate": round(float(next_rate), 5),
            "plan":         plan,
            "render":       render_meta,
        }
        transitions.append(transition_record)
        timeline.append({"type": "transition", **transition_record})
        set_cursor_seconds = transition_set_end

        # 3. Carry forward stretched Track B suffix
        next_resume = int(np.clip(
            start_sample_b + mix_samples, 0, len(next_audio_stretched)))
        current_audio           = next_audio_stretched[next_resume:]
        current_track           = next_track
        current_bpm             = current_bpm
        current_original_offset = actual_b_orig + (mix_duration * next_rate)
        current_rate            = next_rate

    # Final suffix
    if len(current_audio) > 0:
        suffix_norm, suffix_gain = _rms_normalise(current_audio, target_rms=target_rms)
        suffix_norm = _soft_limit(suffix_norm)
        _append_part(full_parts, suffix_norm)
        final_start = set_cursor_seconds
        final_end   = set_cursor_seconds + len(current_audio) / TARGET_SR
        timeline.append({
            "type":                  "track_body",
            "track_id":              current_track["metadata"].get("track_id"),
            "filename":              current_track["metadata"].get("filename"),
            "set_start":             round(final_start, 3),
            "set_end":               round(final_end, 3),
            "source_original_start": round(current_original_offset, 3),
            "source_original_end":   round(
                current_original_offset + (len(current_audio) / TARGET_SR) * current_rate, 3),
            "rate":          round(float(current_rate), 5),
            "loudness_gain": round(float(suffix_gain), 3),
        })
        set_cursor_seconds = final_end

    if not full_parts:
        raise HTTPException(status_code=400, detail="Final set produced no audio.")

    final_audio = np.concatenate(full_parts)
    final_audio = _soft_limit(final_audio, threshold=PEAK_CEILING)
    peak = float(np.max(np.abs(final_audio)))
    if peak > PEAK_CEILING:
        final_audio = final_audio / peak * PEAK_CEILING

    sf.write(output_path, final_audio, TARGET_SR)

    return {
        "status":           "success",
        "output_path":      os.path.abspath(output_path),
        "duration_seconds": round(float(len(final_audio) / TARGET_SR), 3),
        "track_count":      len(ordered_tracks),
        "transition_count": len(transitions),
        "transitions":      transitions,
        "timeline":         timeline,
    }


# ---------------------------------------------------------------------------
# Helpers used by both renderers
# ---------------------------------------------------------------------------

def _load_audio(path):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Track not found: {path}")
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return y


def _get_bpm(track_metadata, y):
    bpm = float(track_metadata.get("bpm") or 0.0)
    if bpm <= 0:
        bpm, _ = safe_bpm(y, TARGET_SR)
    return float(bpm)


# ---------------------------------------------------------------------------
# Single-transition renderer
# ---------------------------------------------------------------------------

def render_dj_transition(
    track_a_path, track_b_path,
    transition_start_time, mix_duration, output_dir,
    transition_strategy="long_eq_blend",
    fx_parameters=None,
    track_b_entry_time=None,
    planner_metadata=None,
):
    if not os.path.exists(track_a_path):
        raise HTTPException(status_code=400, detail=f"Track A not found: {track_a_path}")
    if not os.path.exists(track_b_path):
        raise HTTPException(status_code=400, detail=f"Track B not found: {track_b_path}")

    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Starting Headless Audio Engine ---")

    y_a, _ = librosa.load(track_a_path, sr=TARGET_SR, mono=True)
    y_b, _ = librosa.load(track_b_path, sr=TARGET_SR, mono=True)

    if planner_metadata:
        bpm_a   = float(planner_metadata["track_a"]["bpm"])
        bpm_b   = float(planner_metadata["track_b"]["bpm"])
        beats_a = np.asarray(librosa.time_to_samples(
            planner_metadata["track_a"].get("beats", []), sr=TARGET_SR), dtype=int)
        beats_b = np.asarray(librosa.time_to_samples(
            planner_metadata["track_b"].get("beats", []), sr=TARGET_SR), dtype=int)
        if len(beats_a) == 0: bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)
        if len(beats_b) == 0: bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)
    else:
        bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)
        bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)

    stretch_ratio = bpm_b / bpm_a
    if not (0.80 <= stretch_ratio <= 1.25):
        if 0.80 <= stretch_ratio / 2.0 <= 1.25:   stretch_ratio = stretch_ratio / 2.0
        elif 0.80 <= stretch_ratio * 2.0 <= 1.25:  stretch_ratio = stretch_ratio * 2.0
        else:
            raise HTTPException(status_code=400,
                detail=f"BPM gap too large: A={bpm_a:.2f}, B={bpm_b:.2f}")

    if abs(stretch_ratio - 1.0) > 0.01:
        y_b = pyrb.time_stretch(y_b, TARGET_SR, stretch_ratio)
        y_b = sharpen_transients(y_b, TARGET_SR, strength=0.35)
        _, beats_b = safe_bpm(y_b, TARGET_SR)

    bpm_b_after = bpm_a

    phrase_beats = 32 if transition_strategy == "phrase_mix" else 4
    snapped_start_time, start_sample_a = snap_to_phrase_boundary(
        beats=beats_a, requested_time=transition_start_time,
        sr=TARGET_SR, phrase_beats=phrase_beats)

    if planner_metadata:
        key_a  = planner_metadata["track_a"]["key"].split()[0]
        mode_a = planner_metadata["track_a"]["key"].split()[1]
        key_b  = planner_metadata["track_b"]["key"].split()[0]
        mode_b = planner_metadata["track_b"]["key"].split()[1]
        camelot_a   = planner_metadata["track_a"].get("camelot")
        camelot_b   = planner_metadata["track_b"].get("camelot")
        harmonic_ok = bool(planner_metadata.get(
            "harmonic_compatible", camelot_compatible(camelot_a, camelot_b)))
    else:
        key_a, mode_a, _ = estimate_key(y_a, TARGET_SR)
        key_b, mode_b, _ = estimate_key(y_b, TARGET_SR)
        camelot_a   = CAMELOT_MAP.get((key_a, mode_a))
        camelot_b   = CAMELOT_MAP.get((key_b, mode_b))
        harmonic_ok = camelot_compatible(camelot_a, camelot_b)

    if transition_strategy == "harmonic_mix" and not harmonic_ok:
        raise HTTPException(status_code=400,
            detail=f"Tracks not harmonically compatible: {camelot_a} vs {camelot_b}")

    if mix_duration is None:
        mix_duration = get_strategy_mix_duration(transition_strategy, bpm_a)

    mix_samples = int(mix_duration * TARGET_SR)

    remaining_a = len(y_a) - start_sample_a
    use_loop = remaining_a < mix_samples or (
        fx_parameters is not None and getattr(fx_parameters, "loop_track_a", False))
    if use_loop:
        segment_a = auto_loop_track_a_segment(
            y_a=y_a, start_sample_a=start_sample_a,
            mix_samples=mix_samples, sr=TARGET_SR)
    else:
        segment_a = pad_or_trim(y_a[start_sample_a:start_sample_a + mix_samples], mix_samples)

    if track_b_entry_time is not None:
        start_sample_b = int(float(track_b_entry_time) * TARGET_SR)
        sync_accuracy  = None
    else:
        start_sample_b, sync_accuracy = find_best_sync_point(
            track_a_beats=beats_a, track_b_beats=beats_b,
            transition_start_sample=start_sample_a, mix_samples=mix_samples,
            offset_samples=1200, track_b_total_samples=len(y_b),
            min_b_entry_percent=0.0, max_b_entry_percent=0.35)
        track_b_entry_time = start_sample_b / TARGET_SR

    segment_b = pad_or_trim(y_b[start_sample_b:start_sample_b + mix_samples], mix_samples)

    # RMS-normalise both
    segment_a, gain_a = _rms_normalise(segment_a, target_rms=TARGET_RMS)
    segment_b, gain_b = _rms_normalise(segment_b, target_rms=TARGET_RMS)

    shift, polarity_flip = phase_align(segment_a, segment_b, TARGET_SR)
    if polarity_flip:
        segment_b = -segment_b
    if shift > 0:
        segment_b = np.pad(segment_b, (shift, 0))[:mix_samples]
    elif shift < 0:
        segment_b = np.pad(segment_b[abs(shift):], (0, abs(shift)))[:mix_samples]

    drift_ms, is_aligned = verify_beat_alignment(segment_a, segment_b, TARGET_SR, bpm_a)
    drift_correction_applied = False
    if not is_aligned:
        if abs(drift_ms) <= MAX_DRIFT_CORRECTION_MS:
            correction = int((drift_ms / 1000.0) * TARGET_SR)
            print(f"Drift correction: {drift_ms:.1f}ms → shifting {correction} samples")
            if correction > 0:
                segment_b = np.pad(segment_b, (correction, 0))[:mix_samples]
            elif correction < 0:
                segment_b = np.pad(segment_b[abs(correction):], (0, abs(correction)))[:mix_samples]
            drift_correction_applied = True

    segment_b = _apply_gain_automation(segment_b, mix_samples, TARGET_SR, bpm_a)
    segment_b = apply_entry_ramp(segment_b, ramp_ms=15.0, sr=TARGET_SR)

    energy_score = _derive_energy_score(segment_a, TARGET_SR)
    if fx_parameters is None:
        fx_parameters = FXParameters()
    try:
        fx_parameters = fx_parameters.model_copy(update={"bpm": bpm_a, "energy_score": energy_score})
    except Exception:
        fx_parameters = fx_parameters.copy(update={"bpm": bpm_a, "energy_score": energy_score})

    mixed_segment = apply_transition_strategy(
        segment_a=segment_a, segment_b=segment_b,
        sr=TARGET_SR, transition_strategy=transition_strategy,
        fx_parameters=fx_parameters)

    mixed_segment = _soft_limit(mixed_segment, threshold=0.95)
    peak = np.max(np.abs(mixed_segment))
    if peak > 1e-9:
        mixed_segment = mixed_segment / peak * 0.95
    mixed_segment = pad_or_trim(mixed_segment, mix_samples)

    output_id              = uuid.uuid4().hex[:8]
    transition_output_path = os.path.normpath(os.path.join(output_dir, f"transition_{output_id}.wav"))
    suffix_output_path     = os.path.normpath(os.path.join(output_dir, f"track_b_suffix_{output_id}.wav"))
    sf.write(transition_output_path, mixed_segment, TARGET_SR)
    b_suffix_start = start_sample_b + mix_samples
    sf.write(suffix_output_path, y_b[b_suffix_start:], TARGET_SR)

    return {
        "status":              "success",
        "audio_clip_url":      transition_output_path,
        "transition_file":     transition_output_path,
        "track_b_suffix_file": suffix_output_path,
        "duration_seconds":              mix_duration,
        "transition_strategy":           transition_strategy,
        "snapped_transition_start_time": round(snapped_start_time, 3),
        "track_a_bpm":            round(bpm_a, 2),
        "track_b_original_bpm":   round(bpm_b, 2),
        "track_b_stretched_bpm":  round(bpm_b_after, 2),
        "segment_a_gain":         round(float(gain_a), 3),
        "segment_b_gain":         round(float(gain_b), 3),
        "sync_accuracy":          None if sync_accuracy is None else round(float(sync_accuracy), 3),
        "energy_score":           round(float(energy_score), 3),
        "track_b_entry_time":     round(float(track_b_entry_time), 3),
        "track_b_entry_sample":   int(start_sample_b),
        "track_b_suffix_start_time": round(float(b_suffix_start / TARGET_SR), 3),
        "track_b_suffix_duration":   round(float(len(y_b[b_suffix_start:]) / TARGET_SR), 3),
        "phase_shift_samples":     int(shift),
        "phase_shift_ms":          round(shift / TARGET_SR * 1000.0, 2),
        "beat_alignment_drift_ms": round(drift_ms, 2),
        "drift_correction_applied": drift_correction_applied,
        "track_a_key":         f"{key_a} {mode_a}",
        "track_b_key":         f"{key_b} {mode_b}",
        "track_a_camelot":     camelot_a,
        "track_b_camelot":     camelot_b,
        "harmonic_compatible": harmonic_ok,
    }


# ---------------------------------------------------------------------------
# Queue wrapper
# ---------------------------------------------------------------------------

def render_queue_order_set(
    queue_path, library_path,
    setlist_path=None,
    preferred_mix_duration=None,
    output_dir="outputs",
    final_output_path="outputs/final_set.wav",
):
    library = load_library_metadata(library_path)
    queue   = load_queue_state(queue_path)
    ordered_ids = []
    if queue.get("current_track_id"):
        ordered_ids.append(queue["current_track_id"])
    ordered_ids.extend(queue.get("upcoming_track_ids", []))
    if len(ordered_ids) < 2:
        raise HTTPException(status_code=400, detail="Queue needs at least 2 tracks.")
    ordered_tracks = []
    for tid in ordered_ids:
        if tid not in library:
            raise HTTPException(status_code=404, detail=f"Track not found: {tid}")
        ordered_tracks.append(library[tid])
    result = render_continuous_set_from_ordered_tracks(
        ordered_tracks=ordered_tracks,
        preferred_mix_duration=preferred_mix_duration,
        output_path=final_output_path,
    )
    return {
        "status":     "success",
        "mode":       "queue_order_continuous_timestretch",
        "track_count": len(ordered_tracks),
        "ordered_tracks": [
            {"position": i+1, "track_id": t["track_id"], "filename": t["filename"],
             "bpm": t.get("bpm"), "key": t.get("key"), "camelot": t.get("camelot")}
            for i, t in enumerate(ordered_tracks)
        ],
        "transition_count": result["transition_count"],
        "transitions":      result["transitions"],
        "timeline":         result["timeline"],
        "final_mix": {
            "status":           "success",
            "output_path":      result["output_path"],
            "duration_seconds": result["duration_seconds"],
            "transition_count": result["transition_count"],
        },
    }