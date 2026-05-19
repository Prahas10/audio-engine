import os
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as pyrb
from fastapi import HTTPException

from core.analysis import (
    TARGET_SR,
    safe_bpm,
    sharpen_transients,
    verify_beat_alignment,
    apply_entry_ramp,
)
from core.transitions import (
    pad_or_trim,
    phase_align,
    auto_loop_track_a_segment,
)
from core.strategy_router import apply_transition_strategy
from core.planner import plan_transition_logic
from core.library import load_library_metadata
from core.queue_state import load_queue_state
from models.schemas import FXParameters


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _soft_limit(y, threshold=0.95):
    if len(y) == 0:
        return y

    y = y.copy()
    over = np.abs(y) > threshold

    if np.any(over):
        signs = np.sign(y)
        excess = np.abs(y[over]) - threshold
        y[over] = signs[over] * (
            threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold))
        )

    return y


def _normalise(y, peak_target=0.95):
    peak = np.max(np.abs(y)) if len(y) else 0.0

    if peak > 1e-9:
        return y / peak * peak_target

    return y


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
# Timeline-safe append
# ---------------------------------------------------------------------------

def _append_part(parts, part, fade_ms=35.0):
    """
    Appends audio with a tiny equal-power join.

    This is NOT the musical transition.
    It only prevents micro-clicks/gaps between array chunks.
    """
    if part is None or len(part) == 0:
        return

    if not parts:
        parts.append(part)
        return

    prev = parts[-1]

    fade_len = min(
        int((fade_ms / 1000.0) * TARGET_SR),
        len(prev),
        len(part),
    )

    if fade_len < 2:
        parts.append(part)
        return

    t = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.cos(t * np.pi / 2.0)
    fade_in = np.sin(t * np.pi / 2.0)

    joined = prev[-fade_len:] * fade_out + part[:fade_len] * fade_in

    parts[-1] = np.concatenate([
        prev[:-fade_len],
        joined,
        part[fade_len:],
    ])


# ---------------------------------------------------------------------------
# Time-stretch helpers
# ---------------------------------------------------------------------------

def _compute_stretch_rate(bpm_from, bpm_to):
    """
    pyrubberband rate:
        rate > 1 = faster / shorter
        rate < 1 = slower / longer

    To move Track B from bpm_from to bpm_to:
        rate = bpm_from / bpm_to
    """
    if bpm_from <= 0 or bpm_to <= 0:
        return 1.0

    rate = bpm_from / bpm_to

    if 0.80 <= rate <= 1.25:
        return float(rate)

    if 0.80 <= rate / 2.0 <= 1.25:
        return float(rate / 2.0)

    if 0.80 <= rate * 2.0 <= 1.25:
        return float(rate * 2.0)

    raise HTTPException(
        status_code=400,
        detail=f"BPM gap too large: from={bpm_from:.2f}, to={bpm_to:.2f}, rate={rate:.4f}",
    )


def _stretch_audio(y, rate):
    if abs(rate - 1.0) <= 0.01:
        return y

    y_stretched = pyrb.time_stretch(y, TARGET_SR, rate)
    y_stretched = sharpen_transients(y_stretched, TARGET_SR, strength=0.35)

    return y_stretched


# ---------------------------------------------------------------------------
# Time mapping helpers
# ---------------------------------------------------------------------------

def _map_original_time_to_current_sample(original_time, current_original_offset, current_rate):
    """
    current_audio may be a stretched suffix.

    original_time:
        timestamp in the original full source track.

    current_original_offset:
        original source timestamp represented by current_audio[0].

    current_rate:
        rate used to stretch current_audio from original source.
        output_seconds = original_seconds / current_rate.
    """
    local_original_seconds = float(original_time) - float(current_original_offset)
    local_output_seconds = local_original_seconds / max(float(current_rate), 1e-9)

    return int(local_output_seconds * TARGET_SR)


def _map_original_time_to_stretched_sample(original_time, stretch_rate):
    """
    Track B entry time from planner is in original Track B seconds.

    If Track B is stretched:
        stretched_seconds = original_seconds / stretch_rate
    """
    stretched_seconds = float(original_time) / max(float(stretch_rate), 1e-9)
    return int(stretched_seconds * TARGET_SR)


def _current_sample_to_original_time(sample, current_original_offset, current_rate):
    output_seconds = sample / TARGET_SR
    original_seconds = output_seconds * current_rate
    return float(current_original_offset + original_seconds)


# ---------------------------------------------------------------------------
# Strategy guard
# ---------------------------------------------------------------------------

def _safe_strategy(plan):
    strategy = plan.get("recommended_strategy", "bass_swap")
    harmonic = bool(plan.get("harmonic_compatible", False))
    mix_duration = float(plan.get("mix_duration", 30.0))

    if strategy == "harmonic_mix" and not harmonic:
        return "energy_blend"

    if strategy == "long_eq_blend" and mix_duration < 24:
        return "bass_swap"

    if strategy in {"ambient_transition", "techno_filter_drive"}:
        return "energy_blend"

    return strategy


# ---------------------------------------------------------------------------
# Gain matching for continuous mode
# ---------------------------------------------------------------------------

def _match_gain_for_continuous_set(segment_b, segment_a, sr, bpm, max_gain_db=4.0):
    beat_samples = int((60.0 / bpm) * sr)
    check_len = min(4 * beat_samples, len(segment_a), len(segment_b))

    if check_len < 1024:
        return segment_b, 1.0

    a_rms = np.sqrt(np.mean(segment_a[:check_len] ** 2) + 1e-9)
    b_rms = np.sqrt(np.mean(segment_b[:check_len] ** 2) + 1e-9)

    if b_rms <= 1e-9:
        return segment_b, 1.0

    gain = a_rms / b_rms
    max_gain = 10 ** (max_gain_db / 20.0)
    gain = float(np.clip(gain, 1.0 / max_gain, max_gain))

    return segment_b * gain, gain


# ---------------------------------------------------------------------------
# Transition segment renderer
# ---------------------------------------------------------------------------

def _render_transition_segment(
    y_a,
    y_b,
    start_sample_a,
    start_sample_b,
    mix_duration,
    bpm_a,
    strategy,
):
    """
    Continuous-set transition renderer.

    IMPORTANT:
    This does NOT physically shift/pad Track B.
    If we shift only the transition segment and then continue with the unshifted
    Track B suffix, the boundary after the transition will jump/gap.

    Beat sync should come from planner entry-time selection.
    """
    mix_samples = int(float(mix_duration) * TARGET_SR)

    start_sample_a = int(np.clip(start_sample_a, 0, max(0, len(y_a) - 1)))
    start_sample_b = int(np.clip(start_sample_b, 0, max(0, len(y_b) - 1)))

    remaining_a = len(y_a) - start_sample_a

    if remaining_a < mix_samples:
        segment_a = auto_loop_track_a_segment(
            y_a=y_a,
            start_sample_a=start_sample_a,
            mix_samples=mix_samples,
            sr=TARGET_SR,
        )
    else:
        segment_a = pad_or_trim(
            y_a[start_sample_a:start_sample_a + mix_samples],
            mix_samples,
        )

    segment_b = pad_or_trim(
        y_b[start_sample_b:start_sample_b + mix_samples],
        mix_samples,
    )

    # Diagnostic only. Do not apply the shift in continuous mode.
    shift, polarity_flip = phase_align(segment_a, segment_b, TARGET_SR)

    if polarity_flip:
        segment_b = -segment_b

    drift_ms, is_aligned = verify_beat_alignment(
        segment_a,
        segment_b,
        TARGET_SR,
        bpm_a,
    )

    segment_b, gain = _match_gain_for_continuous_set(
        segment_b=segment_b,
        segment_a=segment_a,
        sr=TARGET_SR,
        bpm=bpm_a,
        max_gain_db=4.0,
    )

    segment_b = apply_entry_ramp(segment_b, ramp_ms=10.0, sr=TARGET_SR)

    fx_parameters = FXParameters()

    try:
        fx_parameters = fx_parameters.model_copy(update={"bpm": bpm_a})
    except Exception:
        fx_parameters = fx_parameters.copy(update={"bpm": bpm_a})

    mixed = apply_transition_strategy(
        segment_a=segment_a,
        segment_b=segment_b,
        sr=TARGET_SR,
        transition_strategy=strategy,
        fx_parameters=fx_parameters,
    )

    mixed = _soft_limit(mixed, threshold=0.95)
    mixed = pad_or_trim(mixed, mix_samples)

    return mixed, {
        "phase_shift_samples_detected": int(shift),
        "phase_shift_ms_detected": round(float(shift / TARGET_SR * 1000.0), 2),
        "phase_shift_applied": False,
        "beat_alignment_drift_ms": round(float(drift_ms), 2),
        "drift_correction_applied": False,
        "is_aligned": bool(is_aligned),
        "track_b_gain": round(float(gain), 3),
    }


# ---------------------------------------------------------------------------
# Main continuous renderer
# ---------------------------------------------------------------------------

def render_continuous_set_from_ordered_tracks(
    ordered_tracks,
    preferred_mix_duration=None,
    output_path="outputs/final_set.wav",
):
    """
    Direct continuous set renderer.

    This is the correct mode when time-stretching is allowed.

    It avoids:
    - rendering separate transition clips and assembling later
    - resuming from original Track B after stretching
    - phase-shifting only a transition slice and then resuming unshifted B

    It does:
    - carry forward the exact stretched Track B suffix
    - map planner's original timestamps into the current stretched suffix timeline
    - return full timeline + transition metadata
    """
    if len(ordered_tracks) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 tracks.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    loaded = []

    for track in ordered_tracks:
        y = _load_audio(track["path"])
        bpm = _get_bpm(track, y)

        loaded.append({
            "metadata": track,
            "path": track["path"],
            "audio": y,
            "bpm": bpm,
        })

    full_parts = []
    transitions = []
    timeline = []

    current_audio = loaded[0]["audio"]
    current_track = loaded[0]
    current_bpm = loaded[0]["bpm"]

    # Original timestamp represented by current_audio[0]
    current_original_offset = 0.0

    # Stretch rate of current_audio compared to its original source
    current_rate = 1.0

    set_cursor_seconds = 0.0

    for idx in range(len(loaded) - 1):
        next_track = loaded[idx + 1]

        print(
            f"\nRendering transition {idx + 1}: "
            f"{current_track['metadata'].get('filename')} -> {next_track['metadata'].get('filename')}"
        )

        plan = plan_transition_logic(
            track_a_path=current_track["path"],
            track_b_path=next_track["path"],
            preferred_mix_duration=preferred_mix_duration,
        )

        plan_transition_time_original = float(plan["recommended_transition_start_time"])
        plan_b_entry_time_original = float(plan["recommended_track_b_entry_time"])
        mix_duration = float(plan["mix_duration"])
        strategy = _safe_strategy(plan)

        mix_samples = int(mix_duration * TARGET_SR)

        start_sample_a = _map_original_time_to_current_sample(
            original_time=plan_transition_time_original,
            current_original_offset=current_original_offset,
            current_rate=current_rate,
        )

        # If planner picked a point before the current suffix starts, move transition later.
        if start_sample_a < 0:
            start_sample_a = 0

        # If not enough runway, move transition to the latest possible point.
        if start_sample_a + mix_samples > len(current_audio):
            start_sample_a = max(0, len(current_audio) - mix_samples)

        actual_a_original_time = _current_sample_to_original_time(
            sample=start_sample_a,
            current_original_offset=current_original_offset,
            current_rate=current_rate,
        )

        bpm_b_original = next_track["bpm"]

        next_rate = _compute_stretch_rate(
            bpm_from=bpm_b_original,
            bpm_to=current_bpm,
        )

        next_audio_stretched = _stretch_audio(
            y=next_track["audio"],
            rate=next_rate,
        )

        start_sample_b = _map_original_time_to_stretched_sample(
            original_time=plan_b_entry_time_original,
            stretch_rate=next_rate,
        )

        if start_sample_b < 0:
            start_sample_b = 0

        if start_sample_b + mix_samples > len(next_audio_stretched):
            start_sample_b = max(0, len(next_audio_stretched) - mix_samples)

        actual_b_original_time = (start_sample_b / TARGET_SR) * next_rate

        # 1. Add current track body up to transition.
        prefix = current_audio[:start_sample_a]

        if len(prefix) > 0:
            _append_part(full_parts, prefix)

            body_start = set_cursor_seconds
            body_end = set_cursor_seconds + len(prefix) / TARGET_SR

            timeline.append({
                "type": "track_body",
                "track_id": current_track["metadata"].get("track_id"),
                "filename": current_track["metadata"].get("filename"),
                "set_start": round(body_start, 3),
                "set_end": round(body_end, 3),
                "source_original_start": round(current_original_offset, 3),
                "source_original_end": round(actual_a_original_time, 3),
                "rate": round(float(current_rate), 5),
            })

            set_cursor_seconds = body_end

        # 2. Render transition.
        mixed_segment, render_meta = _render_transition_segment(
            y_a=current_audio,
            y_b=next_audio_stretched,
            start_sample_a=start_sample_a,
            start_sample_b=start_sample_b,
            mix_duration=mix_duration,
            bpm_a=current_bpm,
            strategy=strategy,
        )

        _append_part(full_parts, mixed_segment)

        transition_set_start = set_cursor_seconds
        transition_set_end = set_cursor_seconds + mix_duration

        transition_record = {
            "index": idx + 1,
            "from_track": {
                "track_id": current_track["metadata"].get("track_id"),
                "filename": current_track["metadata"].get("filename"),
                "bpm": current_track["metadata"].get("bpm"),
                "key": current_track["metadata"].get("key"),
                "camelot": current_track["metadata"].get("camelot"),
            },
            "to_track": {
                "track_id": next_track["metadata"].get("track_id"),
                "filename": next_track["metadata"].get("filename"),
                "bpm": next_track["metadata"].get("bpm"),
                "key": next_track["metadata"].get("key"),
                "camelot": next_track["metadata"].get("camelot"),
            },
            "strategy": strategy,
            "planner_strategy": plan.get("recommended_strategy"),
            "set_start": round(transition_set_start, 3),
            "set_end": round(transition_set_end, 3),
            "mix_duration": round(mix_duration, 3),
            "track_a_planned_original_time": round(plan_transition_time_original, 3),
            "track_a_actual_original_time": round(actual_a_original_time, 3),
            "track_a_local_sample": int(start_sample_a),
            "track_b_planned_original_entry_time": round(plan_b_entry_time_original, 3),
            "track_b_actual_original_entry_time": round(actual_b_original_time, 3),
            "track_b_stretched_entry_sample": int(start_sample_b),
            "stretch_rate": round(float(next_rate), 5),
            "plan": plan,
            "render": render_meta,
        }

        transitions.append(transition_record)

        timeline.append({
            "type": "transition",
            **transition_record,
        })

        set_cursor_seconds = transition_set_end

        # 3. Continue from the exact same stretched Track B timeline after transition.
        next_resume_sample = start_sample_b + mix_samples
        next_resume_sample = int(np.clip(next_resume_sample, 0, len(next_audio_stretched)))

        current_audio = next_audio_stretched[next_resume_sample:]
        current_track = next_track

        # After stretching B to A, the carried-forward timeline plays at current_bpm.
        # Keep current_bpm unchanged so the set continues at the active tempo.
        current_bpm = current_bpm

        # Original source timestamp represented by current_audio[0].
        current_original_offset = actual_b_original_time + (mix_duration * next_rate)

        # Current audio is stretched relative to its original source.
        current_rate = next_rate

    # Add final remaining suffix.
    if len(current_audio) > 0:
        _append_part(full_parts, current_audio)

        final_start = set_cursor_seconds
        final_end = set_cursor_seconds + len(current_audio) / TARGET_SR

        timeline.append({
            "type": "track_body",
            "track_id": current_track["metadata"].get("track_id"),
            "filename": current_track["metadata"].get("filename"),
            "set_start": round(final_start, 3),
            "set_end": round(final_end, 3),
            "source_original_start": round(current_original_offset, 3),
            "source_original_end": round(
                current_original_offset + (len(current_audio) / TARGET_SR) * current_rate,
                3,
            ),
            "rate": round(float(current_rate), 5),
        })

        set_cursor_seconds = final_end

    if not full_parts:
        raise HTTPException(status_code=400, detail="Final set produced no audio.")

    final_audio = np.concatenate(full_parts)
    final_audio = _soft_limit(final_audio, threshold=0.95)
    final_audio = _normalise(final_audio, peak_target=0.95)

    sf.write(output_path, final_audio, TARGET_SR)

    return {
        "status": "success",
        "output_path": os.path.abspath(output_path),
        "duration_seconds": round(float(len(final_audio) / TARGET_SR), 3),
        "track_count": len(ordered_tracks),
        "transition_count": len(transitions),
        "transitions": transitions,
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# Queue wrapper
# ---------------------------------------------------------------------------

def render_queue_order_set(
    queue_path,
    library_path,
    setlist_path=None,
    preferred_mix_duration=None,
    output_dir="outputs",
    final_output_path="outputs/final_set.wav",
):
    library = load_library_metadata(library_path)
    queue = load_queue_state(queue_path)

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
        "status": "success",
        "mode": "queue_order_continuous_timestretch",
        "track_count": len(ordered_tracks),
        "ordered_tracks": [
            {
                "position": i + 1,
                "track_id": t["track_id"],
                "filename": t["filename"],
                "bpm": t.get("bpm"),
                "key": t.get("key"),
                "camelot": t.get("camelot"),
            }
            for i, t in enumerate(ordered_tracks)
        ],
        "transition_count": result["transition_count"],
        "transitions": result["transitions"],
        "timeline": result["timeline"],
        "final_mix": {
            "status": "success",
            "output_path": result["output_path"],
            "duration_seconds": result["duration_seconds"],
            "transition_count": result["transition_count"],
        },
    }