
import os
import uuid
import soundfile as sf
import pyrubberband as pyrb
from fastapi import FastAPI, HTTPException
from core.analysis import *
from core.transitions import *
from core.strategy_router import apply_transition_strategy
from core.planner import find_best_sync_point
from models.schemas import CAMELOT_MAP


# Utility to convert stereo audio to mono by averaging the channels.
def ensure_mono(y):
    if y.ndim > 1:
        return np.mean(y, axis=0)
    return y


# The primary engine function: handles loading, stretching, syncing, and rendering the final transition audio file.    
def render_dj_transition(track_a_path, track_b_path, transition_start_time, mix_duration, output_dir,transition_strategy="bass_swap",
    fx_parameters=None,track_b_entry_time=None):
    if not os.path.exists(track_a_path):
        raise HTTPException(status_code=400, detail=f"Track A not found: {track_a_path}")

    if not os.path.exists(track_b_path):
        raise HTTPException(status_code=400, detail=f"Track B not found: {track_b_path}")

    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Starting MixingBear Headless Audio Engine ---")

    mix_samples = int(mix_duration * TARGET_SR)

    print("Loading tracks...")
    y_a, _ = librosa.load(track_a_path, sr=TARGET_SR, mono=True)
    y_b, _ = librosa.load(track_b_path, sr=TARGET_SR, mono=True)

    y_a = ensure_mono(y_a)
    y_b = ensure_mono(y_b)

    print("Analyzing tempo and beat grids...")
    bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)
    bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)

    print(f"Track A BPM: {bpm_a:.2f}")
    print(f"Track B BPM: {bpm_b:.2f}")

    if bpm_a <= 0 or bpm_b <= 0:
        raise HTTPException(status_code=400, detail="Could not estimate BPM reliably.")

    if abs(bpm_a - bpm_b) > 15:
        raise HTTPException(
            status_code=400,
            detail=f"BPM difference too large: Track A={bpm_a:.2f}, Track B={bpm_b:.2f}"
        )

    print("Time-stretching Track B to match Track A...")
    stretch_ratio = bpm_b / bpm_a

    if abs(stretch_ratio - 1.0) > 0.003:
        y_b = pyrb.time_stretch(y_b, TARGET_SR, stretch_ratio)
    else:
        print("BPMs are close enough. No stretching required.")

    print("Re-analyzing Track B after stretching...")
    bpm_b_after, beats_b = safe_bpm(y_b, TARGET_SR)

    print(f"Track B BPM after stretch: {bpm_b_after:.2f}")

    if transition_strategy == "phrase_mix":
        print("Snapping to phrase boundary...")
        snapped_start_time, start_sample_a = snap_to_phrase_boundary(
            beats=beats_a,
            requested_time=transition_start_time,
            sr=TARGET_SR,
            phrase_beats=32
        )
    else:
        beat_times_a = librosa.samples_to_time(beats_a, sr=TARGET_SR)
        closest_beat_idx = np.argmin(np.abs(beat_times_a - transition_start_time))

        snapped_start_time = float(beat_times_a[closest_beat_idx])
        start_sample_a = int(snapped_start_time * TARGET_SR)

    print(f"Requested start: {transition_start_time:.3f}s")
    print(f"Snapped start: {snapped_start_time:.3f}s")
    print("Estimating musical keys...")

    key_a, mode_a, key_conf_a = estimate_key(y_a, TARGET_SR)
    key_b, mode_b, key_conf_b = estimate_key(y_b, TARGET_SR)

    camelot_a = CAMELOT_MAP.get((key_a, mode_a))
    camelot_b = CAMELOT_MAP.get((key_b, mode_b))

    harmonic_ok = camelot_compatible(camelot_a, camelot_b)

    print(f"Track A key: {key_a} {mode_a} / Camelot {camelot_a}")
    print(f"Track B key: {key_b} {mode_b} / Camelot {camelot_b}")
    print(f"Harmonic compatible: {harmonic_ok}")
    
    if transition_strategy == "harmonic_mix":
        if not harmonic_ok:
            raise HTTPException(
                status_code=400,
                detail=f"Tracks are not harmonically compatible. Track A={camelot_a}, Track B={camelot_b}"
            )

    remaining_a = len(y_a) - start_sample_a
    track_a_needs_loop = remaining_a < mix_samples

    if track_a_needs_loop:
        print("Track A is short near the end. Auto-loop may be used.")

    if track_b_entry_time is not None:
        print("Using Brain-provided Track B entry time...")
        start_sample_b = int(track_b_entry_time * TARGET_SR)
        sync_accuracy = None
    else:
        print("Finding best Track B sync point...")
        start_sample_b, sync_accuracy = find_best_sync_point(
            track_a_beats=beats_a,
            track_b_beats=beats_b,
            transition_start_sample=start_sample_a,
            mix_samples=mix_samples,
            offset_samples=int(0.027 * TARGET_SR)
        )

        track_b_entry_time = start_sample_b / TARGET_SR

    print(f"Track B entry time: {track_b_entry_time:.3f}s")

    print(f"Best Track B start sample: {start_sample_b}")
    if sync_accuracy is None:
        print("Beat sync accuracy: Brain-provided entry time, not recalculated.")
    else:
        print(f"Beat sync accuracy: {sync_accuracy:.2f}")

    if transition_strategy == "auto_loop" or (
        fx_parameters is not None and fx_parameters.loop_track_a
    ):
        segment_a = auto_loop_track_a_segment(
            y_a=y_a,
            start_sample_a=start_sample_a,
            mix_samples=mix_samples,
            sr=TARGET_SR
        )
    else:
        segment_a = pad_or_trim(
            y_a[start_sample_a:start_sample_a + mix_samples],
            mix_samples
        )

    segment_b = pad_or_trim(
        y_b[start_sample_b:start_sample_b + mix_samples],
        mix_samples
    )
    
    segment_b, rms_gain = match_rms_to_reference(
        y=segment_b,
        reference_y=segment_a,
        max_gain_db=3.0
    )

    print(f"Track B RMS gain applied: {rms_gain:.3f}")

    print("Performing micro phase alignment...")
    shift, polarity_flip = phase_align(segment_a, segment_b, TARGET_SR)

    if polarity_flip:
        print("Polarity flip detected. Flipping Track B.")
        segment_b = -segment_b

    if shift > 0:
        segment_b = np.pad(segment_b, (shift, 0))[:mix_samples]
    elif shift < 0:
        segment_b = np.pad(segment_b[abs(shift):], (0, abs(shift)))[:mix_samples]

    print(f"Phase shift applied: {shift} samples")

    print(f"Applying transition strategy: {transition_strategy}")

    mixed_segment = apply_transition_strategy(
        segment_a=segment_a,
        segment_b=segment_b,
        sr=TARGET_SR,
        transition_strategy=transition_strategy,
        fx_parameters=fx_parameters
    )

    print("Normalizing output...")
    peak = np.max(np.abs(mixed_segment))

    if peak > 0.98:
        mixed_segment = mixed_segment / peak * 0.98

    mixed_segment = pad_or_trim(mixed_segment, mix_samples)

    output_filename = f"transition_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.normpath(os.path.join(output_dir, output_filename))

    sf.write(output_path, mixed_segment, TARGET_SR)

    print(f"Exported: {output_path}")
    print("--- Mix Complete ---\n")

    return {
        "status": "success",
        "audio_clip_url": output_path,
        "duration_seconds": mix_duration,
        "transition_strategy": transition_strategy,
        "snapped_transition_start_time": round(snapped_start_time, 3),
        "track_a_bpm": round(bpm_a, 2),
        "track_b_original_bpm": round(bpm_b, 2),
        "track_b_stretched_bpm": round(bpm_b_after, 2),
        "track_b_rms_gain": round(float(rms_gain), 3),
        "sync_accuracy": None if sync_accuracy is None else round(float(sync_accuracy), 3),
        "track_b_entry_time": round(float(track_b_entry_time), 3),
        "track_b_entry_sample": int(start_sample_b),
        "phase_shift_samples": int(shift),
        "track_a_key": f"{key_a} {mode_a}",
        "track_b_key": f"{key_b} {mode_b}",
        "track_a_camelot": camelot_a,
        "track_b_camelot": camelot_b,
        "harmonic_compatible": harmonic_ok
    }