
import os
import uuid
import soundfile as sf
import pyrubberband as pyrb
from fastapi import FastAPI, HTTPException
from core.analysis import *
from core.transitions import *
from core.strategy_router import apply_transition_strategy, get_strategy_mix_duration
from core.planner import find_best_sync_point
from models.schemas import CAMELOT_MAP,FXParameters


# Utility to convert stereo audio to mono by averaging the channels.
def ensure_mono(y):
    if y.ndim > 1:
        return np.mean(y, axis=0)
    return y

def _soft_limit(y, threshold=0.95):
    """
    Tanh-knee limiter applied before writing.
    Prevents inter-sample clipping from transient peaks that survive
    the peak normalisation step (common after reverb wash / techno drive).
    """
    over  = np.abs(y) > threshold
    signs = np.sign(y)
    excess = np.abs(y) - threshold
    y[over] = signs[over] * (
        threshold + (1.0 - threshold) * np.tanh(excess[over] / (1.0 - threshold))
    )
    return y
 
 
def match_rms_entry(segment_b, segment_a, sr, bpm, max_gain_db=6.0):
    """
    Matches Track B's gain to Track A over the first 4 beats only.
 
    FIX: The original matched RMS over the FULL segment, which includes
    breakdowns, silence, and outros. A track whose full-segment RMS is low
    but whose entry point is loud gets over-boosted and slams in.
    Measuring only the first 4 beats (the actual blend entry region) gives
    the correct loudness match for the moment that matters.
    """
    beat_samples = int((60.0 / bpm) * sr)
    check_len    = min(4 * beat_samples, len(segment_a), len(segment_b))
 
    if check_len < 512:
        return match_rms_to_reference(segment_b, segment_a, max_gain_db=max_gain_db)
 
    ref_rms = rms_level(segment_a[:check_len])
    b_rms   = rms_level(segment_b[:check_len])
 
    if b_rms < 1e-9:
        return segment_b, 1.0
 
    gain     = ref_rms / b_rms
    max_gain = 10 ** (max_gain_db / 20.0)
    gain     = float(np.clip(gain, 1.0 / max_gain, max_gain))
 
    return segment_b * gain, gain

# The primary engine function: handles loading, stretching, syncing, and rendering the final transition audio file.    
def render_dj_transition(
    track_a_path,
    track_b_path,
    transition_start_time,
    mix_duration,
    output_dir,
    transition_strategy="long_eq_blend",
    fx_parameters=None,
    track_b_entry_time=None
):
    if not os.path.exists(track_a_path):
        raise HTTPException(status_code=400, detail=f"Track A not found: {track_a_path}")

    if not os.path.exists(track_b_path):
        raise HTTPException(status_code=400, detail=f"Track B not found: {track_b_path}")

    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Starting Headless Audio Engine ---")

    mix_samples = int(mix_duration * TARGET_SR)

    print("Loading tracks...")
    y_a, _ = librosa.load(track_a_path, sr=TARGET_SR, mono=True)
    y_b, _ = librosa.load(track_b_path, sr=TARGET_SR, mono=True)

    print("Analyzing tempo and beat grids...")
    bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)
    bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)

    print(f"Track A BPM: {bpm_a:.2f}")
    print(f"Track B BPM: {bpm_b:.2f}")

    if bpm_a <= 0 or bpm_b <= 0:
        raise HTTPException(status_code=400, detail="Could not estimate BPM reliably.")

    stretch_ratio = bpm_b / bpm_a
 
    if not (0.80 <= stretch_ratio <= 1.25):
        if 0.80 <= stretch_ratio / 2.0 <= 1.25:
            stretch_ratio = stretch_ratio / 2.0
            print(f"Octave correction: halved stretch ratio to {stretch_ratio:.4f}")
        elif 0.80 <= stretch_ratio * 2.0 <= 1.25:
            stretch_ratio = stretch_ratio * 2.0
            print(f"Octave correction: doubled stretch ratio to {stretch_ratio:.4f}")
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"BPM gap too large to bridge: "
                    f"Track A={bpm_a:.2f}, Track B={bpm_b:.2f} "
                    f"(stretch ratio {stretch_ratio:.3f} is outside 0.80-1.25)"
                )
            )

    if abs(stretch_ratio - 1.0) > 0.01:
        print(f"Time-stretching Track B (ratio {stretch_ratio:.4f})...")
        y_b = pyrb.time_stretch(y_b, TARGET_SR, stretch_ratio)
        y_b = sharpen_transients(y_b, TARGET_SR, strength=0.35)
        print("Transients sharpened after stretch.")
    else:
        print("BPMs are close enough — no stretching required.")
 
    bpm_b_after = bpm_a
    _, beats_b  = safe_bpm(y_b, TARGET_SR)
    print(f"Track B effective BPM after stretch: {bpm_b_after:.2f}")


    print("Snapping Track A to 32-beat phrase boundary...")
    snapped_start_time, start_sample_a = snap_to_phrase_boundary(
        beats=beats_a,
        requested_time=transition_start_time,
        sr=TARGET_SR,
        phrase_beats=32
    )

    print(f"Requested start: {transition_start_time:.3f}s")
    print(f"Snapped start: {snapped_start_time:.3f}s")

    print("Estimating musical keys...")
    key_a, mode_a, key_conf_a = estimate_key(y_a, TARGET_SR)
    key_b, mode_b, key_conf_b = estimate_key(y_b, TARGET_SR)
 
    camelot_a  = CAMELOT_MAP.get((key_a, mode_a))
    camelot_b  = CAMELOT_MAP.get((key_b, mode_b))
    harmonic_ok = camelot_compatible(camelot_a, camelot_b)
 
    print(f"Track A: {key_a} {mode_a} / {camelot_a} (conf {key_conf_a:.3f})")
    print(f"Track B: {key_b} {mode_b} / {camelot_b} (conf {key_conf_b:.3f})")
    print(f"Harmonic compatible: {harmonic_ok}")
 
    if transition_strategy == "harmonic_mix" and not harmonic_ok:
        raise HTTPException(
            status_code=400,
            detail=f"Tracks not harmonically compatible: {camelot_a} vs {camelot_b}"
        )
    if mix_duration is None:
        mix_duration = get_strategy_mix_duration(transition_strategy, bpm_a)
        print(f"mix_duration not supplied — computed: {mix_duration:.1f}s")
 
    mix_samples = int(mix_duration * TARGET_SR)    
    remaining_a= len(y_a) - start_sample_a
    track_a_needs_loop = remaining_a < mix_samples
 
    if track_a_needs_loop:
        print("Track A is short near the end — using auto-loop runway.")
 
    use_loop = track_a_needs_loop or (
        fx_parameters is not None and getattr(fx_parameters, "loop_track_a", False)
    )
 
    if use_loop:
        segment_a = auto_loop_track_a_segment(
            y_a=y_a,
            start_sample_a=start_sample_a,
            mix_samples=mix_samples,
            sr=TARGET_SR
        )
    else:
        segment_a = pad_or_trim(y_a[start_sample_a:start_sample_a + mix_samples], mix_samples)
 
    # -----------------------------------------------------------------------
    # 9. Find Track B sync point
    # -----------------------------------------------------------------------
    if track_b_entry_time is not None:
        print("Using provided Track B entry time...")
        start_sample_b = int(track_b_entry_time * TARGET_SR)
        sync_accuracy  = None
    else:
        print("Finding best Track B sync point...")
        start_sample_b, sync_accuracy = find_best_sync_point(
            track_a_beats=beats_a,
            track_b_beats=beats_b,
            transition_start_sample=start_sample_a,
            mix_samples=mix_samples,
            offset_samples=1200,
            track_b_total_samples=len(y_b)
        )
        track_b_entry_time = start_sample_b / TARGET_SR
 
    print(f"Track B entry time: {track_b_entry_time:.3f}s  |  sample: {start_sample_b}")
    if sync_accuracy is not None:
        print(f"Beat sync accuracy: {sync_accuracy:.3f}")
 
    segment_b = pad_or_trim(y_b[start_sample_b:start_sample_b + mix_samples], mix_samples)
    print("Performing micro phase alignment...")
    shift, polarity_flip = phase_align(segment_a, segment_b, TARGET_SR)
 
    if polarity_flip:
        print("Polarity flip detected — inverting Track B.")
        segment_b = -segment_b
 
    if shift > 0:
        segment_b = np.pad(segment_b, (shift, 0))[:mix_samples]
    elif shift < 0:
        segment_b = np.pad(segment_b[abs(shift):], (0, abs(shift)))[:mix_samples]
 
    print(f"Phase shift applied: {shift} samples ({shift / TARGET_SR * 1000:.1f} ms)")
    
    drift_ms, is_aligned = verify_beat_alignment(segment_a, segment_b, TARGET_SR, bpm_a)
    print(f"Beat alignment drift: {drift_ms:.1f} ms  |  aligned: {is_aligned}")
 
    if not is_aligned:
        correction_samples = int((drift_ms / 1000.0) * TARGET_SR)
        print(f"Applying drift correction: {correction_samples} samples")
        if correction_samples > 0:
            segment_b = np.pad(segment_b, (correction_samples, 0))[:mix_samples]
        elif correction_samples < 0:
            segment_b = np.pad(segment_b[abs(correction_samples):], (0, abs(correction_samples)))[:mix_samples]
            
    segment_b, rms_gain = match_rms_entry(
        segment_b=segment_b,
        segment_a=segment_a,
        sr=TARGET_SR,
        bpm=bpm_a,
        max_gain_db=6.0
    )
    print(f"Track B entry-region RMS gain applied: {rms_gain:.3f} ({20 * np.log10(rms_gain + 1e-9):.1f} dB)")

    segment_b = apply_entry_ramp(segment_b, ramp_ms=15.0, sr=TARGET_SR)

    if fx_parameters is None:
        fx_parameters = FXParameters()
 
    # Pydantic models are immutable by default — create updated copy
    fx_parameters = fx_parameters.model_copy(update={"bpm": bpm_a})

    print(f"Applying transition strategy: {transition_strategy}")
    mixed_segment = apply_transition_strategy(
        segment_a=segment_a,
        segment_b=segment_b,
        sr=TARGET_SR,
        transition_strategy=transition_strategy,
        fx_parameters=fx_parameters
    )

    print("Limiting and normalising output...")
    mixed_segment = _soft_limit(mixed_segment, threshold=0.95)
 
    peak = np.max(np.abs(mixed_segment))
    if peak > 1e-9:
        mixed_segment = mixed_segment / peak * 0.95  # always normalise
 
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
        "phase_shift_ms":round(shift / TARGET_SR * 1000.0, 2),
        "track_a_key": f"{key_a} {mode_a}",
        "track_b_key": f"{key_b} {mode_b}",
        "track_a_camelot": camelot_a,
        "track_b_camelot": camelot_b,
        "harmonic_compatible": harmonic_ok
    }