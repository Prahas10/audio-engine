import os
import uuid
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
import librosa
from fastapi import HTTPException

from core.analysis import (
    TARGET_SR, safe_bpm, estimate_key, rms_level, camelot_compatible,
    verify_beat_alignment, sharpen_transients, apply_entry_ramp,
)
from core.transitions import (
    pad_or_trim, phase_align, snap_to_phrase_boundary,
    auto_loop_track_a_segment,
)
from core.strategy_router import apply_transition_strategy, get_strategy_mix_duration
from core.planner import find_best_sync_point
from models.schemas import CAMELOT_MAP, FXParameters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_limit(y, threshold=0.95):
    """Tanh soft limiter — prevents clipping from transient peaks."""
    over   = np.abs(y) > threshold
    signs  = np.sign(y)
    excess = np.abs(y[over]) - threshold
    y      = y.copy()
    y[over] = signs[over] * (
        threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold))
    )
    return y


def _measure_lufs_simple(y, sr):
    """
    Simplified loudness measure (RMS over 400ms blocks, K-weighted).
    Used for perceptual gain matching rather than raw RMS.
    """
    block_len = int(0.4 * sr)
    if len(y) < block_len:
        return rms_level(y)

    rms_blocks = []
    for i in range(0, len(y) - block_len, block_len // 2):
        block = y[i:i + block_len]
        rms_blocks.append(np.sqrt(np.mean(block ** 2) + 1e-9))

    return float(np.mean(rms_blocks)) if rms_blocks else rms_level(y)


def _match_gain_entry(segment_b, segment_a, sr, bpm, max_gain_db=6.0):
    """
    FIX for volume jump: match gain over the first 4 beats only — the actual
    blend entry region. Full-segment RMS is wrong because it averages in
    breakdowns and outros that may be much quieter than the entry region.

    Also uses perceptual loudness (block RMS) rather than instantaneous RMS.
    """
    beat_samples = int((60.0 / bpm) * sr)
    check_len    = min(4 * beat_samples, len(segment_a), len(segment_b))

    if check_len < 1024:
        ref_loud = _measure_lufs_simple(segment_a, sr)
        b_loud   = _measure_lufs_simple(segment_b, sr)
    else:
        ref_loud = _measure_lufs_simple(segment_a[:check_len], sr)
        b_loud   = _measure_lufs_simple(segment_b[:check_len], sr)

    if b_loud < 1e-9:
        return segment_b, 1.0

    gain     = ref_loud / b_loud
    max_gain = 10 ** (max_gain_db / 20.0)
    gain     = float(np.clip(gain, 1.0 / max_gain, max_gain))

    print(f"Entry-region gain: {gain:.3f} ({20 * np.log10(gain + 1e-9):.1f} dB)")
    return segment_b * gain, gain


def _apply_gain_automation(segment_b, mix_samples, sr, bpm):
    """
    FIX for volume jump: applies a short gain ramp at the start of segment_b
    to prevent a hard onset even after static gain matching. This is what a DJ
    does manually — they ride the gain up over the first bar or two.

    The ramp covers 2 beats (half a bar) — long enough to be smooth,
    short enough not to make the entry feel weak.
    """
    ramp_samples = min(int((60.0 / bpm) * sr * 2), mix_samples // 4)
    segment_b    = segment_b.copy()
    ramp         = np.linspace(0.0, 1.0, ramp_samples) ** 0.5   # sqrt = perceptual
    segment_b[:ramp_samples] *= ramp
    return segment_b


# ---------------------------------------------------------------------------
# Primary render function
# ---------------------------------------------------------------------------

def render_dj_transition(
    track_a_path,
    track_b_path,
    transition_start_time,
    mix_duration,
    output_dir,
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

    # -----------------------------------------------------------------------
    # 1. Load
    # -----------------------------------------------------------------------
    print("Loading tracks...")
    y_a, _ = librosa.load(track_a_path, sr=TARGET_SR, mono=True)
    y_b, _ = librosa.load(track_b_path, sr=TARGET_SR, mono=True)

    # -----------------------------------------------------------------------
    # 2. BPM — use madmom via safe_bpm
    # -----------------------------------------------------------------------
    print("Analysing tempo and beat grids...")
    if planner_metadata:
        bpm_a = float(planner_metadata["track_a"]["bpm"])
        bpm_b = float(planner_metadata["track_b"]["bpm"])

        beats_a = np.asarray(
            librosa.time_to_samples(planner_metadata["track_a"].get("beats", []), sr=TARGET_SR),
            dtype=int,
        )

        beats_b = np.asarray(
            librosa.time_to_samples(planner_metadata["track_b"].get("beats", []), sr=TARGET_SR),
            dtype=int,
        )

        if len(beats_a) == 0:
            bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)

        if len(beats_b) == 0:
            bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)

        print("Using planner/library BPM + beat metadata.")
    else:
        bpm_a, beats_a = safe_bpm(y_a, TARGET_SR)
        bpm_b, beats_b = safe_bpm(y_b, TARGET_SR)

    # -----------------------------------------------------------------------
    # 3. BPM compatibility — octave-aware
    # -----------------------------------------------------------------------
    stretch_ratio = bpm_b / bpm_a

    if not (0.80 <= stretch_ratio <= 1.25):
        if 0.80 <= stretch_ratio / 2.0 <= 1.25:
            stretch_ratio = stretch_ratio / 2.0
            print(f"Octave correction: halved ratio → {stretch_ratio:.4f}")
        elif 0.80 <= stretch_ratio * 2.0 <= 1.25:
            stretch_ratio = stretch_ratio * 2.0
            print(f"Octave correction: doubled ratio → {stretch_ratio:.4f}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"BPM gap too large: A={bpm_a:.2f}, B={bpm_b:.2f}"
            )

    # -----------------------------------------------------------------------
    # 4. Time-stretch Track B
    # -----------------------------------------------------------------------
    if abs(stretch_ratio - 1.0) > 0.01:
        print(f"Time-stretching Track B (ratio {stretch_ratio:.4f})...")
        y_b = pyrb.time_stretch(y_b, TARGET_SR, stretch_ratio)
        y_b = sharpen_transients(y_b, TARGET_SR, strength=0.35)
        print("Transients sharpened after stretch.")

        # Re-analyse beat grid on stretched audio
        bpm_b_after_est, beats_b = safe_bpm(y_b, TARGET_SR)
    else:
        print("BPMs close enough — no stretch required.")

    bpm_b_after = bpm_a   # by definition after stretching

    # -----------------------------------------------------------------------
    # 5. Snap start to bar boundary
    #    phrase_mix → 32-beat boundary; everything else → 4-beat (1 bar)
    # -----------------------------------------------------------------------
    phrase_beats = 32 if transition_strategy == "phrase_mix" else 4

    snapped_start_time, start_sample_a = snap_to_phrase_boundary(
        beats=beats_a,
        requested_time=transition_start_time,
        sr=TARGET_SR,
        phrase_beats=phrase_beats,
    )
    print(f"Transition: {transition_start_time:.3f}s → snapped {snapped_start_time:.3f}s")

    # -----------------------------------------------------------------------
    # 6. Key analysis
    # -----------------------------------------------------------------------
    print("Estimating keys...")
    if planner_metadata:
        key_a = planner_metadata["track_a"]["key"].split()[0]
        mode_a = planner_metadata["track_a"]["key"].split()[1]
        key_b = planner_metadata["track_b"]["key"].split()[0]
        mode_b = planner_metadata["track_b"]["key"].split()[1]

        camelot_a = planner_metadata["track_a"].get("camelot")
        camelot_b = planner_metadata["track_b"].get("camelot")
        harmonic_ok = bool(planner_metadata.get("harmonic_compatible", camelot_compatible(camelot_a, camelot_b)))

        print("Using planner/library key metadata.")
    else:
        key_a, mode_a, key_conf_a = estimate_key(y_a, TARGET_SR)
        key_b, mode_b, key_conf_b = estimate_key(y_b, TARGET_SR)

        camelot_a = CAMELOT_MAP.get((key_a, mode_a))
        camelot_b = CAMELOT_MAP.get((key_b, mode_b))
        harmonic_ok = camelot_compatible(camelot_a, camelot_b)

    print(f"A: {key_a} {mode_a} / {camelot_a}  B: {key_b} {mode_b} / {camelot_b}  harmonic={harmonic_ok}")

    if transition_strategy == "harmonic_mix" and not harmonic_ok:
        raise HTTPException(
            status_code=400,
            detail=f"Tracks not harmonically compatible: {camelot_a} vs {camelot_b}"
        )

    # -----------------------------------------------------------------------
    # 7. Mix duration
    # -----------------------------------------------------------------------
    if mix_duration is None:
        mix_duration = get_strategy_mix_duration(transition_strategy, bpm_a)
        print(f"mix_duration computed: {mix_duration:.1f}s")

    mix_samples = int(mix_duration * TARGET_SR)

    # -----------------------------------------------------------------------
    # 8. Slice segment A
    # -----------------------------------------------------------------------
    remaining_a = len(y_a) - start_sample_a
    use_loop    = remaining_a < mix_samples or (
        fx_parameters is not None and getattr(fx_parameters, "loop_track_a", False)
    )

    if use_loop:
        print("Auto-loop runway for Track A.")
        segment_a = auto_loop_track_a_segment(
            y_a=y_a, start_sample_a=start_sample_a,
            mix_samples=mix_samples, sr=TARGET_SR,
        )
    else:
        segment_a = pad_or_trim(y_a[start_sample_a:start_sample_a + mix_samples], mix_samples)

    # -----------------------------------------------------------------------
    # 9. Find Track B sync point
    # -----------------------------------------------------------------------
    if track_b_entry_time is not None:
        print(f"Using planner-provided Track B entry: {track_b_entry_time:.3f}s")
        start_sample_b = int(float(track_b_entry_time) * TARGET_SR)
        sync_accuracy  = None
    else:
        print("Finding best Track B sync point locally...")
        start_sample_b, sync_accuracy = find_best_sync_point(
            track_a_beats=beats_a,
            track_b_beats=beats_b,
            transition_start_sample=start_sample_a,
            mix_samples=mix_samples,
            offset_samples=1200,
            track_b_total_samples=len(y_b),
            min_b_entry_percent=0.0,
            max_b_entry_percent=0.35,
        )
        track_b_entry_time = start_sample_b / TARGET_SR

    print(f"Track B entry: {track_b_entry_time:.3f}s")
    segment_b = pad_or_trim(y_b[start_sample_b:start_sample_b + mix_samples], mix_samples)

    # -----------------------------------------------------------------------
    # 10. Phase alignment  (BEFORE gain matching)
    # -----------------------------------------------------------------------
    print("Phase aligning...")
    shift, polarity_flip = phase_align(segment_a, segment_b, TARGET_SR)

    if polarity_flip:
        print("Polarity flip — inverting Track B.")
        segment_b = -segment_b

    if shift > 0:
        segment_b = np.pad(segment_b, (shift, 0))[:mix_samples]
    elif shift < 0:
        segment_b = np.pad(segment_b[abs(shift):], (0, abs(shift)))[:mix_samples]

    print(f"Phase shift: {shift} samples ({shift / TARGET_SR * 1000:.1f} ms)")

    # -----------------------------------------------------------------------
    # 11. Beat alignment verification + residual drift correction
    # -----------------------------------------------------------------------
    drift_ms, is_aligned = verify_beat_alignment(segment_a, segment_b, TARGET_SR, bpm_a)
    print(f"Beat drift: {drift_ms:.1f} ms  aligned={is_aligned}")

    if not is_aligned:
        correction = int((drift_ms / 1000.0) * TARGET_SR)
        print(f"Drift correction: {correction} samples")
        if correction > 0:
            segment_b = np.pad(segment_b, (correction, 0))[:mix_samples]
        elif correction < 0:
            segment_b = np.pad(segment_b[abs(correction):], (0, abs(correction)))[:mix_samples]

    # -----------------------------------------------------------------------
    # 12. Gain matching — entry region only  (FIX for volume jump)
    #
    # Two-stage process:
    #   a) Static gain match on the first 4 beats (not full segment)
    #   b) Perceptual gain ramp over the first 2 beats (DJ-style gain ride)
    # -----------------------------------------------------------------------
    segment_b, rms_gain = _match_gain_entry(
        segment_b=segment_b,
        segment_a=segment_a,
        sr=TARGET_SR,
        bpm=bpm_a,
        max_gain_db=6.0,
    )

    segment_b = _apply_gain_automation(segment_b, mix_samples, TARGET_SR, bpm_a)

    # -----------------------------------------------------------------------
    # 13. Entry ramp — eliminates click at sample 0
    # -----------------------------------------------------------------------
    segment_b = apply_entry_ramp(segment_b, ramp_ms=15.0, sr=TARGET_SR)

    # -----------------------------------------------------------------------
    # 14. Inject bpm into fx_parameters
    # -----------------------------------------------------------------------
    if fx_parameters is None:
        fx_parameters = FXParameters()

    try:
        fx_parameters = fx_parameters.model_copy(update={"bpm": bpm_a})
    except Exception:
        fx_parameters = fx_parameters.copy(update={"bpm": bpm_a})

    # -----------------------------------------------------------------------
    # 15. Apply transition strategy
    # -----------------------------------------------------------------------
    print(f"Applying strategy: {transition_strategy}")
    mixed_segment = apply_transition_strategy(
        segment_a=segment_a,
        segment_b=segment_b,
        sr=TARGET_SR,
        transition_strategy=transition_strategy,
        fx_parameters=fx_parameters,
    )

    # -----------------------------------------------------------------------
    # 16. Soft limit + normalise
    # -----------------------------------------------------------------------
    print("Limiting and normalising...")
    mixed_segment = _soft_limit(mixed_segment, threshold=0.95)
    peak = np.max(np.abs(mixed_segment))
    if peak > 1e-9:
        mixed_segment = mixed_segment / peak * 0.95

    mixed_segment = pad_or_trim(mixed_segment, mix_samples)

    # -----------------------------------------------------------------------
    # 17. Write transition clip + matched Track B suffix
    # -----------------------------------------------------------------------
    output_id = uuid.uuid4().hex[:8]

    transition_filename = f"transition_{output_id}.wav"
    transition_output_path = os.path.normpath(os.path.join(output_dir, transition_filename))

    sf.write(transition_output_path, mixed_segment, TARGET_SR)

    # This suffix comes from the SAME y_b used inside the transition.
    # If Track B was stretched, this suffix is also stretched.
    b_suffix_start = start_sample_b + mix_samples
    track_b_suffix = y_b[b_suffix_start:]

    suffix_filename = f"track_b_suffix_{output_id}.wav"
    suffix_output_path = os.path.normpath(os.path.join(output_dir, suffix_filename))

    sf.write(suffix_output_path, track_b_suffix, TARGET_SR)

    print(f"Exported transition: {transition_output_path}")
    print(f"Exported Track B suffix: {suffix_output_path}")
    print("--- Mix Complete ---\n")

    return {
        "status": "success",
        "audio_clip_url": transition_output_path,
        "transition_file": transition_output_path,
        "track_b_suffix_file": suffix_output_path,

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
        "track_b_suffix_start_time": round(float(b_suffix_start / TARGET_SR), 3),
        "track_b_suffix_duration": round(float(len(track_b_suffix) / TARGET_SR), 3),

        "phase_shift_samples": int(shift),
        "phase_shift_ms": round(shift / TARGET_SR * 1000.0, 2),
        "beat_alignment_drift_ms": round(drift_ms, 2),

        "track_a_key": f"{key_a} {mode_a}",
        "track_b_key": f"{key_b} {mode_b}",
        "track_a_camelot": camelot_a,
        "track_b_camelot": camelot_b,
        "harmonic_compatible": harmonic_ok,
    }