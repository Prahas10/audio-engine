import os
import uuid
import numpy as np
import librosa
import soundfile as sf
import scipy.signal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="MixingBear Headless Audio Engine")


class TransitionRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    transition_start_time: float
    mix_duration: int = 30
    output_dir: str = "outputs"


TARGET_SR = 44100


def ensure_mono(y):
    if y.ndim > 1:
        return np.mean(y, axis=0)
    return y


def pad_or_trim(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]


def safe_bpm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    tempo = float(np.asarray(tempo).squeeze())
    return tempo, beats.astype(int)


def find_best_sync_point(track_a_beats, track_b_beats, transition_start_sample, mix_samples, offset_samples=1200):
    """
    Finds the best point in Track B to align with Track A during the transition window.
    This borrows the useful idea from the second code: try multiple beat alignments and score them.
    """

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


def apply_filter(y, sr, cutoff, btype, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype)
    return scipy.signal.filtfilt(b, a, y)


def phase_align(track_a_slice, track_b_slice, sr):
    slice_len = int(0.05 * sr)

    a = pad_or_trim(track_a_slice, slice_len)
    b = pad_or_trim(track_b_slice, slice_len)

    try:
        a_sub = apply_filter(a, sr, 100, "low", order=2)
        b_sub = apply_filter(b, sr, 100, "low", order=2)

        correlation = scipy.signal.correlate(a_sub, b_sub, mode="full")
        best_index = np.argmax(np.abs(correlation))
        shift = best_index - (len(b_sub) - 1)

        max_corr_value = correlation[best_index]
        polarity_flip = max_corr_value < 0

        return shift, polarity_flip

    except Exception:
        return 0, False


def equal_power_fades(n):
    t = np.linspace(0, 1, n)
    fade_out = np.cos(t * np.pi / 2)
    fade_in = np.sin(t * np.pi / 2)
    return fade_out, fade_in


def render_dj_transition(track_a_path, track_b_path, transition_start_time, mix_duration, output_dir):
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
        y_b = librosa.effects.time_stretch(y_b, rate=stretch_ratio)

    print("Re-analyzing Track B after stretching...")
    bpm_b_after, beats_b = safe_bpm(y_b, TARGET_SR)

    print(f"Track B BPM after stretch: {bpm_b_after:.2f}")

    print("Snapping requested transition start to nearest beat in Track A...")
    beat_times_a = librosa.samples_to_time(beats_a, sr=TARGET_SR)

    closest_beat_idx = np.argmin(np.abs(beat_times_a - transition_start_time))
    snapped_start_time = float(beat_times_a[closest_beat_idx])
    start_sample_a = int(snapped_start_time * TARGET_SR)

    print(f"Requested start: {transition_start_time:.3f}s")
    print(f"Snapped start: {snapped_start_time:.3f}s")

    if start_sample_a + mix_samples > len(y_a):
        raise HTTPException(
            status_code=400,
            detail="Track A does not have enough audio after transition_start_time for this mix_duration."
        )

    print("Finding best Track B sync point using beat-overlap scoring...")
    start_sample_b, sync_accuracy = find_best_sync_point(
        track_a_beats=beats_a,
        track_b_beats=beats_b,
        transition_start_sample=start_sample_a,
        mix_samples=mix_samples,
        offset_samples=int(0.027 * TARGET_SR)
    )

    print(f"Best Track B start sample: {start_sample_b}")
    print(f"Beat sync accuracy: {sync_accuracy:.2f}")

    segment_a = y_a[start_sample_a:start_sample_a + mix_samples]
    segment_b = y_b[start_sample_b:start_sample_b + mix_samples]

    segment_a = pad_or_trim(segment_a, mix_samples)
    segment_b = pad_or_trim(segment_b, mix_samples)

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

    print("Splitting bass and mids/highs...")
    a_bass = apply_filter(segment_a, TARGET_SR, 250, "low")
    a_mids_highs = apply_filter(segment_a, TARGET_SR, 250, "high")

    b_bass = apply_filter(segment_b, TARGET_SR, 250, "low")
    b_mids_highs = apply_filter(segment_b, TARGET_SR, 250, "high")

    fade_out, fade_in = equal_power_fades(mix_samples)

    print("Applying DJ-style bass swap...")
    mixed_bass = np.zeros(mix_samples)

    mid_point = mix_samples // 2
    bass_fade_samples = int(0.05 * TARGET_SR)

    fade_start = max(0, mid_point - bass_fade_samples // 2)
    fade_end = min(mix_samples, fade_start + bass_fade_samples)

    mixed_bass[:fade_start] = a_bass[:fade_start]

    bass_fade_out = np.linspace(1.0, 0.0, fade_end - fade_start)
    bass_fade_in = np.linspace(0.0, 1.0, fade_end - fade_start)

    mixed_bass[fade_start:fade_end] = (
        a_bass[fade_start:fade_end] * bass_fade_out
        + b_bass[fade_start:fade_end] * bass_fade_in
    )

    mixed_bass[fade_end:] = b_bass[fade_end:]

    print("Applying smooth mid/high crossfade...")
    mixed_mids_highs = (
        a_mids_highs * fade_out
        + b_mids_highs * fade_in
    )

    mixed_segment = mixed_bass + mixed_mids_highs * 0.85

    print("Normalizing output...")
    peak = np.max(np.abs(mixed_segment))

    if peak > 0:
        mixed_segment = mixed_segment / peak * 0.95

    mixed_segment = pad_or_trim(mixed_segment, mix_samples)

    output_filename = f"transition_test.wav"
    output_path = os.path.join(output_dir, output_filename)

    sf.write(output_path, mixed_segment, TARGET_SR)

    print(f"Exported: {output_path}")
    print("--- Mix Complete ---\n")

    return {
        "status": "success",
        "audio_clip_url": output_path,
        "duration_seconds": mix_duration,
        "snapped_transition_start_time": round(snapped_start_time, 3),
        "track_a_bpm": round(bpm_a, 2),
        "track_b_original_bpm": round(bpm_b, 2),
        "track_b_stretched_bpm": round(bpm_b_after, 2),
        "sync_accuracy": round(sync_accuracy, 3),
        "phase_shift_samples": int(shift)
    }


@app.post("/v1/engine/render-transition")
async def render_transition(req: TransitionRequest):
    return render_dj_transition(
        track_a_path=req.track_a_path,
        track_b_path=req.track_b_path,
        transition_start_time=req.transition_start_time,
        mix_duration=req.mix_duration,
        output_dir=req.output_dir
    )