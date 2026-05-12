import os
import uuid
import numpy as np
import librosa
import pyrubberband as pyrb
import soundfile as sf
import scipy.signal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

app = FastAPI(title="Headless Audio Engine")

TARGET_SR = 44100

CAMELOT_MAP = {
    ("A", "minor"): "8A",
    ("E", "minor"): "9A",
    ("B", "minor"): "10A",
    ("F#", "minor"): "11A",
    ("C#", "minor"): "12A",
    ("G#", "minor"): "1A",
    ("D#", "minor"): "2A",
    ("A#", "minor"): "3A",
    ("F", "minor"): "4A",
    ("C", "minor"): "5A",
    ("G", "minor"): "6A",
    ("D", "minor"): "7A",

    ("C", "major"): "8B",
    ("G", "major"): "9B",
    ("D", "major"): "10B",
    ("A", "major"): "11B",
    ("E", "major"): "12B",
    ("B", "major"): "1B",
    ("F#", "major"): "2B",
    ("C#", "major"): "3B",
    ("G#", "major"): "4B",
    ("D#", "major"): "5B",
    ("A#", "major"): "6B",
    ("F", "major"): "7B",
}

class FXParameters(BaseModel):
    apply_reverb_tail: bool = False
    loop_track_a: bool = False
    hpf_sweep_end_freq: Optional[float] = None

class TransitionRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    transition_start_time: float
    mix_duration: int = 30
    output_dir: str = "outputs"

    transition_strategy: Literal[
        "bass_swap",
        "hpf_sweep",
        "auto_loop",
        "reverb_wash",
        "drop_mix",
        "harmonic_mix"
    ] = "bass_swap"

    fx_parameters: FXParameters = FXParameters()


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
    cutoff = min(cutoff, nyquist - 100)
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

def rms_level(y):
    return np.sqrt(np.mean(y ** 2) + 1e-9)


def match_rms_to_reference(y, reference_y, max_gain_db=6.0):
    """
    Matches y's RMS loudness to reference_y, with gain limiting.
    Prevents Track B from suddenly sounding too loud or too quiet.
    """
    ref_rms = rms_level(reference_y)
    y_rms = rms_level(y)

    gain = ref_rms / y_rms

    max_gain = 10 ** (max_gain_db / 20)
    min_gain = 1 / max_gain

    gain = np.clip(gain, min_gain, max_gain)

    return y * gain, gain

def bass_swap_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Splitting bass and mids/highs...")
    a_bass = apply_filter(segment_a, sr, 250, "low")
    a_mids_highs = apply_filter(segment_a, sr, 250, "high")

    b_bass = apply_filter(segment_b, sr, 250, "low")
    b_mids_highs = apply_filter(segment_b, sr, 250, "high")

    fade_out, fade_in = equal_power_fades(mix_samples)

    print("Applying DJ-style bass swap...")
    mixed_bass = np.zeros(mix_samples)

    mid_point = mix_samples // 2
    bass_fade_samples = int(0.05 * sr)

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

    return mixed_bass + mixed_mids_highs * 0.85

def dynamic_hpf_sweep(y, sr, start_freq=20, end_freq=5000, steps=64):
    """
    Applies a stepped high-pass filter sweep across the audio.
    Track A gradually loses low/mid energy as the cutoff rises.
    """
    n = len(y)
    output = np.zeros_like(y)

    step_size = max(1, n // steps)

    freqs = np.linspace(start_freq, end_freq, steps)

    for i in range(steps):
        start = i * step_size
        end = n if i == steps - 1 else min(n, (i + 1) * step_size)

        if start >= n:
            break

        cutoff = min(freqs[i], sr / 2 - 100)

        chunk = y[start:end]

        if len(chunk) < 32:
            output[start:end] = chunk
            continue

        try:
            output[start:end] = apply_filter(
                chunk,
                sr,
                cutoff=cutoff,
                btype="high",
                order=2
            )
        except Exception:
            output[start:end] = chunk

    return output


def hpf_sweep_transition(segment_a, segment_b, sr, end_freq=5000):
    mix_samples = len(segment_a)

    print("Applying HPF sweep transition...")

    fade_out, fade_in = equal_power_fades(mix_samples)

    swept_a = dynamic_hpf_sweep(
        y=segment_a,
        sr=sr,
        start_freq=20,
        end_freq=end_freq,
        steps=64
    )

    mixed = (swept_a * fade_out) + (segment_b * fade_in)

    return mixed

def auto_loop_track_a_segment(y_a, start_sample_a, mix_samples, sr):
    available = y_a[start_sample_a:]

    if len(available) >= mix_samples:
        return available[:mix_samples]

    print("Track A is short. Creating auto-loop runway.")

    min_loop_len = int(4 * sr)

    if len(available) < min_loop_len:
        loop_source = y_a[max(0, len(y_a) - min_loop_len):]
    else:
        loop_source = available

    if len(loop_source) == 0:
        return np.zeros(mix_samples)

    repeats = int(np.ceil(mix_samples / len(loop_source)))
    looped = np.tile(loop_source, repeats)

    return looped[:mix_samples]

def simple_reverb_tail(y, sr, decay_seconds=4.0, wet=0.45):
    decay_samples = int(decay_seconds * sr)

    impulse = np.exp(-np.linspace(0, 6, decay_samples))
    impulse = impulse / np.max(np.abs(impulse))

    reverb = scipy.signal.fftconvolve(y, impulse, mode="full")
    reverb = reverb[:len(y)]

    return (y * (1.0 - wet)) + (reverb * wet)


def reverb_wash_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Reverb Wash transition...")

    fade_out, fade_in = equal_power_fades(mix_samples)

    tail_samples = int(4 * sr)

    dry_a = segment_a.copy()
    tail_source = dry_a[-tail_samples:]

    washed_tail = simple_reverb_tail(
        y=tail_source,
        sr=sr,
        decay_seconds=6.0,
        wet=0.65
    )

    washed_a = dry_a.copy()
    washed_a[-tail_samples:] = washed_tail

    mixed = (washed_a * fade_out) + (segment_b * fade_in)

    return mixed

def drop_mix_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Drop Mix transition...")

    cut_samples = int(0.10 * sr)
    cut_samples = min(cut_samples, mix_samples)

    mixed = np.zeros(mix_samples)

    cut_point = mix_samples // 2

    mixed[:cut_point] = segment_a[:cut_point]
    mixed[cut_point:] = segment_b[cut_point:]

    fade_start = max(0, cut_point - cut_samples // 2)
    fade_end = min(mix_samples, cut_point + cut_samples // 2)

    fade_len = fade_end - fade_start

    if fade_len > 0:
        fade_out = np.linspace(1.0, 0.0, fade_len)
        fade_in = np.linspace(0.0, 1.0, fade_len)

        mixed[fade_start:fade_end] = (
            segment_a[fade_start:fade_end] * fade_out
            + segment_b[fade_start:fade_end] * fade_in
        )

    return mixed

def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    key_names = ["C", "C#", "D", "D#", "E", "F",
                 "F#", "G", "G#", "A", "A#", "B"]

    best_score = -np.inf
    best_key = None
    best_mode = None

    for i in range(12):
        major_score = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
        minor_score = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = key_names[i]
            best_mode = "major"

        if minor_score > best_score:
            best_score = minor_score
            best_key = key_names[i]
            best_mode = "minor"

    return best_key, best_mode, float(best_score)

def camelot_compatible(camelot_a, camelot_b):
    if camelot_a is None or camelot_b is None:
        return False

    num_a = int(camelot_a[:-1])
    mode_a = camelot_a[-1]

    num_b = int(camelot_b[:-1])
    mode_b = camelot_b[-1]

    compatible = set()

    compatible.add((num_a, mode_a))

    compatible.add(((num_a - 2) % 12 + 1, mode_a))
    compatible.add((num_a % 12 + 1, mode_a))

    compatible.add((num_a, "A" if mode_a == "B" else "B"))

    return (num_b, mode_b) in compatible

def apply_transition_strategy(segment_a, segment_b, sr, transition_strategy, fx_parameters=None):
    if transition_strategy == "bass_swap":
        return bass_swap_transition(segment_a, segment_b, sr)

    if transition_strategy == "hpf_sweep":
        end_freq = 5000

        if fx_parameters is not None and fx_parameters.hpf_sweep_end_freq is not None:
            end_freq = fx_parameters.hpf_sweep_end_freq

        return hpf_sweep_transition(
            segment_a=segment_a,
            segment_b=segment_b,
            sr=sr,
            end_freq=end_freq
        )

    if transition_strategy == "auto_loop":
        return bass_swap_transition(segment_a, segment_b, sr)

    if transition_strategy == "reverb_wash":
        return reverb_wash_transition(segment_a, segment_b, sr)

    if transition_strategy == "drop_mix":
        return drop_mix_transition(segment_a, segment_b, sr)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported transition_strategy: {transition_strategy}"
    )
    
def render_dj_transition(track_a_path, track_b_path, transition_start_time, mix_duration, output_dir,transition_strategy="bass_swap",
    fx_parameters=None):
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

    print("Snapping requested transition start to nearest beat in Track A...")
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

        transition_strategy = "bass_swap"
    remaining_a = len(y_a) - start_sample_a
    track_a_needs_loop = remaining_a < mix_samples

    if track_a_needs_loop:
        print("Track A is short near the end. Auto-loop may be used.")

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
        "sync_accuracy": round(sync_accuracy, 3),
        "phase_shift_samples": int(shift),
        "track_a_key": f"{key_a} {mode_a}",
        "track_b_key": f"{key_b} {mode_b}",
        "track_a_camelot": camelot_a,
        "track_b_camelot": camelot_b,
        "harmonic_compatible": harmonic_ok
    }


@app.post("/v1/engine/render-transition")
async def render_transition(req: TransitionRequest):
    return render_dj_transition(
        track_a_path=req.track_a_path,
        track_b_path=req.track_b_path,
        transition_start_time=req.transition_start_time,
        mix_duration=req.mix_duration,
        output_dir=req.output_dir,
        transition_strategy=req.transition_strategy,
        fx_parameters=req.fx_parameters
    )