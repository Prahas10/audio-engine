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

class AutoRenderRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    preferred_mix_duration: int = 30
    output_dir: str = "outputs"
    
class TransitionRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    transition_start_time: float
    track_b_entry_time: Optional[float] = None
    mix_duration: int = 30
    output_dir: str = "outputs"

    transition_strategy: Literal[
        "bass_swap",
        "hpf_sweep",
        "auto_loop",
        "reverb_wash",
        "drop_mix",
        "harmonic_mix",
        "phrase_mix",
        "lpf_sweep",
        "echo_out",
        "loop_roll",
        "long_eq_blend",
        "energy_blend",
        "percussion_blend",
        "breakdown_blend",
        "ambient_transition",
        "techno_filter_drive"
    ] = "bass_swap"

    fx_parameters: FXParameters = FXParameters()

class PlanTransitionRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    preferred_mix_duration: int = 30

# Calculates the RMS energy over time to identify high and low intensity sections of a track.
def get_energy_curve(y, sr, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    if np.max(rms) > 0:
        rms = rms / np.max(rms)

    return times, rms

# Identifies valid musical phrase boundaries (e.g., every 32 beats) within the typical transition window of a song.
def phrase_boundary_candidates(beats, sr, song_duration, phrase_beats=32):
    beat_times = librosa.samples_to_time(beats, sr=sr)

    if len(beat_times) == 0:
        return []

    start_search = song_duration * 0.55
    end_search = song_duration * 0.92

    candidates = []

    for i in range(0, len(beat_times), phrase_beats):
        t = float(beat_times[i])

        if start_search <= t <= end_search:
            candidates.append(t)

    return candidates

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

# Determines the most appropriate DJ transition technique based on BPM difference and harmonic compatibility.
def choose_strategy(
    harmonic_ok,
    bpm_a,
    bpm_b,
    best_time,
    song_duration_a,
    mix_duration
):
    bpm_delta = abs(bpm_a - bpm_b)
    remaining = song_duration_a - best_time

    if not harmonic_ok and bpm_delta > 8:
        return "reverb_wash"

    if remaining < mix_duration:
        return "auto_loop"

    if harmonic_ok and bpm_delta <= 6:
        return "energy_blend"

    if bpm_delta <= 5:
        return "phrase_mix"

    if bpm_delta <= 10:
        return "hpf_sweep"

    return "reverb_wash"

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

    strategy = choose_strategy(
        harmonic_ok=harmonic_ok,
        bpm_a=bpm_a,
        bpm_b=bpm_b,
        best_time=best_time,
        song_duration_a=duration_a,
        mix_duration=preferred_mix_duration
    )

    reason = (
        f"Selected {strategy} because Track A is {camelot_a}, "
        f"Track B is {camelot_b}, harmonic compatibility is {harmonic_ok}, "
        f"BPM delta is {abs(bpm_a - bpm_b):.2f}, and the best phrase-energy point is {best_time:.2f}s."
    )

    return {
        "status": "success",
        "recommended_transition_start_time": round(float(best_time), 3),
        "recommended_track_b_entry_time": round(float(track_b_entry_time), 3),
        "recommended_track_b_entry_sample": int(start_sample_b),
        "sync_accuracy": round(float(sync_accuracy), 3),
        "recommended_strategy": strategy,
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

# Utility to convert stereo audio to mono by averaging the channels.
def ensure_mono(y):
    if y.ndim > 1:
        return np.mean(y, axis=0)
    return y

# Ensures an audio array matches a specific length by either padding with silence or trimming the end.
def pad_or_trim(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

# Wraps librosa's beat tracking to safely return the estimated tempo and beat timestamps.
def safe_bpm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    tempo = float(np.asarray(tempo).squeeze())
    return tempo, beats.astype(int)

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

# Applies a standard Butterworth filter (Low-pass or High-pass) to a specific audio segment.
def apply_filter(y, sr, cutoff, btype, order=4):
    nyquist = 0.5 * sr
    cutoff = min(cutoff, nyquist - 100)
    normal_cutoff = cutoff / nyquist

    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype)
    return scipy.signal.filtfilt(b, a, y)

# Performs cross-correlation on sub-bass frequencies to perfectly align the waveforms of two tracks and prevent phase cancellation.
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

# Generates the mathematical curves for an equal-power crossfade to maintain consistent volume during a mix.
def equal_power_fades(n):
    t = np.linspace(0, 1, n)
    fade_out = np.cos(t * np.pi / 2)
    fade_in = np.sin(t * np.pi / 2)
    return fade_out, fade_in

# Calculates the Root Mean Square (RMS) of an audio signal to determine its average loudness.
def rms_level(y):
    return np.sqrt(np.mean(y ** 2) + 1e-9)

# Adjusts the volume of Track B to match the perceived loudness of Track A, preventing jarring volume jumps.
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

# Swaps the low-end frequencies of the tracks at the midpoint while crossfading the mids and highs.
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

# Progressively increases a high-pass filter cutoff on Track A to make it 'thin out' as Track B enters.
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

# A transition strategy that uses the dynamic high-pass filter sweep for a smooth blend.
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

# Creates an artificial 'runway' by looping the end of Track A if it is too short for the requested transition.
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

# Adds a basic exponential decay reverb to a signal to create a 'tail'.
def simple_reverb_tail(y, sr, decay_seconds=4.0, wet=0.45):
    decay_samples = int(decay_seconds * sr)

    impulse = np.exp(-np.linspace(0, 6, decay_samples))
    impulse = impulse / np.max(np.abs(impulse))

    reverb = scipy.signal.fftconvolve(y, impulse, mode="full")
    reverb = reverb[:len(y)]

    return (y * (1.0 - wet)) + (reverb * wet)

# Washes out Track A with heavy reverb during the crossfade, useful for non-harmonic or tempo-clashing transitions.
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

# An abrupt transition that cuts Track A and starts Track B at the midpoint with a very short crossfade.
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

# Estimates the musical key and mode (Major/Minor) of an audio track using chromagram analysis.
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

# Checks if two Camelot keys are compatible (adjacent on the circle of fifths or relative major/minor).
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

# A gentle blend that keeps both tracks musical, using subtle EQ adjustments instead of aggressive swapping.
def harmonic_mix_transition(segment_a, segment_b, sr):
    """
    Smooth harmonic blend:
    - Keeps both tracks musical
    - Avoids aggressive bass swap
    - Uses gentler EQ and equal-power fade
    """
    mix_samples = len(segment_a)

    fade_out, fade_in = equal_power_fades(mix_samples)

    a_bass = apply_filter(segment_a, sr, 180, "low")
    a_high = apply_filter(segment_a, sr, 180, "high")

    b_bass = apply_filter(segment_b, sr, 180, "low")
    b_high = apply_filter(segment_b, sr, 180, "high")

    # Bass is blended more gently than bass_swap
    bass_fade_out = np.linspace(1.0, 0.35, mix_samples)
    bass_fade_in = np.linspace(0.15, 1.0, mix_samples)

    mixed_bass = (a_bass * bass_fade_out) + (b_bass * bass_fade_in)
    mixed_high = (a_high * fade_out) + (b_high * fade_in)

    return mixed_bass * 0.75 + mixed_high * 0.95

# Finds the nearest musical phrase boundary (e.g., the start of a bar) to ensure the transition feels rhythmically natural.
def snap_to_phrase_boundary(beats, requested_time, sr, phrase_beats=32):
    """
    Snaps transition start to a musical phrase boundary.
    Default phrase_beats=32 = 8 bars in 4/4.
    """
    beat_times = librosa.samples_to_time(beats, sr=sr)

    if len(beat_times) == 0:
        return requested_time, 0

    closest_beat_idx = np.argmin(np.abs(beat_times - requested_time))

    phrase_idx = closest_beat_idx - (closest_beat_idx % phrase_beats)

    phrase_idx = max(0, min(phrase_idx, len(beat_times) - 1))

    snapped_phrase_time = float(beat_times[phrase_idx])
    snapped_phrase_sample = int(snapped_phrase_time * sr)

    return snapped_phrase_time, snapped_phrase_sample

# A phrase-aware transition that introduces Track B slowly and uses S-curve fades for a more 'musical' feel.
def phrase_mix_transition(segment_a, segment_b, sr):
    """
    Phrase-aware blend:
    - Introduces Track B slowly
    - Keeps Track A dominant in first half
    - Swaps energy more clearly in second half
    """
    mix_samples = len(segment_a)

    t = np.linspace(0, 1, mix_samples)

    # S-curve fades for more musical phrase movement
    fade_in = 1 / (1 + np.exp(-10 * (t - 0.55)))
    fade_out = 1 - (1 / (1 + np.exp(-10 * (t - 0.45))))

    a_bass = apply_filter(segment_a, sr, 220, "low")
    a_high = apply_filter(segment_a, sr, 220, "high")

    b_bass = apply_filter(segment_b, sr, 220, "low")
    b_high = apply_filter(segment_b, sr, 220, "high")

    # Track A bass stays longer, Track B bass enters later
    b_bass_gate = 1 / (1 + np.exp(-18 * (t - 0.70)))
    a_bass_gate = 1 - (1 / (1 + np.exp(-18 * (t - 0.65))))

    mixed_bass = (a_bass * a_bass_gate) + (b_bass * b_bass_gate)
    mixed_high = (a_high * fade_out) + (b_high * fade_in)

    return mixed_bass * 0.85 + mixed_high * 0.9

# Progressively decreases a low-pass filter cutoff to 'muffle' the audio over time.
def dynamic_lpf_sweep(y, sr, start_freq=18000, end_freq=300, steps=64):
    n = len(y)
    output = np.zeros_like(y)

    step_size = max(1, n // steps)
    freqs = np.linspace(start_freq, end_freq, steps)

    for i in range(steps):
        start = i * step_size
        end = n if i == steps - 1 else min(n, (i + 1) * step_size)

        if start >= n:
            break

        chunk = y[start:end]

        if len(chunk) < 32:
            output[start:end] = chunk
            continue

        try:
            output[start:end] = apply_filter(
                chunk,
                sr,
                cutoff=freqs[i],
                btype="low",
                order=2
            )
        except Exception:
            output[start:end] = chunk

    return output

# Mixes tracks by sweeping a low-pass filter on Track A.
def lpf_sweep_transition(segment_a, segment_b, sr, end_freq=300):
    mix_samples = len(segment_a)

    print("Applying LPF sweep transition...")

    fade_out, fade_in = equal_power_fades(mix_samples)

    swept_a = dynamic_lpf_sweep(
        y=segment_a,
        sr=sr,
        start_freq=18000,
        end_freq=end_freq,
        steps=64
    )

    mixed = (swept_a * fade_out) + (segment_b * fade_in)

    return mixed

# Adds a rhythmic feedback delay to a signal to create an echo effect.
def simple_echo(y, sr, delay_seconds=0.375, feedback=0.45, wet=0.5):
    delay_samples = int(delay_seconds * sr)

    output = y.copy()
    echo_buffer = np.zeros(len(y) + delay_samples * 4)
    echo_buffer[:len(y)] = y

    current_gain = feedback

    for i in range(1, 5):
        start = delay_samples * i
        end = start + len(y)

        echo_buffer[start:end] += y * current_gain
        current_gain *= feedback

    echo = echo_buffer[:len(y)]

    return (y * (1.0 - wet)) + (echo * wet)

# Applies an echo effect to Track A as it fades out to bridge the gap into Track B.
def echo_out_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Echo Out transition...")

    fade_out, fade_in = equal_power_fades(mix_samples)

    echo_region_samples = min(int(8 * sr), mix_samples)

    dry_a = segment_a.copy()
    echo_region = dry_a[-echo_region_samples:]

    echoed_tail = simple_echo(
        y=echo_region,
        sr=sr,
        delay_seconds=0.375,
        feedback=0.5,
        wet=0.65
    )

    processed_a = dry_a.copy()
    processed_a[-echo_region_samples:] = echoed_tail

    mixed = (processed_a * fade_out) + (segment_b * fade_in)

    return mixed

# Captures a small loop of Track A and repeats it rhythmically while fading out, mimicking a DJ 'roll' effect.
def loop_roll_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Loop Roll transition...")

    fade_out, fade_in = equal_power_fades(mix_samples)

    # Last 25% of the transition becomes a rhythmic roll
    roll_start = int(mix_samples * 0.75)

    processed_a = segment_a.copy()

    roll_source_len = int(0.5 * sr)  # 0.5 second loop
    roll_source_start = max(0, roll_start - roll_source_len)

    roll_source = segment_a[roll_source_start:roll_start]

    if len(roll_source) == 0:
        roll_source = segment_a[max(0, roll_start - int(0.25 * sr)):roll_start]

    roll_target_len = mix_samples - roll_start

    if len(roll_source) > 0 and roll_target_len > 0:
        repeats = int(np.ceil(roll_target_len / len(roll_source)))
        rolled = np.tile(roll_source, repeats)[:roll_target_len]

        # Roll fades out so it does not overpower Track B
        roll_fade = np.linspace(1.0, 0.15, roll_target_len)
        processed_a[roll_start:] = rolled * roll_fade

    mixed = (processed_a * fade_out) + (segment_b * fade_in)

    return mixed

# Simulates a long DJ blend by gradually moving individual EQ bands (Low, Mid, High) between tracks.
def long_eq_blend_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Long EQ Blend transition...")

    t = np.linspace(0, 1, mix_samples)

    # Split into 3 broad bands
    a_low = apply_filter(segment_a, sr, 180, "low")
    a_high = apply_filter(segment_a, sr, 2500, "high")
    a_mid = segment_a - a_low - a_high

    b_low = apply_filter(segment_b, sr, 180, "low")
    b_high = apply_filter(segment_b, sr, 2500, "high")
    b_mid = segment_b - b_low - b_high

    # Long DJ-style EQ movement
    a_low_gain = np.linspace(1.0, 0.0, mix_samples)
    b_low_gain = np.linspace(0.0, 1.0, mix_samples)

    a_mid_gain = 1.0 - (1 / (1 + np.exp(-8 * (t - 0.45))))
    b_mid_gain = 1 / (1 + np.exp(-8 * (t - 0.55)))

    a_high_gain = np.linspace(1.0, 0.2, mix_samples)
    b_high_gain = np.linspace(0.2, 1.0, mix_samples)

    mixed = (
        a_low * a_low_gain +
        b_low * b_low_gain +
        a_mid * a_mid_gain +
        b_mid * b_mid_gain +
        a_high * a_high_gain +
        b_high * b_high_gain
    )

    return mixed * 0.9

# A transition focused on smooth energy handoff, delaying the entry of Track B's bass to avoid low-end clutter.
def energy_blend_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Energy Blend transition...")

    t = np.linspace(0, 1, mix_samples)

    # Smooth energy handoff
    fade_out = 1 - (1 / (1 + np.exp(-9 * (t - 0.55))))
    fade_in = 1 / (1 + np.exp(-9 * (t - 0.45)))

    # Bass enters later to avoid low-end clutter
    a_low = apply_filter(segment_a, sr, 220, "low")
    a_high = apply_filter(segment_a, sr, 220, "high")

    b_low = apply_filter(segment_b, sr, 220, "low")
    b_high = apply_filter(segment_b, sr, 220, "high")

    a_low_gain = 1 - (1 / (1 + np.exp(-14 * (t - 0.62))))
    b_low_gain = 1 / (1 + np.exp(-14 * (t - 0.68)))

    mixed_low = (a_low * a_low_gain) + (b_low * b_low_gain)
    mixed_high = (a_high * fade_out) + (b_high * fade_in)

    return (mixed_low * 0.85) + (mixed_high * 0.9)

# Emphasizes the rhythmic elements and percussion bands during the blend for a groove-heavy transition.
def percussion_blend_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Percussion Blend transition...")

    t = np.linspace(0, 1, mix_samples)

    # Broad rhythm/percussion band emphasis
    # Most kick/body sits low, hats/claps/transients sit high.
    a_low = apply_filter(segment_a, sr, 180, "low")
    a_high = apply_filter(segment_a, sr, 1800, "high")
    a_mid = segment_a - a_low - a_high

    b_low = apply_filter(segment_b, sr, 180, "low")
    b_high = apply_filter(segment_b, sr, 1800, "high")
    b_mid = segment_b - b_low - b_high

    # Keep Track A groove first, introduce Track B percussion early
    a_low_gain = 1 - (1 / (1 + np.exp(-14 * (t - 0.60))))
    b_low_gain = 1 / (1 + np.exp(-14 * (t - 0.72)))

    a_mid_gain = np.linspace(1.0, 0.35, mix_samples)
    b_mid_gain = np.linspace(0.25, 1.0, mix_samples)

    # Percussion/highs enter earlier than bass
    a_high_gain = np.linspace(0.9, 0.25, mix_samples)
    b_high_gain = np.linspace(0.35, 1.0, mix_samples)

    mixed = (
        a_low * a_low_gain +
        b_low * b_low_gain +
        a_mid * a_mid_gain +
        b_mid * b_mid_gain +
        a_high * a_high_gain +
        b_high * b_high_gain
    )

    return mixed * 0.88

# A softer blend designed for breakdowns, prioritizing atmospheric mids/highs over heavy bass.
def breakdown_blend_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Breakdown Blend transition...")

    t = np.linspace(0, 1, mix_samples)

    # Softer, emotional blend. Less bass dominance, more mids/high atmosphere.
    a_low = apply_filter(segment_a, sr, 160, "low")
    a_high = apply_filter(segment_a, sr, 160, "high")

    b_low = apply_filter(segment_b, sr, 160, "low")
    b_high = apply_filter(segment_b, sr, 160, "high")

    # Track A gently dissolves
    a_high_gain = 1 - (1 / (1 + np.exp(-7 * (t - 0.55))))
    b_high_gain = 1 / (1 + np.exp(-7 * (t - 0.35)))

    # Bass enters very late, because breakdowns usually need space
    a_low_gain = 1 - (1 / (1 + np.exp(-14 * (t - 0.48))))
    b_low_gain = 1 / (1 + np.exp(-14 * (t - 0.78)))

    mixed_low = (a_low * a_low_gain) + (b_low * b_low_gain)
    mixed_high = (a_high * a_high_gain) + (b_high * b_high_gain)

    return mixed_low * 0.75 + mixed_high * 0.95

# Removes the low-end and adds a long reverb tail to Track A, turning it into an atmospheric backdrop for Track B.
def ambient_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Ambient Transition...")

    t = np.linspace(0, 1, mix_samples)

    fade_out, fade_in = equal_power_fades(mix_samples)

    # Remove heavy low-end from Track A so it becomes atmospheric
    a_air = apply_filter(segment_a, sr, 350, "high", order=2)

    # Let Track B enter softly, with low-end delayed
    b_low = apply_filter(segment_b, sr, 220, "low")
    b_air = apply_filter(segment_b, sr, 220, "high")

    b_low_gain = 1 / (1 + np.exp(-14 * (t - 0.75)))

    # Add reverb texture to Track A
    washed_a = simple_reverb_tail(
        y=a_air,
        sr=sr,
        decay_seconds=7.0,
        wet=0.55
    )

    mixed = (
        washed_a * fade_out * 0.85
        + b_air * fade_in * 0.9
        + b_low * b_low_gain * 0.75
    )

    return mixed

# Applies tanh-based soft saturation to a signal to add harmonic warmth or 'drive'.
def soft_clip_drive(y, drive=2.0):
    """
    Soft saturation/drive without harsh digital clipping.
    """
    driven = np.tanh(y * drive)
    return driven / max(np.max(np.abs(driven)), 1e-9)

# A high-energy transition that adds saturation and an aggressive HPF sweep to Track A.
def techno_filter_drive_transition(segment_a, segment_b, sr):
    mix_samples = len(segment_a)

    print("Applying Techno Filter Drive transition...")

    t = np.linspace(0, 1, mix_samples)

    fade_out, fade_in = equal_power_fades(mix_samples)

    # Drive Track A as it exits
    driven_a = soft_clip_drive(segment_a, drive=2.2)

    # HPF sweep makes Track A thinner/aggressive over time
    filtered_a = dynamic_hpf_sweep(
        y=driven_a,
        sr=sr,
        start_freq=80,
        end_freq=3500,
        steps=64
    )

    # Track B enters clean, then gets full energy
    b_low = apply_filter(segment_b, sr, 220, "low")
    b_high = apply_filter(segment_b, sr, 220, "high")

    b_low_gain = 1 / (1 + np.exp(-16 * (t - 0.68)))

    mixed = (
        filtered_a * fade_out * 0.85
        + b_high * fade_in * 0.9
        + b_low * b_low_gain * 0.9
    )

    return mixed

# A router function that maps a strategy name to its corresponding transition implementation.
def apply_transition_strategy(segment_a, segment_b, sr, transition_strategy, fx_parameters=None):
    if transition_strategy == "bass_swap":
        return bass_swap_transition(segment_a, segment_b, sr)

    if transition_strategy == "harmonic_mix":
        return harmonic_mix_transition(segment_a, segment_b, sr)

    if transition_strategy == "phrase_mix":
        return phrase_mix_transition(segment_a, segment_b, sr)

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
    if transition_strategy == "lpf_sweep":
        return lpf_sweep_transition(
            segment_a=segment_a,
            segment_b=segment_b,
            sr=sr,
            end_freq=300
        )

    if transition_strategy == "auto_loop":
        return bass_swap_transition(segment_a, segment_b, sr)

    if transition_strategy == "reverb_wash":
        return reverb_wash_transition(segment_a, segment_b, sr)

    if transition_strategy == "drop_mix":
        return drop_mix_transition(segment_a, segment_b, sr)
    
    if transition_strategy == "echo_out":
        return echo_out_transition(segment_a, segment_b, sr)
    
    if transition_strategy == "loop_roll":
        return loop_roll_transition(segment_a, segment_b, sr)
    
    if transition_strategy == "long_eq_blend":
        return long_eq_blend_transition(segment_a, segment_b, sr)
    
    if transition_strategy == "energy_blend":
        return energy_blend_transition(segment_a, segment_b, sr)

    if transition_strategy == "percussion_blend":
        return percussion_blend_transition(segment_a, segment_b, sr)
    
    if transition_strategy == "breakdown_blend":
        return breakdown_blend_transition(segment_a, segment_b, sr)
    
    if transition_strategy == "ambient_transition":
        return ambient_transition(segment_a, segment_b, sr)
    
    if transition_strategy == "techno_filter_drive":
        return techno_filter_drive_transition(segment_a, segment_b, sr) 

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported transition_strategy: {transition_strategy}"
    )

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

# FastAPI endpoint that combines 'Plan' and 'Render' into a single automated 'One-Click' transition request.
@app.post("/v1/autodj/render-planned-transition")
async def render_planned_transition(req: AutoRenderRequest):
    plan = plan_transition_logic(
        track_a_path=req.track_a_path,
        track_b_path=req.track_b_path,
        preferred_mix_duration=req.preferred_mix_duration
    )

    payload = plan["render_payload"]

    result = render_dj_transition(
        track_a_path=payload["track_a_path"],
        track_b_path=payload["track_b_path"],
        transition_start_time=payload["transition_start_time"],
        track_b_entry_time=payload.get("track_b_entry_time"),
        mix_duration=payload["mix_duration"],
        output_dir=req.output_dir,
        transition_strategy=payload["transition_strategy"],
        fx_parameters=FXParameters(**payload["fx_parameters"])
    )

    return {
        "status": "success",
        "plan": plan,
        "render": result
    }