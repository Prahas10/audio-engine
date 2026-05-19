import numpy as np
import scipy.signal
import librosa
from core.analysis import rms_level

EPS = 1e-9

# ---------------------------------------------------------------------------
# Crossover frequencies — standardised across all transitions
# ---------------------------------------------------------------------------
XO_SUB  =  80.0   # sub-bass / kick body boundary
XO_LOW  = 200.0   # kick body / upper bass
XO_MID  = 2500.0  # upper bass / mids
XO_HIGH = 8000.0  # mids / air


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def pad_or_trim(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]


def remove_dc(y):
    return y - np.mean(y)


def apply_filter(y, sr, cutoff, btype, order=4):
    """Zero-phase Butterworth via SOS — stable at all orders and cutoffs."""
    nyquist = 0.5 * sr
    cutoff  = float(np.clip(cutoff, 20.0, nyquist - 100.0))
    sos     = scipy.signal.butter(order, cutoff / nyquist, btype=btype, output="sos")
    return scipy.signal.sosfiltfilt(sos, y)


def equal_power_fades(n):
    """Constant-power (cos/sin) crossfade — maintains perceived loudness."""
    t = np.linspace(0.0, 1.0, n)
    return np.cos(t * np.pi / 2.0), np.sin(t * np.pi / 2.0)


def hann_crossfade(n):
    """Raised-cosine crossfade for short anti-click edits."""
    t      = np.linspace(0.0, 1.0, n)
    fade_in = 0.5 * (1.0 - np.cos(np.pi * t))
    return fade_in[::-1], fade_in


def sigmoid(n, center=0.5, steepness=10.0):
    t = np.linspace(0.0, 1.0, n)
    return 1.0 / (1.0 + np.exp(-steepness * (t - center)))


def soft_limiter(y, ceiling=0.985):
    """Tanh soft limiter — transparent below ceiling, smooth above."""
    threshold = ceiling - 0.05
    abs_y     = np.abs(y)
    sign      = np.sign(y)
    headroom  = ceiling - threshold
    excess    = np.maximum(abs_y - threshold, 0.0)
    shaped    = headroom * np.tanh(excess / max(headroom, EPS))
    return sign * (np.minimum(abs_y, threshold) + shaped)


def _finalize(y):
    """Always strip DC and soft-limit before returning from a transition."""
    return soft_limiter(remove_dc(y))


def split_bands(y, sr):
    """
    Split a signal into 3 bands: low (≤XO_LOW), mid (XO_LOW–XO_MID), high (≥XO_MID).
    Mid is computed by subtraction to avoid phase issues from cascaded filters.
    """
    low  = apply_filter(y, sr, XO_LOW, "low")
    high = apply_filter(y, sr, XO_MID, "high")
    mid  = y - low - high
    return low, mid, high


# ---------------------------------------------------------------------------
# Phase alignment
# ---------------------------------------------------------------------------

def phase_align(track_a_slice, track_b_slice, sr):
    """
    Cross-correlates sub-bass to find sample-accurate shift and polarity.
    """
    slice_len = int(0.05 * sr)
    a = pad_or_trim(track_a_slice, slice_len)
    b = pad_or_trim(track_b_slice, slice_len)
    try:
        a_sub = apply_filter(a, sr, 100.0, "low", order=2)
        b_sub = apply_filter(b, sr, 100.0, "low", order=2)
        corr  = scipy.signal.correlate(a_sub, b_sub, mode="full")
        idx   = int(np.argmax(np.abs(corr)))
        shift = idx - (len(b_sub) - 1)
        return shift, bool(corr[idx] < 0.0)
    except Exception:
        return 0, False


# ---------------------------------------------------------------------------
# Loudness
# ---------------------------------------------------------------------------

def match_rms_to_reference(y, reference_y, max_gain_db=6.0):
    ref_rms  = max(rms_level(reference_y), EPS)
    y_rms    = max(rms_level(y), EPS)
    gain     = ref_rms / y_rms
    max_gain = 10 ** (max_gain_db / 20.0)
    gain     = float(np.clip(gain, 1.0 / max_gain, max_gain))
    return y * gain, gain


# ---------------------------------------------------------------------------
# Reverb / echo
# ---------------------------------------------------------------------------

def simple_reverb_tail(y, sr, decay_seconds=4.0, wet=0.45):
    decay_samples = max(int(decay_seconds * sr), 64)
    rng           = np.random.default_rng(0xC0FFEE)
    noise         = rng.standard_normal(decay_samples)
    sos           = scipy.signal.butter(2, 5000.0 / (sr / 2.0), btype="low", output="sos")
    noise         = scipy.signal.sosfilt(sos, noise)
    t             = np.arange(decay_samples) / sr
    impulse       = noise * np.exp(-t * (6.91 / decay_seconds))
    peak          = np.max(np.abs(impulse))
    if peak > EPS:
        impulse /= peak
    reverb    = scipy.signal.fftconvolve(y, impulse, mode="full")[:len(y)]
    dry_rms   = max(rms_level(y), EPS)
    wet_rms   = max(rms_level(reverb), EPS)
    reverb   *= dry_rms / wet_rms
    return (y * (1.0 - wet)) + (reverb * wet)


def tempo_synced_echo(y, sr, bpm, beats=1, feedback=0.45, wet=0.5):
    """Echo delay locked to the track's BPM."""
    delay_samples = int((60.0 / bpm) * beats * sr)
    if delay_samples < 1:
        return y
    buf = np.zeros(len(y) + delay_samples * 6)
    buf[:len(y)] = y
    gain = feedback
    for i in range(1, 6):
        start = delay_samples * i
        end   = start + len(y)
        if end > len(buf):
            break
        buf[start:end] += y * gain
        gain *= feedback
        if gain < 1e-4:
            break
    return (y * (1.0 - wet)) + (buf[:len(y)] * wet)

def vocal_band_duck(y, sr, duck_amount=0.45):
    """
    Ducks vocal/mid band around 300–3400 Hz.
    Used to prevent vocal-vocal clashes during transitions.
    """
    low = apply_filter(y, sr, 300.0, "low", order=3)
    high = apply_filter(y, sr, 3400.0, "high", order=3)
    mid = y - low - high

    return low + (mid * duck_amount) + high
# ---------------------------------------------------------------------------
# Filter sweeps (STFT-domain — no stepping artefacts)
# ---------------------------------------------------------------------------

def _stft_sweep(y, sr, freq_curve, btype="high", order=4):
    """
    Continuous STFT-domain time-varying filter. Builds a per-frame Butterworth
    magnitude mask — smooth in both time and frequency, zero zipper noise.
    """
    n     = len(y)
    n_fft = 2048
    hop   = 512
    pad   = max(0, n_fft - n)
    y_in  = np.pad(y, (0, pad)) if pad else y

    spec  = librosa.stft(y_in, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = np.where(freqs <= 0, 1e-3, freqs)

    n_frames = spec.shape[1]
    curve    = np.asarray(freq_curve, dtype=np.float64)
    x_in     = np.linspace(0.0, 1.0, curve.size)
    x_out    = np.linspace(0.0, 1.0, n_frames)
    cutoff   = np.clip(np.interp(x_out, x_in, curve), 20.0, sr / 2.0 - 100.0)

    p2 = 2 * order
    for i in range(n_frames):
        ratio = freqs / cutoff[i]
        rp    = ratio ** p2
        mask  = (rp / (1.0 + rp)) if btype == "high" else (1.0 / (1.0 + rp))
        spec[:, i] *= mask

    out = librosa.istft(spec, hop_length=hop, length=n + pad)
    return out[:n]


def dynamic_hpf_sweep(y, sr, start_freq=20.0, end_freq=3500.0):
    curve = np.linspace(float(start_freq), float(end_freq), 256)
    return _stft_sweep(y, sr, curve, btype="high")


def dynamic_lpf_sweep(y, sr, start_freq=18000.0, end_freq=400.0):
    curve = np.linspace(float(start_freq), float(end_freq), 256)
    return _stft_sweep(y, sr, curve, btype="low")


# ---------------------------------------------------------------------------
# Snap / loop helpers
# ---------------------------------------------------------------------------

def snap_to_phrase_boundary(beats, requested_time, sr, phrase_beats=32):
    beat_times = librosa.samples_to_time(beats, sr=sr)
    if len(beat_times) == 0:
        return float(requested_time), int(requested_time * sr)
    closest   = int(np.argmin(np.abs(beat_times - requested_time)))
    idx       = int(np.clip(closest - (closest % phrase_beats), 0, len(beat_times) - 1))
    snapped_t = float(beat_times[idx])
    return snapped_t, int(snapped_t * sr)


def auto_loop_track_a_segment(y_a, start_sample_a, mix_samples, sr):
    available = y_a[start_sample_a:]
    if len(available) >= mix_samples:
        return available[:mix_samples]
    print("Track A short — auto-loop runway.")
    min_loop    = int(4 * sr)
    loop_source = available if len(available) >= min_loop else y_a[max(0, len(y_a) - min_loop):]
    if len(loop_source) == 0:
        return np.zeros(mix_samples)
    edge = max(1, int(0.02 * sr))
    if len(loop_source) > 2 * edge:
        loop_source = loop_source.copy()
        loop_source[:edge]  *= np.linspace(0.0, 1.0, edge)
        loop_source[-edge:] *= np.linspace(1.0, 0.0, edge)
    repeats = int(np.ceil(mix_samples / len(loop_source)))
    return np.tile(loop_source, repeats)[:mix_samples]


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def soft_clip_drive(y, drive=2.0):
    driven = np.tanh(y * drive)
    peak   = np.max(np.abs(driven))
    return driven / max(peak, EPS)


# ---------------------------------------------------------------------------
#
# TRANSITIONS
#
# Every transition follows the same structure:
#   1. Split into frequency bands
#   2. Apply band-specific gain curves (DJ technique)
#   3. Sum and _finalize() (DC strip + soft limit)
#
# The key principle: NEVER simply crossfade the full signal.
# A DJ always manages bass separately from mids/highs.
#
# ---------------------------------------------------------------------------

def bass_swap_transition(segment_a, segment_b, sr):
    """
    Classic DJ bass swap.
    - Mids/highs: equal-power crossfade over the full window
    - Bass: hard swap at midpoint with a 60ms Hann crossfade
    This is the most common technique in house/techno — keeps exactly
    one kick in the mix at all times.
    """
    n = len(segment_a)

    a_low, a_mid, a_high = split_bands(segment_a, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    fout, fin = equal_power_fades(n)

    # Bass swap at midpoint
    mid       = n // 2
    xfade     = min(int(0.060 * sr), mid)
    s, e      = max(0, mid - xfade // 2), min(n, mid + xfade // 2)
    bfo, bfi  = hann_crossfade(e - s)

    bass = np.empty(n)
    bass[:s]  = a_low[:s]
    bass[s:e] = a_low[s:e] * bfo + b_low[s:e] * bfi
    bass[e:]  = b_low[e:]

    mids = (a_mid + a_high) * fout + (b_mid + b_high) * fin

    return _finalize(bass * 0.95 + mids * 0.90)


def long_eq_blend_transition(segment_a, segment_b, sr):
    """
    Long DJ EQ blend — each band moves on a different schedule.
    Low: linear over full duration
    Mid: S-curve crossing at centre
    High: equal-power crossfade
    This mimics a DJ turning the EQ knobs gradually over 2–4 minutes.
    """
    n = len(segment_a)

    a_low, a_mid, a_high = split_bands(segment_a, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    fout, fin = equal_power_fades(n)

    a_low_g  = np.linspace(1.0, 0.0, n)
    b_low_g  = np.linspace(0.0, 1.0, n)
    a_mid_g  = 1.0 - sigmoid(n, center=0.48, steepness=9.0)
    b_mid_g  = sigmoid(n, center=0.52, steepness=9.0)

    mixed = (
        a_low * a_low_g + b_low * b_low_g +
        a_mid * a_mid_g + b_mid * b_mid_g +
        a_high * fout   + b_high * fin
    )
    return _finalize(mixed * 0.92)


def harmonic_mix_transition(segment_a, segment_b, sr):
    """
    Gentle blend for harmonically compatible tracks.
    Bass crosses with a slight delay so there's never a full double-kick.
    """
    n = len(segment_a)

    a_low, a_mid, a_high = split_bands(segment_a, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    fout, fin = equal_power_fades(n)

    a_bass_g = 1.0 - sigmoid(n, center=0.55, steepness=10.0)
    b_bass_g = sigmoid(n, center=0.60, steepness=10.0)

    mixed_low  = a_low * a_bass_g + b_low * b_bass_g
    mixed_high = (a_mid + a_high) * fout + (b_mid + b_high) * fin

    return _finalize(mixed_low * 0.88 + mixed_high * 0.92)


def phrase_mix_transition(segment_a, segment_b, sr):
    """
    Phrase-aware blend. Track A dominates the first half; Track B takes
    over in the second half. Bass gating prevents double-kick.
    """
    n = len(segment_a)

    a_low, a_mid, a_high = split_bands(segment_a, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    fade_in  = sigmoid(n, center=0.55, steepness=10.0)
    fade_out = 1.0 - sigmoid(n, center=0.45, steepness=10.0)

    # Bass gate gap prevents simultaneous kick from both tracks
    a_bass_g = 1.0 - sigmoid(n, center=0.58, steepness=18.0)
    b_bass_g = sigmoid(n, center=0.63, steepness=18.0)

    mixed_low  = a_low * a_bass_g + b_low * b_bass_g
    mixed_high = (a_mid + a_high) * fade_out + (b_mid + b_high) * fade_in

    return _finalize(mixed_low * 0.88 + mixed_high * 0.92)


def energy_blend_transition(segment_a, segment_b, sr):
    """
    Smooth energy handoff. Bass enters late to avoid low-end clutter.
    Good for tracks where BPM matches but keys don't perfectly.
    """
    n = len(segment_a)

    a_low, a_mid, a_high = split_bands(segment_a, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    fout = 1.0 - sigmoid(n, center=0.55, steepness=9.0)
    fin  = sigmoid(n, center=0.45, steepness=9.0)

    a_bass_g = 1.0 - sigmoid(n, center=0.60, steepness=14.0)
    b_bass_g = sigmoid(n, center=0.68, steepness=14.0)

    mixed_low  = a_low * a_bass_g + b_low * b_bass_g
    mixed_high = (a_mid + a_high) * fout + (b_mid + b_high) * fin

    return _finalize(mixed_low * 0.87 + mixed_high * 0.92)


def percussion_blend_transition(segment_a, segment_b, sr):
    """
    Groove-forward blend. Percussion/highs cross early; bass crosses late.
    Good for tracks with complementary rhythmic textures.
    """
    n = len(segment_a)

    a_low, a_mid, a_high = split_bands(segment_a, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    # Highs enter early (rhythmic texture handover)
    a_high_g = 1.0 - sigmoid(n, center=0.40, steepness=8.0)
    b_high_g = sigmoid(n, center=0.40, steepness=8.0)
    a_mid_g  = 1.0 - sigmoid(n, center=0.50, steepness=8.0)
    b_mid_g  = sigmoid(n, center=0.50, steepness=8.0)

    # Bass enters late
    a_bass_g = 1.0 - sigmoid(n, center=0.60, steepness=14.0)
    b_bass_g = sigmoid(n, center=0.72, steepness=14.0)

    mixed = (
        a_low * a_bass_g + b_low * b_bass_g +
        a_mid * a_mid_g  + b_mid * b_mid_g  +
        a_high * a_high_g + b_high * b_high_g
    )
    return _finalize(mixed * 0.90)


def breakdown_blend_transition(segment_a, segment_b, sr):
    """
    Soft atmospheric blend for breakdown sections. Very delayed bass entry
    leaves space for the mix to breathe.
    """
    n = len(segment_a)

    a_low, a_mid, a_high = split_bands(segment_a, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    a_high_g = 1.0 - sigmoid(n, center=0.55, steepness=6.0)
    b_high_g = sigmoid(n, center=0.40, steepness=6.0)
    a_bass_g = 1.0 - sigmoid(n, center=0.48, steepness=14.0)
    b_bass_g = sigmoid(n, center=0.78, steepness=14.0)

    mixed_low  = a_low * a_bass_g + b_low * b_bass_g
    mixed_high = (a_mid + a_high) * a_high_g + (b_mid + b_high) * b_high_g

    return _finalize(mixed_low * 0.78 + mixed_high * 0.95)


def hpf_sweep_transition(segment_a, segment_b, sr, end_freq=3500.0):
    """
    Rising HPF on Track A makes it progressively thinner as B enters.
    Track B's bass enters with a delayed gate so low-end only arrives once A is thin.
    """
    n = len(segment_a)

    swept_a = dynamic_hpf_sweep(segment_a, sr, start_freq=20.0, end_freq=end_freq)
    fout, fin = equal_power_fades(n)

    b_low    = apply_filter(segment_b, sr, XO_LOW, "low")
    b_high   = apply_filter(segment_b, sr, XO_LOW, "high")
    b_bass_g = sigmoid(n, center=0.65, steepness=14.0)

    mixed = swept_a * fout + b_high * fin * 0.95 + b_low * b_bass_g * fin * 0.90
    return _finalize(mixed)


def lpf_sweep_transition(segment_a, segment_b, sr, end_freq=400.0):
    """
    Falling LPF on Track A muffles it progressively as B enters.
    B's highs enter before its bass for a natural DJ intro feel.
    """
    n = len(segment_a)

    swept_a = dynamic_lpf_sweep(segment_a, sr, start_freq=18000.0, end_freq=end_freq)
    fout, fin = equal_power_fades(n)

    b_low    = apply_filter(segment_b, sr, XO_LOW, "low")
    b_high   = apply_filter(segment_b, sr, XO_LOW, "high")
    b_bass_g = sigmoid(n, center=0.70, steepness=14.0)

    mixed = swept_a * fout + b_high * fin * 0.95 + b_low * b_bass_g * fin * 0.90
    return _finalize(mixed)


def reverb_wash_transition(segment_a, segment_b, sr):
    """
    Washes out Track A with reverb, useful for non-harmonic transitions.
    Reverb only on last 40% of A to preserve groove up to that point.
    """
    n = len(segment_a)

    fout, fin = equal_power_fades(n)
    tail      = int(0.40 * n)
    washed_a  = segment_a.copy()
    washed_a[-tail:] = simple_reverb_tail(segment_a[-tail:], sr, decay_seconds=4.5, wet=0.50)

    return _finalize(washed_a * fout + segment_b * fin)


def echo_out_transition(segment_a, segment_b, sr, bpm=128.0):
    """
    Rhythmic echo on Track A's tail, tempo-synced to actual BPM.
    """
    n = len(segment_a)

    fout, fin  = equal_power_fades(n)
    beat_samps = int((60.0 / bpm) * sr)
    region     = min(8 * beat_samps, n)

    processed      = segment_a.copy()
    processed[-region:] = tempo_synced_echo(
        segment_a[-region:], sr, bpm=bpm, beats=1, feedback=0.45, wet=0.55
    )
    return _finalize(processed * fout + segment_b * fin)


def loop_roll_transition(segment_a, segment_b, sr, bpm=128.0):
    """
    Beat-synced loop roll on the last 25% of Track A.
    Loop length = 1 beat, tempo-derived so the roll is always on-grid.
    """
    n = len(segment_a)

    fout, fin  = equal_power_fades(n)
    roll_start = int(n * 0.75)

    beat_samples   = int((60.0 / bpm) * sr)
    src_start      = max(0, roll_start - beat_samples)
    source         = segment_a[src_start:roll_start]
    target_len     = n - roll_start

    processed = segment_a.copy()
    if len(source) > 0 and target_len > 0:
        edge = max(1, int(0.002 * sr))
        if len(source) > 2 * edge:
            source = source.copy()
            source[:edge]  *= np.linspace(0.0, 1.0, edge)
            source[-edge:] *= np.linspace(1.0, 0.0, edge)
        repeats = int(np.ceil(target_len / len(source)))
        rolled  = np.tile(source, repeats)[:target_len]
        roll_env = np.exp(-3.0 * np.linspace(0.0, 1.0, target_len))
        processed[roll_start:] = rolled * roll_env

    return _finalize(processed * fout + segment_b * fin)


def drop_mix_transition(segment_a, segment_b, sr, bpm=128.0):
    """
    Hard cut at a beat boundary with a 10ms Hann crossfade to prevent clicks.
    Cut point is aligned to the nearest beat for musical accuracy.
    """
    n = len(segment_a)

    beat_samples = int((60.0 / bpm) * sr)
    mid          = n // 2
    cut          = (mid // beat_samples) * beat_samples
    cut          = int(np.clip(cut, 0, n - 1))

    xfade = min(int(0.010 * sr), cut, n - cut)
    if xfade < 2:
        out = np.concatenate([segment_a[:cut], segment_b[cut:]])
        return _finalize(out)

    s, e      = cut - xfade // 2, cut + xfade // 2
    bfo, bfi  = hann_crossfade(e - s)

    out = np.empty(n)
    out[:s]  = segment_a[:s]
    out[s:e] = segment_a[s:e] * bfo + segment_b[s:e] * bfi
    out[e:]  = segment_b[e:]
    return _finalize(out)


def ambient_transition(segment_a, segment_b, sr):
    """
    Strips A's low-end and washes in reverb. B's bass enters very late.
    Spatial and atmospheric — good for ambient/progressive drops.
    """
    n = len(segment_a)

    fout, fin = equal_power_fades(n)

    a_air    = apply_filter(segment_a, sr, 400.0, "high", order=2)
    b_low    = apply_filter(segment_b, sr, XO_LOW, "low")
    b_air    = apply_filter(segment_b, sr, XO_LOW, "high")
    b_bass_g = sigmoid(n, center=0.78, steepness=14.0)

    washed_a = simple_reverb_tail(a_air, sr, decay_seconds=5.5, wet=0.48)

    mixed = washed_a * fout * 0.88 + b_air * fin * 0.90 + b_low * b_bass_g * 0.75
    return _finalize(mixed)


def techno_filter_drive_transition(segment_a, segment_b, sr):
    """
    Saturates and HPF-sweeps Track A aggressively. B enters clean underneath.
    High-energy — suited for techno, hardstyle, peak-hour moments.
    """
    n = len(segment_a)

    fout, fin = equal_power_fades(n)

    driven_a   = soft_clip_drive(segment_a, drive=1.8)
    filtered_a = dynamic_hpf_sweep(driven_a, sr, start_freq=80.0, end_freq=2800.0)

    b_low    = apply_filter(segment_b, sr, XO_LOW, "low")
    b_high   = apply_filter(segment_b, sr, XO_LOW, "high")
    b_bass_g = sigmoid(n, center=0.70, steepness=16.0)

    mixed = filtered_a * fout * 0.82 + b_high * fin * 0.92 + b_low * b_bass_g * 0.90
    return _finalize(mixed)

def vocal_safe_blend_transition(segment_a, segment_b, sr):
    """
    DJ-safe vocal-aware blend.
    Ducks Track A vocal/mid band while letting Track B take focus.
    Avoids two full vocal bands overlapping.
    """
    n = len(segment_a)

    a_ducked = vocal_band_duck(segment_a, sr, duck_amount=0.35)

    a_low, a_mid, a_high = split_bands(a_ducked, sr)
    b_low, b_mid, b_high = split_bands(segment_b, sr)

    a_out = 1.0 - sigmoid(n, center=0.42, steepness=10.0)
    b_in = sigmoid(n, center=0.50, steepness=10.0)

    a_bass_g = 1.0 - sigmoid(n, center=0.52, steepness=16.0)
    b_bass_g = sigmoid(n, center=0.66, steepness=16.0)

    mixed = (
        a_low * a_bass_g +
        b_low * b_bass_g +
        (a_mid + a_high) * a_out * 0.72 +
        (b_mid + b_high) * b_in
    )

    return _finalize(mixed * 0.92)