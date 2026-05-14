import numpy as np
import scipy.signal
import librosa
from core.analysis import rms_level

EPS = 1e-9
 
# Canonical 3-band crossover points.
XO_LOW = 180.0
XO_MID = 2500.0
 
 
# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
 
def pad_or_trim(y, target_len):
    """Pad with zeros or trim the end so len(y) == target_len."""
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]
 
 
def remove_dc(y):
    """Strip DC offset; prevents asymmetric clipping later in the chain."""
    return y - np.mean(y)
 
 
def equal_power_fades(n):
    """Constant-power crossfade curves (cos / sin)."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.cos(t * np.pi / 2.0), np.sin(t * np.pi / 2.0)
 
 
def hann_fades(n):
    """Symmetric raised-cosine fades for short anti-click crossfades."""
    if n <= 0:
        return np.array([]), np.array([])
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    fade_in = 0.5 * (1.0 - np.cos(np.pi * t))
    return fade_in[::-1], fade_in
 
 
def s_curve(n, midpoint=0.5, steepness=10.0):
    """Logistic S-curve; smoother than linear for musical fades."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))
 
 
def soft_limiter(y, ceiling=0.985, knee=0.05):
    """
    Tanh soft limiter with a small knee. Transparent below (ceiling - knee),
    smoothly compresses peaks above. Prevents clipping when buses are summed.
    """
    if y.size == 0:
        return y
    threshold = ceiling - knee
    abs_y = np.abs(y)
    sign = np.sign(y)
    headroom = ceiling - threshold
    excess = np.maximum(abs_y - threshold, 0.0)
    shaped = headroom * np.tanh(excess / max(headroom, EPS))
    return sign * (np.minimum(abs_y, threshold) + shaped)
 
 
def _finalize(y):
    """End-of-transition pass: DC strip + soft limit. Always called last."""
    return soft_limiter(remove_dc(y))
 
 
# ---------------------------------------------------------------------------
# Filtering -- static and time-varying
# ---------------------------------------------------------------------------
 
def apply_filter(y, sr, cutoff, btype, order=4):
    """Zero-phase Butterworth filter on the whole signal (SOS for stability)."""
    nyquist = 0.5 * sr
    cutoff = float(np.clip(cutoff, 20.0, nyquist - 100.0))
    sos = scipy.signal.butter(order, cutoff / nyquist, btype=btype, output="sos")
    return scipy.signal.sosfiltfilt(sos, y)
 
 
def _stft_sweep(y, sr, freq_curve, btype="low", slope_order=4):
    """
    STFT-domain time-varying filter. Builds a per-frame magnitude mask shaped
    like a Butterworth response of order `slope_order` and applies it. The
    response is continuous in both time and frequency, so the sweep has no
    boundary clicks no matter how the curve is shaped.
    """
    n = len(y)
    if n == 0:
        return y.copy()
 
    n_fft = 2048
    hop = 512
    if n < n_fft:
        pad = n_fft - n
        y_in = np.pad(y, (0, pad))
    else:
        pad = 0
        y_in = y
 
    spec = librosa.stft(y_in, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = np.where(freqs <= 0, 1e-3, freqs)  # avoid div-by-zero at DC bin
 
    n_frames = spec.shape[1]
    curve = np.asarray(freq_curve, dtype=np.float64)
    if curve.size != n_frames:
        x_in = np.linspace(0.0, 1.0, curve.size)
        x_out = np.linspace(0.0, 1.0, n_frames)
        cutoff = np.interp(x_out, x_in, curve)
    else:
        cutoff = curve
    cutoff = np.clip(cutoff, 20.0, sr / 2 - 100.0)
 
    p2 = 2 * slope_order  # squared response exponent for Butterworth shape
    for i in range(n_frames):
        ratio = freqs / cutoff[i]
        ratio_p = ratio ** p2
        denom = 1.0 + ratio_p
        if btype == "low":
            mask = 1.0 / denom
        else:
            mask = ratio_p / denom
        spec[:, i] *= mask
 
    out = librosa.istft(spec, hop_length=hop, length=n + pad)
    return out[:n]
 
 
def dynamic_hpf_sweep(y, sr, start_freq=20.0, end_freq=5000.0, steps=64):
    """Smooth high-pass sweep from start_freq to end_freq across the signal."""
    curve = np.linspace(float(start_freq), float(end_freq), max(int(steps), 32))
    return _stft_sweep(y, sr, curve, btype="high")
 
 
def dynamic_lpf_sweep(y, sr, start_freq=18000.0, end_freq=300.0, steps=64):
    """Smooth low-pass sweep from start_freq to end_freq across the signal."""
    curve = np.linspace(float(start_freq), float(end_freq), max(int(steps), 32))
    return _stft_sweep(y, sr, curve, btype="low")
 
 
# ---------------------------------------------------------------------------
# Phase alignment
# ---------------------------------------------------------------------------
 
def phase_align(track_a_slice, track_b_slice, sr):
    """
    Cross-correlate sub-bass energy to find the integer-sample shift and
    polarity that best align Track B to Track A.
 
    Returns
    -------
    shift : int
        Samples Track B should be rolled by (positive = move later).
    polarity_flip : bool
        True if Track B should be multiplied by -1 to phase-match.
    """
    slice_len = int(0.05 * sr)
    a = pad_or_trim(track_a_slice, slice_len)
    b = pad_or_trim(track_b_slice, slice_len)
    try:
        a_sub = apply_filter(a, sr, 100.0, "low", order=2)
        b_sub = apply_filter(b, sr, 100.0, "low", order=2)
        corr = scipy.signal.correlate(a_sub, b_sub, mode="full")
        idx = int(np.argmax(np.abs(corr)))
        shift = idx - (len(b_sub) - 1)
        polarity_flip = bool(corr[idx] < 0.0)
        return shift, polarity_flip
    except Exception:
        print("phase_align failed; returning identity")
        return 0, False
 
 
def apply_phase_correction(track_b, shift, polarity_flip):
    """
    Apply the output of `phase_align` to Track B. Use `np.roll`-style shift
    (small values only -- typically < a few ms) and invert if needed.
    """
    out = np.roll(track_b, int(shift)) if shift else track_b.copy()
    if polarity_flip:
        out = -out
    return out
 
 
# ---------------------------------------------------------------------------
# Loudness matching
# ---------------------------------------------------------------------------
 
def match_rms_to_reference(y, reference_y, max_gain_db=6.0):
    """
    Match `y`'s RMS to `reference_y`, with hard +/- gain limits in dB.
    Divide-by-zero safe for silent inputs.
    """
    ref_rms = max(rms_level(reference_y), EPS)
    y_rms = max(rms_level(y), EPS)
    gain = ref_rms / y_rms
    max_gain = 10 ** (max_gain_db / 20.0)
    min_gain = 1.0 / max_gain
    gain = float(np.clip(gain, min_gain, max_gain))
    return y * gain, gain
 
 
# ---------------------------------------------------------------------------
# Reverb and echo
# ---------------------------------------------------------------------------
 
def simple_reverb_tail(y, sr, decay_seconds=4.0, wet=0.45,
                      pre_delay=0.02, color_hz=5000.0):
    """
    Convolution reverb using a filtered-noise impulse with an exponential
    envelope. Sounds like a small/medium room, not a buzz.
 
    The wet bus is RMS-normalised to the dry signal so the `wet` knob is
    perceptually predictable.
    """
    decay_samples = max(int(decay_seconds * sr), 64)
    pre_delay_samples = max(int(pre_delay * sr), 0)
 
    rng = np.random.default_rng(0xC0FFEE)
    noise = rng.standard_normal(decay_samples)
 
    nyq = 0.5 * sr
    color_hz = float(np.clip(color_hz, 500.0, nyq - 200.0))
    sos = scipy.signal.butter(2, color_hz / nyq, btype="low", output="sos")
    noise = scipy.signal.sosfilt(sos, noise)
 
    # -60 dB at decay_seconds -> tau = decay_seconds / 6.91
    t = np.arange(decay_samples) / sr
    env = np.exp(-t * (6.91 / decay_seconds))
    impulse = noise * env
    if pre_delay_samples > 0:
        impulse = np.concatenate([np.zeros(pre_delay_samples), impulse])
 
    peak = np.max(np.abs(impulse))
    if peak > EPS:
        impulse = impulse / peak
 
    reverb = scipy.signal.fftconvolve(y, impulse, mode="full")[:len(y)]
 
    dry_rms = max(rms_level(y), EPS)
    wet_rms = max(rms_level(reverb), EPS)
    reverb = reverb * (dry_rms / wet_rms)
 
    return (y * (1.0 - wet)) + (reverb * wet)
 
 
def simple_echo(y, sr, delay_seconds=0.375, feedback=0.45, wet=0.5,
                num_taps=6, hp_per_tap_hz=80.0):
    """
    Feedback delay line. Each tap is high-passed before feeding the next,
    keeping the low end clean as taps decay.
    """
    n = len(y)
    delay_samples = max(int(delay_seconds * sr), 1)
    total = n + delay_samples * num_taps
    bus = np.zeros(total)
    bus[:n] = y
 
    nyq = 0.5 * sr
    hp = float(np.clip(hp_per_tap_hz, 20.0, nyq - 100.0))
    sos = scipy.signal.butter(2, hp / nyq, btype="high", output="sos")
 
    tap = y.copy()
    for i in range(1, num_taps + 1):
        tap = scipy.signal.sosfilt(sos, tap) * feedback
        start = delay_samples * i
        end = start + n
        if end > len(bus):
            break
        bus[start:end] += tap
        if np.max(np.abs(tap)) < 1e-4:
            break
 
    echo_sig = bus[:n]
    return (y * (1.0 - wet)) + (echo_sig * wet)
 
 
# ---------------------------------------------------------------------------
# Drive / saturation
# ---------------------------------------------------------------------------
 
def soft_clip_drive(y, drive=2.0):
    """tanh saturation; output renormalised to unity peak."""
    driven = np.tanh(y * drive)
    peak = np.max(np.abs(driven))
    return driven / max(peak, EPS)
 
 
# ---------------------------------------------------------------------------
# Musical alignment helpers
# ---------------------------------------------------------------------------
 
def snap_to_phrase_boundary(beats, requested_time, sr, phrase_beats=32):
    """Snap a target time to the nearest phrase boundary (default 8 bars in 4/4)."""
    beat_times = librosa.samples_to_time(beats, sr=sr)
    if len(beat_times) == 0:
        return float(requested_time), int(requested_time * sr)
    closest = int(np.argmin(np.abs(beat_times - requested_time)))
    phrase_idx = closest - (closest % phrase_beats)
    phrase_idx = int(np.clip(phrase_idx, 0, len(beat_times) - 1))
    snapped_t = float(beat_times[phrase_idx])
    return snapped_t, int(snapped_t * sr)
 
 
def auto_loop_track_a_segment(y_a, start_sample_a, mix_samples, sr, beats_a=None):
    """
    Build a runway of length `mix_samples` from Track A.
 
    If `beats_a` (sample indices) is provided, pick a one-bar loop (4 beats)
    that ends at or before `start_sample_a`, so the loop seam lands on a
    beat. Otherwise fall back to using all available audio after
    `start_sample_a`, then to the trailing N seconds of Track A.
 
    Loop edges get a 1 ms ramp to soften the seam.
    """
    available = y_a[start_sample_a:]
    if len(available) >= mix_samples:
        return available[:mix_samples]
 
    print("Track A is short -- creating auto-loop runway.")
 
    loop_source = None
    if beats_a is not None:
        bts = np.asarray(beats_a, dtype=int)
        bts = bts[(bts > 0) & (bts < len(y_a))]
        if len(bts) >= 5:
            cutoff = min(start_sample_a, int(bts[-1]))
            usable = bts[bts <= cutoff]
            if len(usable) >= 5:
                src = y_a[int(usable[-5]):int(usable[-1])]
                if len(src) >= int(0.25 * sr):
                    loop_source = src
 
    if loop_source is None or len(loop_source) == 0:
        min_loop = int(4 * sr)
        if len(available) >= min_loop:
            loop_source = available
        else:
            loop_source = y_a[max(0, len(y_a) - min_loop):]
 
    if len(loop_source) == 0:
        return np.zeros(mix_samples)
 
    edge = max(1, int(0.001 * sr))
    if len(loop_source) > 2 * edge:
        loop_source = loop_source.copy()
        loop_source[:edge] = loop_source[:edge] * np.linspace(0.0, 1.0, edge)
        loop_source[-edge:] = loop_source[-edge:] * np.linspace(1.0, 0.0, edge)
 
    repeats = int(np.ceil(mix_samples / len(loop_source)))
    return np.tile(loop_source, repeats)[:mix_samples]
 
 
# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------
 
def bass_swap_transition(segment_a, segment_b, sr):
    """
    Classic DJ bass swap: lows swap at the midpoint with a short Hann
    crossfade; mids/highs do an equal-power crossfade over the full window.
    """
    n = len(segment_a)
    print("Bass swap transition (n=%d)", n)
 
    a_low = apply_filter(segment_a, sr, XO_LOW, "low")
    a_high = apply_filter(segment_a, sr, XO_LOW, "high")
    b_low = apply_filter(segment_b, sr, XO_LOW, "low")
    b_high = apply_filter(segment_b, sr, XO_LOW, "high")
 
    fade_out, fade_in = equal_power_fades(n)
 
    mid = n // 2
    xfade = min(int(0.060 * sr), mid)  # 60 ms Hann pivot
    start = max(0, mid - xfade // 2)
    end = min(n, start + xfade)
    fout_b, fin_b = hann_fades(end - start)
 
    bass = np.empty(n)
    bass[:start] = a_low[:start]
    bass[start:end] = a_low[start:end] * fout_b + b_low[start:end] * fin_b
    bass[end:] = b_low[end:]
 
    highs = a_high * fade_out + b_high * fade_in
    return _finalize(bass + highs * 0.9)
 
 
def hpf_sweep_transition(segment_a, segment_b, sr, end_freq=5000.0):
    n = len(segment_a)
    print("HPF sweep transition (n=%d, end=%.0f Hz)", n, end_freq)
    swept_a = _stft_sweep(segment_a, sr,
                          np.linspace(20.0, float(end_freq), 256),
                          btype="high")
    fout, fin = equal_power_fades(n)
    return _finalize(swept_a * fout + segment_b * fin)
 
 
def lpf_sweep_transition(segment_a, segment_b, sr, end_freq=300.0):
    n = len(segment_a)
    print("LPF sweep transition (n=%d, end=%.0f Hz)", n, end_freq)
    swept_a = _stft_sweep(segment_a, sr,
                          np.linspace(18000.0, float(end_freq), 256),
                          btype="low")
    fout, fin = equal_power_fades(n)
    return _finalize(swept_a * fout + segment_b * fin)
 
 
def reverb_wash_transition(segment_a, segment_b, sr):
    """
    Track A's tail gets washed in reverb while crossfading into Track B.
    Useful for non-harmonic / tempo-clashing pairs.
    """
    n = len(segment_a)
    print("Reverb wash transition (n=%d)", n)
    fout, fin = equal_power_fades(n)
    tail = min(int(4 * sr), n)
    washed = segment_a.copy()
    washed[-tail:] = simple_reverb_tail(
        segment_a[-tail:], sr, decay_seconds=6.0, wet=0.6
    )
    return _finalize(washed * fout + segment_b * fin)
 
 
def drop_mix_transition(segment_a, segment_b, sr):
    """
    Abrupt cut at the midpoint with a very short Hann crossfade so the
    edit point isn't audible as a click.
    """
    n = len(segment_a)
    print("Drop mix transition (n=%d)", n)
    cut = n // 2
    xfade = min(int(0.010 * sr), cut, n - cut)  # 10 ms
    if xfade < 2:
        out = np.concatenate([segment_a[:cut], segment_b[cut:]])
        return _finalize(out)
 
    s = cut - xfade // 2
    e = s + xfade
    fout, fin = hann_fades(xfade)
    out = np.empty(n)
    out[:s] = segment_a[:s]
    out[s:e] = segment_a[s:e] * fout + segment_b[s:e] * fin
    out[e:] = segment_b[e:]
    return _finalize(out)
 
 
def harmonic_mix_transition(segment_a, segment_b, sr):
    """Gentle equal-power blend with a softer bass handover."""
    n = len(segment_a)
    print("Harmonic mix transition (n=%d)", n)
    fout, fin = equal_power_fades(n)
 
    a_low = apply_filter(segment_a, sr, XO_LOW, "low")
    a_high = apply_filter(segment_a, sr, XO_LOW, "high")
    b_low = apply_filter(segment_b, sr, XO_LOW, "low")
    b_high = apply_filter(segment_b, sr, XO_LOW, "high")
 
    bass_out, bass_in = equal_power_fades(n)
    mixed_low = a_low * bass_out * 0.85 + b_low * bass_in * 0.85
    mixed_high = a_high * fout + b_high * fin
    return _finalize(mixed_low + mixed_high)
 
 
def phrase_mix_transition(segment_a, segment_b, sr):
    """Phrase-aware blend with S-curve fades; Track B enters in the back half."""
    n = len(segment_a)
    print("Phrase mix transition (n=%d)", n)
    fade_in = s_curve(n, midpoint=0.55, steepness=10.0)
    fade_out = 1.0 - s_curve(n, midpoint=0.45, steepness=10.0)
 
    a_low = apply_filter(segment_a, sr, 220.0, "low")
    a_high = apply_filter(segment_a, sr, 220.0, "high")
    b_low = apply_filter(segment_b, sr, 220.0, "low")
    b_high = apply_filter(segment_b, sr, 220.0, "high")
 
    b_low_g = s_curve(n, midpoint=0.70, steepness=18.0)
    a_low_g = 1.0 - s_curve(n, midpoint=0.65, steepness=18.0)
 
    mixed_low = a_low * a_low_g + b_low * b_low_g
    mixed_high = a_high * fade_out + b_high * fade_in
    return _finalize(mixed_low * 0.9 + mixed_high * 0.9)
 
 
def echo_out_transition(segment_a, segment_b, sr):
    """Apply feedback echo to Track A's tail while crossfading into Track B."""
    n = len(segment_a)
    print("Echo out transition (n=%d)", n)
    fout, fin = equal_power_fades(n)
    region = min(int(8 * sr), n)
    processed = segment_a.copy()
    processed[-region:] = simple_echo(
        segment_a[-region:], sr,
        delay_seconds=0.375, feedback=0.55, wet=0.65
    )
    return _finalize(processed * fout + segment_b * fin)
 
 
def loop_roll_transition(segment_a, segment_b, sr):
    """
    Capture a short slice of Track A and roll it through the last 25% of
    the transition window, decaying as Track B enters.
    """
    n = len(segment_a)
    print("Loop roll transition (n=%d)", n)
    fout, fin = equal_power_fades(n)
 
    roll_start = int(n * 0.75)
    src_len = int(0.5 * sr)
    src_start = max(0, roll_start - src_len)
    source = segment_a[src_start:roll_start]
    target_len = n - roll_start
 
    if len(source) < int(0.05 * sr) or target_len <= 0:
        return _finalize(segment_a * fout + segment_b * fin)
 
    edge = max(1, int(0.002 * sr))
    if len(source) > 2 * edge:
        source = source.copy()
        source[:edge] = source[:edge] * np.linspace(0.0, 1.0, edge)
        source[-edge:] = source[-edge:] * np.linspace(1.0, 0.0, edge)
 
    repeats = int(np.ceil(target_len / len(source)))
    rolled = np.tile(source, repeats)[:target_len]
    roll_env = np.linspace(1.0, 0.15, target_len)
    processed = segment_a.copy()
    processed[roll_start:] = rolled * roll_env
 
    return _finalize(processed * fout + segment_b * fin)
 
 
def long_eq_blend_transition(segment_a, segment_b, sr):
    """
    Three-band DJ-style EQ blend: lows fade linearly, mids cross at the
    midpoint via S-curve, highs taper gently.
    """
    n = len(segment_a)
    print("Long EQ blend transition (n=%d)", n)
 
    a_low = apply_filter(segment_a, sr, XO_LOW, "low")
    a_high = apply_filter(segment_a, sr, XO_MID, "high")
    a_mid = segment_a - a_low - a_high
    b_low = apply_filter(segment_b, sr, XO_LOW, "low")
    b_high = apply_filter(segment_b, sr, XO_MID, "high")
    b_mid = segment_b - b_low - b_high
 
    a_low_g = np.linspace(1.0, 0.0, n)
    b_low_g = np.linspace(0.0, 1.0, n)
    a_mid_g = 1.0 - s_curve(n, midpoint=0.45, steepness=8.0)
    b_mid_g = s_curve(n, midpoint=0.55, steepness=8.0)
    a_high_g = np.linspace(1.0, 0.2, n)
    b_high_g = np.linspace(0.2, 1.0, n)
 
    mixed = (a_low * a_low_g + b_low * b_low_g +
             a_mid * a_mid_g + b_mid * b_mid_g +
             a_high * a_high_g + b_high * b_high_g)
    return _finalize(mixed * 0.92)
 
 
def energy_blend_transition(segment_a, segment_b, sr):
    """Smooth energy handoff; bass enters late to avoid low-end clutter."""
    n = len(segment_a)
    print("Energy blend transition (n=%d)", n)
    fout = 1.0 - s_curve(n, midpoint=0.55, steepness=9.0)
    fin = s_curve(n, midpoint=0.45, steepness=9.0)
 
    a_low = apply_filter(segment_a, sr, 220.0, "low")
    a_high = apply_filter(segment_a, sr, 220.0, "high")
    b_low = apply_filter(segment_b, sr, 220.0, "low")
    b_high = apply_filter(segment_b, sr, 220.0, "high")
 
    a_low_g = 1.0 - s_curve(n, midpoint=0.62, steepness=14.0)
    b_low_g = s_curve(n, midpoint=0.68, steepness=14.0)
 
    mixed_low = a_low * a_low_g + b_low * b_low_g
    mixed_high = a_high * fout + b_high * fin
    return _finalize(mixed_low * 0.88 + mixed_high * 0.92)
 
 
def percussion_blend_transition(segment_a, segment_b, sr):
    """Groove-forward blend: percussion / mids handed over early, bass late."""
    n = len(segment_a)
    print("Percussion blend transition (n=%d)", n)
 
    a_low = apply_filter(segment_a, sr, XO_LOW, "low")
    a_high = apply_filter(segment_a, sr, 1800.0, "high")
    a_mid = segment_a - a_low - a_high
    b_low = apply_filter(segment_b, sr, XO_LOW, "low")
    b_high = apply_filter(segment_b, sr, 1800.0, "high")
    b_mid = segment_b - b_low - b_high
 
    a_low_g = 1.0 - s_curve(n, midpoint=0.60, steepness=14.0)
    b_low_g = s_curve(n, midpoint=0.72, steepness=14.0)
    a_mid_g = np.linspace(1.0, 0.35, n)
    b_mid_g = np.linspace(0.25, 1.0, n)
    a_high_g = np.linspace(0.9, 0.25, n)
    b_high_g = np.linspace(0.35, 1.0, n)
 
    mixed = (a_low * a_low_g + b_low * b_low_g +
             a_mid * a_mid_g + b_mid * b_mid_g +
             a_high * a_high_g + b_high * b_high_g)
    return _finalize(mixed * 0.9)
 
 
def breakdown_blend_transition(segment_a, segment_b, sr):
    """Soft, atmospheric blend; low end held back to leave space."""
    n = len(segment_a)
    print("Breakdown blend transition (n=%d)", n)
 
    a_low = apply_filter(segment_a, sr, 160.0, "low")
    a_high = apply_filter(segment_a, sr, 160.0, "high")
    b_low = apply_filter(segment_b, sr, 160.0, "low")
    b_high = apply_filter(segment_b, sr, 160.0, "high")
 
    a_high_g = 1.0 - s_curve(n, midpoint=0.55, steepness=7.0)
    b_high_g = s_curve(n, midpoint=0.35, steepness=7.0)
    a_low_g = 1.0 - s_curve(n, midpoint=0.48, steepness=14.0)
    b_low_g = s_curve(n, midpoint=0.78, steepness=14.0)
 
    mixed_low = a_low * a_low_g + b_low * b_low_g
    mixed_high = a_high * a_high_g + b_high * b_high_g
    return _finalize(mixed_low * 0.78 + mixed_high * 0.95)
 
 
def ambient_transition(segment_a, segment_b, sr):
    """
    Strip Track A's low end and wash it in long reverb; Track B's lows
    enter very late so the transition feels spatial.
    """
    n = len(segment_a)
    print("Ambient transition (n=%d)", n)
    fout, fin = equal_power_fades(n)
 
    a_air = apply_filter(segment_a, sr, 350.0, "high", order=2)
    b_low = apply_filter(segment_b, sr, 220.0, "low")
    b_air = apply_filter(segment_b, sr, 220.0, "high")
    b_low_g = s_curve(n, midpoint=0.75, steepness=14.0)
 
    washed_a = simple_reverb_tail(a_air, sr, decay_seconds=7.0, wet=0.6)
 
    mixed = (washed_a * fout * 0.85 +
             b_air * fin * 0.9 +
             b_low * b_low_g * 0.8)
    return _finalize(mixed)
 
 
def techno_filter_drive_transition(segment_a, segment_b, sr):
    """Driven Track A with an aggressive HPF sweep; Track B enters underneath."""
    n = len(segment_a)
    print("Techno filter drive transition (n=%d)", n)
    fout, fin = equal_power_fades(n)
 
    driven_a = soft_clip_drive(segment_a, drive=2.2)
    filtered_a = _stft_sweep(
        driven_a, sr,
        np.linspace(80.0, 3500.0, 256),
        btype="high"
    )
 
    b_low = apply_filter(segment_b, sr, 220.0, "low")
    b_high = apply_filter(segment_b, sr, 220.0, "high")
    b_low_g = s_curve(n, midpoint=0.68, steepness=16.0)
 
    mixed = (filtered_a * fout * 0.85 +
             b_high * fin * 0.9 +
             b_low * b_low_g * 0.9)
    return _finalize(mixed)