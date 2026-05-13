import numpy as np
import scipy.signal
import librosa
from core.analysis import rms_level

# Ensures an audio array matches a specific length by either padding with silence or trimming the end.
def pad_or_trim(y, target_len):
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

# Applies a standard Butterworth filter (Low-pass or High-pass) to a specific audio segment.
def apply_filter(y, sr, cutoff, btype, order=4):
    nyquist = 0.5 * sr
    cutoff = min(cutoff, nyquist - 100)
    normal_cutoff = cutoff / nyquist

    b, a = scipy.signal.butter(order, normal_cutoff, btype=btype)
    return scipy.signal.filtfilt(b, a, y)

# Generates the mathematical curves for an equal-power crossfade to maintain consistent volume during a mix.
def equal_power_fades(n):
    t = np.linspace(0, 1, n)
    fade_out = np.cos(t * np.pi / 2)
    fade_in = np.sin(t * np.pi / 2)
    return fade_out, fade_in


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