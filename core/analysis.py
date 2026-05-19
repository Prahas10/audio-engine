import os
import tempfile
import warnings

import numpy as np
import librosa
import scipy.signal
import soundfile as sf

TARGET_SR = 44100


# ---------------------------------------------------------------------------
# Beat tracking
# ---------------------------------------------------------------------------
import os
import tempfile
import numpy as np
import soundfile as sf
import librosa


def _audio_to_temp_wav(y, sr):
    """
    Converts in-memory audio into a temporary WAV file
    so madmom can reliably process it on Windows.

    Returns:
        temp_path (str)
    """
    audio = np.asarray(y, dtype=np.float32)

    # Convert stereo → mono if needed
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    # Prevent clipping issues
    peak = np.max(np.abs(audio)) if len(audio) else 0.0

    if peak > 1.0:
        audio = audio / peak

    # Create temp wav
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as tmp:
        temp_path = tmp.name

    sf.write(temp_path, audio, sr)

    return temp_path


def _safe_remove(path):
    """
    Safely removes temporary files without crashing.
    """
    if not path:
        return

    try:
        if os.path.exists(path):
            os.remove(path)

    except PermissionError:
        print(f"Could not remove temp file (PermissionError): {path}")

    except OSError as e:
        print(f"Could not remove temp file ({e}): {path}")

def safe_bpm(y, sr):
    """
    Backward-compatible function.

    Returns:
        tempo, beat_samples

    Uses madmom first, librosa fallback.
    """
    tempo, beats, _downbeats = safe_bpm_with_downbeats(y, sr)
    return tempo, beats


def safe_bpm_with_downbeats(y, sr):
    """
    Madmom-first beat + downbeat tracking.

    Returns:
        tempo, beat_samples, downbeat_samples

    This keeps your old code compatible while allowing planner.py/library.py
    to use downbeats when you want phrase-aware transitions.
    """
    result = _get_beats_downbeats_madmom_from_audio(y, sr)

    if result is not None:
        beat_times, downbeat_times = result

        if len(beat_times) >= 4:
            tempo = _estimate_bpm_from_beat_times(beat_times)
            beat_samples = librosa.time_to_samples(beat_times, sr=sr).astype(int)

            if downbeat_times is None or len(downbeat_times) < 2:
                downbeat_samples = beat_samples[::4]
            else:
                downbeat_samples = librosa.time_to_samples(downbeat_times, sr=sr).astype(int)

            return float(tempo), beat_samples, downbeat_samples

    return _safe_bpm_librosa_with_downbeats(y, sr)


def safe_bpm_from_path(audio_path, sr=TARGET_SR):
    """
    Loads any format with librosa first, writes temp WAV, then runs madmom.
    This avoids madmom MP3/ffmpeg decode failures on Windows.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    temp_path = None

    try:
        temp_path = _audio_to_temp_wav(y, sr)

        result = _get_beats_downbeats_madmom_from_path(temp_path, sr)

        if result is not None:
            beat_times, downbeat_times = result

            if len(beat_times) >= 4:
                tempo = _estimate_bpm_from_beat_times(beat_times)
                beat_samples = librosa.time_to_samples(beat_times, sr=sr).astype(int)

                if downbeat_times is None or len(downbeat_times) < 2:
                    downbeat_samples = beat_samples[::4]
                else:
                    downbeat_samples = librosa.time_to_samples(
                        downbeat_times,
                        sr=sr,
                    ).astype(int)

                return float(tempo), beat_samples, downbeat_samples

    except Exception as e:
        print(f"madmom temp WAV analysis failed ({e}) — falling back to librosa.")

    finally:
        _safe_remove(temp_path)

    return _safe_bpm_librosa_with_downbeats(y, sr)


def _safe_bpm_librosa_with_downbeats(y, sr):
    tempo, beats_lib = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    tempo = float(np.asarray(tempo).squeeze())

    if not np.isfinite(tempo) or tempo <= 0:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(
            np.asarray(
                librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            ).squeeze()
        )

    if not np.isfinite(tempo) or tempo <= 0:
        tempo = 128.0

    beats_lib = np.asarray(beats_lib, dtype=int)

    if len(beats_lib) >= 4:
        downbeats = beats_lib[::4]
    else:
        downbeats = np.asarray([], dtype=int)

    return float(tempo), beats_lib, downbeats


def _get_beats_downbeats_madmom_from_audio(y, sr):
    """
    Madmom is more stable with file input than raw librosa arrays.
    So for in-memory audio, write a temporary WAV and run path-based analysis.
    """
    temp_path = None

    try:
        audio = np.asarray(y, dtype=np.float32)

        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        peak = np.max(np.abs(audio)) if len(audio) else 0.0
        if peak > 1.0:
            audio = audio / peak

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name

        sf.write(temp_path, audio, sr)
        return _get_beats_downbeats_madmom_from_path(temp_path, sr)

    except Exception as e:
        print(f"madmom audio analysis failed ({e}) — falling back to librosa.")
        return None

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _get_beats_downbeats_madmom_from_path(audio_path, sr=TARGET_SR):
    """
    Returns:
        beat_times, downbeat_times

    beat_times and downbeat_times are in seconds.
    """
    try:
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

        y_tmp, sr_tmp = librosa.load(audio_path, sr=sr, mono=True)
        duration = librosa.get_duration(y=y_tmp, sr=sr_tmp)

        # Beat tracking
        beat_activations = RNNBeatProcessor()(audio_path)
        beat_times = DBNBeatTrackingProcessor(fps=100)(beat_activations)
        beat_times = _sanitize_times(beat_times, duration)

        if len(beat_times) < 4:
            return None

        bpm = _estimate_bpm_from_beat_times(beat_times)
        min_bpm = max(60.0, bpm * 0.75)
        max_bpm = min(200.0, bpm * 1.25)

        # Downbeat tracking
        downbeat_times = None

        try:
            downbeat_activations = RNNDownBeatProcessor()(audio_path)

            downbeat_result = DBNDownBeatTrackingProcessor(
                beats_per_bar=[4],
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                fps=100,
            )(downbeat_activations)

            downbeat_result = np.asarray(downbeat_result, dtype=float)

            if downbeat_result.ndim == 2 and downbeat_result.shape[1] >= 2:
                all_times = downbeat_result[:, 0]
                beat_positions = downbeat_result[:, 1].astype(int)
                downbeat_times = all_times[beat_positions == 1]
                downbeat_times = _sanitize_times(downbeat_times, duration)

        except Exception as e:
            print(f"madmom downbeat tracking failed ({e}) — using every 4th beat.")

        if downbeat_times is None or len(downbeat_times) < 2:
            downbeat_times = beat_times[::4]

        return beat_times, downbeat_times

    except ImportError:
        print("madmom not installed — falling back to librosa beat tracker.")
        print("Install with: pip install madmom")
        return None

    except Exception as e:
        print(f"madmom beat/downbeat tracking failed ({e}) — falling back to librosa.")
        return None


def _sanitize_times(times, duration):
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    times = times[(times >= 0.0) & (times <= duration)]
    return np.unique(np.round(times, 5))


def _estimate_bpm_from_beat_times(beat_times):
    beat_times = np.asarray(beat_times, dtype=float)

    if len(beat_times) < 4:
        return 128.0

    ibi = np.diff(beat_times)
    ibi = ibi[(ibi > 0.25) & (ibi < 1.5)]

    if len(ibi) == 0:
        return 128.0

    bpm = 60.0 / float(np.median(ibi))

    while bpm < 70:
        bpm *= 2.0

    while bpm > 180:
        bpm /= 2.0

    return float(bpm)


# ---------------------------------------------------------------------------
# Key estimation
# ---------------------------------------------------------------------------

def estimate_key(y, sr):
    """
    Uses chroma_cens, which is more robust than chroma_cqt for mastered audio.

    Confidence = margin between winner and second-best key.
    """
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-9)

    major_profile = np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
        2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ])

    minor_profile = np.array([
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
        2.54, 4.75, 3.98, 2.69, 3.34, 3.17
    ])

    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)

    key_names = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    all_scores = []

    for i in range(12):
        major_score = float(np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1])
        minor_score = float(np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1])

        all_scores.append((major_score, key_names[i], "major"))
        all_scores.append((minor_score, key_names[i], "minor"))

    all_scores.sort(key=lambda x: x[0], reverse=True)

    best_score, best_key, best_mode = all_scores[0]
    second_score = all_scores[1][0]

    confidence = float(np.clip(best_score - second_score, 0.0, 1.0))

    return best_key, best_mode, confidence


# ---------------------------------------------------------------------------
# Energy curve
# ---------------------------------------------------------------------------

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

    max_rms = np.max(rms)

    if max_rms > 0:
        rms = rms / max_rms

    return times, rms


# ---------------------------------------------------------------------------
# Phrase / beat helpers
# ---------------------------------------------------------------------------

def get_phrase_boundaries_from_downbeats(downbeats, sr, phrase_bars=8):
    """
    Madmom downbeats = bar starts.
    Every 8 downbeats = one 8-bar phrase boundary.
    """
    downbeat_times = librosa.samples_to_time(np.asarray(downbeats), sr=sr)

    if len(downbeat_times) == 0:
        return []

    return [
        float(downbeat_times[i])
        for i in range(0, len(downbeat_times), phrase_bars)
    ]


def phrase_boundary_candidates_from_downbeats(
    downbeats,
    sr,
    song_duration,
    phrase_bars=8,
    min_percent=0.55,
    max_percent=0.92,
):
    """
    Better Track A transition candidates using madmom downbeats.
    """
    phrase_times = get_phrase_boundaries_from_downbeats(
        downbeats=downbeats,
        sr=sr,
        phrase_bars=phrase_bars,
    )

    candidates = [
        float(t)
        for t in phrase_times
        if song_duration * min_percent <= float(t) <= song_duration * max_percent
    ]

    if candidates:
        return candidates

    # fallback to all downbeats in the search zone
    downbeat_times = librosa.samples_to_time(np.asarray(downbeats), sr=sr)

    return [
        float(t)
        for t in downbeat_times
        if song_duration * min_percent <= float(t) <= song_duration * max_percent
    ]


def intro_phrase_candidates_from_downbeats(
    downbeats,
    sr,
    song_duration,
    phrase_bars=8,
    max_percent=0.35,
):
    """
    Better Track B entry candidates using madmom downbeats.
    """
    phrase_times = get_phrase_boundaries_from_downbeats(
        downbeats=downbeats,
        sr=sr,
        phrase_bars=phrase_bars,
    )

    candidates = [
        float(t)
        for t in phrase_times
        if 0 <= float(t) <= song_duration * max_percent
    ]

    if candidates:
        return candidates

    downbeat_times = librosa.samples_to_time(np.asarray(downbeats), sr=sr)

    return [
        float(t)
        for t in downbeat_times
        if 0 <= float(t) <= song_duration * max_percent
    ][:8]
    
def phrase_boundary_candidates(beats, sr, song_duration, phrase_beats=32):
    """
    If you pass regular beats:
        phrase_beats=32 means every 8 bars.

    If you pass downbeats:
        phrase_beats=8 means every 8 bars.

    Search window stays in the latter part of Track A.
    """
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


def get_phrase_boundaries(beats, sr, phrase_beats=32):
    beat_times = librosa.samples_to_time(beats, sr=sr)

    if len(beat_times) == 0:
        return []

    return [
        float(beat_times[i])
        for i in range(0, len(beat_times), phrase_beats)
    ]


def get_drop_candidates(y, sr, candidate_times, top_k=8):
    """
    Scores phrase candidates by energy jump after the phrase boundary.

    Use this for Track B entry selection.
    """
    if not candidate_times:
        return []

    rms_times, rms = get_energy_curve(y, sr)

    scored = []

    for t in candidate_times:
        pre_mask = (rms_times >= t - 4.0) & (rms_times < t)
        post_mask = (rms_times >= t) & (rms_times < t + 4.0)

        if not np.any(pre_mask) or not np.any(post_mask):
            continue

        pre_energy = float(np.mean(rms[pre_mask]) + 1e-8)
        post_energy = float(np.mean(rms[post_mask]) + 1e-8)

        energy_jump = post_energy / pre_energy

        scored.append((energy_jump, float(t)))

    scored.sort(reverse=True)

    return [t for _, t in scored[:top_k]]


# ---------------------------------------------------------------------------
# Camelot compatibility
# ---------------------------------------------------------------------------

def camelot_compatible(camelot_a, camelot_b):
    """
    Fixed wraparound:
    key 1 previous key correctly maps to 12.
    """
    if camelot_a is None or camelot_b is None:
        return False

    num_a, mode_a = int(camelot_a[:-1]), camelot_a[-1]
    num_b, mode_b = int(camelot_b[:-1]), camelot_b[-1]

    compatible = {
        (num_a, mode_a),
        (num_a % 12 + 1, mode_a),
        ((num_a - 2) % 12 + 1, mode_a),
        (num_a, "A" if mode_a == "B" else "B"),
    }

    return (num_b, mode_b) in compatible


# ---------------------------------------------------------------------------
# RMS
# ---------------------------------------------------------------------------

def rms_level(y):
    return float(np.sqrt(np.mean(y ** 2) + 1e-9))


# ---------------------------------------------------------------------------
# Post-stretch beat grid refinement
# ---------------------------------------------------------------------------

def refine_beat_grid(y_stretched, sr, bpm_target=None):
    """
    After time-stretching Track B, re-estimates beats.

    bpm_target is kept as an optional arg so old calls do not break.
    """
    tempo, beats = safe_bpm(y_stretched, sr)
    return tempo, beats


# ---------------------------------------------------------------------------
# Beat alignment verification
# ---------------------------------------------------------------------------

def verify_beat_alignment(segment_a, segment_b, sr, bpm, tolerance_ms=12.0):
    """
    Cross-correlates onset-strength envelopes to measure actual beat drift.

    Returns:
        drift_ms, is_aligned
    """
    hop = 512

    onset_a = librosa.onset.onset_strength(
        y=segment_a,
        sr=sr,
        hop_length=hop
    )

    onset_b = librosa.onset.onset_strength(
        y=segment_b,
        sr=sr,
        hop_length=hop
    )

    min_len = min(len(onset_a), len(onset_b))

    if min_len <= 1:
        return 0.0, True

    onset_a = onset_a[:min_len]
    onset_b = onset_b[:min_len]

    onset_a = onset_a - np.mean(onset_a)
    onset_b = onset_b - np.mean(onset_b)

    corr = scipy.signal.correlate(onset_a, onset_b, mode="full")
    drift_frames = int(np.argmax(corr) - (min_len - 1))

    drift_ms = (drift_frames * hop / sr) * 1000.0

    tolerance_samples = int((tolerance_ms / 1000.0) * sr)
    drift_samples = int(drift_frames * hop)

    return float(drift_ms), bool(abs(drift_samples) <= tolerance_samples)


# ---------------------------------------------------------------------------
# Transient sharpening
# ---------------------------------------------------------------------------

def sharpen_transients(y, sr, strength=0.35):
    """
    High-shelf style transient restore after time-stretching.
    """
    if len(y) == 0:
        return y

    sos = scipy.signal.butter(
        2,
        3000.0 / (sr / 2.0),
        btype="high",
        output="sos"
    )

    highs = scipy.signal.sosfiltfilt(sos, y)
    out = y + highs * strength

    peak = np.max(np.abs(out))

    if peak > 1.0:
        out = out / peak

    return out


# ---------------------------------------------------------------------------
# Entry ramp
# ---------------------------------------------------------------------------

def apply_entry_ramp(y, ramp_ms=15.0, sr=TARGET_SR):
    """
    Short fade-in on segment_b to eliminate sample-0 clicks.
    """
    if len(y) == 0:
        return y

    ramp_samples = min(int((ramp_ms / 1000.0) * sr), len(y))

    if ramp_samples <= 1:
        return y

    y = y.copy()
    y[:ramp_samples] *= np.linspace(0.0, 1.0, ramp_samples)

    return y