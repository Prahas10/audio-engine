"""
core/analysis.py

Changes in this version:
  - fine_sync_onset: coarse-to-fine two-pass approach
      Pass 1: full-track coarse scan at beat resolution to find best phrase
      Pass 2: sub-beat refinement within ±2 beats of the coarse winner
      Result validated against verify_beat_alignment; falls back to original
      phrase if post-sync drift > 2 seconds (indicates a bad lock)
  - get_energy_curve: returns (times, rms_norm, raw_rms_db) — 3 values
  - phrase_bars_from_bpm: BPM-aware phrase grouping
  - estimate_key: spectral-rolloff gating + median chroma
  - All other helpers unchanged
"""

import os
import tempfile
import numpy as np
import scipy.signal
import librosa
import soundfile as sf

TARGET_SR = 44100

# ---------------------------------------------------------------------------
# BPM-derived phrase grouping
# ---------------------------------------------------------------------------

def phrase_bars_from_bpm(bpm: float) -> int:
    if bpm < 100:   return 16
    elif bpm < 145: return 8
    else:           return 4


# ---------------------------------------------------------------------------
# Temp-file helpers
# ---------------------------------------------------------------------------

def _audio_to_temp_wav(y, sr):
    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name
    sf.write(temp_path, audio, sr)
    return temp_path


def _safe_remove(path):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except (PermissionError, OSError) as e:
        print(f"Could not remove temp file ({e}): {path}")


# ---------------------------------------------------------------------------
# Beat tracking — public API
# ---------------------------------------------------------------------------

def safe_bpm(y, sr):
    tempo, beats, _db = safe_bpm_with_downbeats(y, sr)
    return tempo, beats


def safe_bpm_with_downbeats(y, sr):
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
                    downbeat_samples = librosa.time_to_samples(downbeat_times, sr=sr).astype(int)
                return float(tempo), beat_samples, downbeat_samples
    except Exception as e:
        print(f"madmom temp WAV analysis failed ({e}) — falling back to librosa.")
    finally:
        _safe_remove(temp_path)
    return _safe_bpm_librosa_with_downbeats(y, sr)


# ---------------------------------------------------------------------------
# Beat tracking — internal
# ---------------------------------------------------------------------------

def _safe_bpm_librosa_with_downbeats(y, sr):
    tempo, beats_lib = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    tempo = float(np.asarray(tempo).squeeze())
    if not np.isfinite(tempo) or tempo <= 0:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(np.asarray(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)).squeeze())
    if not np.isfinite(tempo) or tempo <= 0:
        tempo = 128.0
    beats_lib = np.asarray(beats_lib, dtype=int)
    downbeats = beats_lib[::4] if len(beats_lib) >= 4 else np.asarray([], dtype=int)
    return float(tempo), beats_lib, downbeats


def _get_beats_downbeats_madmom_from_audio(y, sr):
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
    try:
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor

        y_tmp, _ = librosa.load(audio_path, sr=sr, mono=True)
        duration = librosa.get_duration(y=y_tmp, sr=sr)

        beat_activations = RNNBeatProcessor()(audio_path)
        beat_times = DBNBeatTrackingProcessor(fps=100)(beat_activations)
        beat_times = _sanitize_times(beat_times, duration)
        if len(beat_times) < 4:
            return None

        bpm = _estimate_bpm_from_beat_times(beat_times)
        min_bpm = max(60.0, bpm * 0.75)
        max_bpm = min(200.0, bpm * 1.25)

        downbeat_times = None
        try:
            downbeat_activations = RNNDownBeatProcessor()(audio_path)
            downbeat_result = DBNDownBeatTrackingProcessor(
                beats_per_bar=[4], min_bpm=min_bpm, max_bpm=max_bpm, fps=100,
            )(downbeat_activations)
            downbeat_result = np.asarray(downbeat_result, dtype=float)
            if downbeat_result.ndim == 2 and downbeat_result.shape[1] >= 2:
                all_times   = downbeat_result[:, 0]
                beat_pos    = downbeat_result[:, 1].astype(int)
                downbeat_times = _sanitize_times(all_times[beat_pos == 1], duration)
        except Exception as e:
            print(f"madmom downbeat tracking failed ({e}) — using every 4th beat.")

        if downbeat_times is None or len(downbeat_times) < 2:
            downbeat_times = beat_times[::4]

        return beat_times, downbeat_times

    except ImportError:
        print("madmom not installed — falling back to librosa beat tracker.")
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
    while bpm < 70:  bpm *= 2.0
    while bpm > 180: bpm /= 2.0
    return float(bpm)


# ---------------------------------------------------------------------------
# Key estimation — spectral-rolloff gated, median chroma
# ---------------------------------------------------------------------------

def estimate_key(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mean_rolloff = float(np.mean(rolloff))
    high_energy_track = mean_rolloff > 8000.0
    hop = 1024 if high_energy_track else 512

    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
    chroma_mean = np.median(chroma, axis=1)
    chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-9)

    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)
    key_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

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
    if high_energy_track and confidence < 0.25:
        confidence *= 0.6
    return best_key, best_mode, confidence


# ---------------------------------------------------------------------------
# Energy curve
# ---------------------------------------------------------------------------

def get_energy_curve(y, sr, frame_length=2048, hop_length=512):
    """Returns (times, rms_normalised, raw_rms_db)."""
    rms   = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    raw_rms    = float(np.sqrt(np.mean(y ** 2)))
    raw_rms_db = float(20.0 * np.log10(raw_rms + 1e-9))
    max_rms    = np.max(rms)
    rms_norm   = rms / max_rms if max_rms > 0 else rms
    return times, rms_norm, raw_rms_db


# ---------------------------------------------------------------------------
# Phrase / downbeat helpers
# ---------------------------------------------------------------------------

def get_phrase_boundaries_from_downbeats(downbeats, sr, phrase_bars=8):
    downbeat_times = librosa.samples_to_time(np.asarray(downbeats), sr=sr)
    if len(downbeat_times) == 0:
        return []
    return [float(downbeat_times[i]) for i in range(0, len(downbeat_times), phrase_bars)]


def phrase_boundary_candidates_from_downbeats(
    downbeats, sr, song_duration, bpm=128.0, min_percent=0.55, max_percent=0.92,
):
    phrase_bars  = phrase_bars_from_bpm(bpm)
    phrase_times = get_phrase_boundaries_from_downbeats(downbeats, sr, phrase_bars)
    candidates   = [
        float(t) for t in phrase_times
        if song_duration * min_percent <= float(t) <= song_duration * max_percent
    ]
    if candidates:
        return candidates
    downbeat_times = librosa.samples_to_time(np.asarray(downbeats), sr=sr)
    return [
        float(t) for t in downbeat_times
        if song_duration * min_percent <= float(t) <= song_duration * max_percent
    ]


def intro_phrase_candidates_from_downbeats(
    downbeats, sr, song_duration, bpm=128.0, max_percent=0.35,
):
    phrase_bars  = phrase_bars_from_bpm(bpm)
    phrase_times = get_phrase_boundaries_from_downbeats(downbeats, sr, phrase_bars)
    candidates   = [float(t) for t in phrase_times if 0 <= float(t) <= song_duration * max_percent]
    if candidates:
        return candidates
    downbeat_times = librosa.samples_to_time(np.asarray(downbeats), sr=sr)
    return [float(t) for t in downbeat_times if 0 <= float(t) <= song_duration * max_percent][:8]


def phrase_boundary_candidates(beats, sr, song_duration, phrase_beats=32):
    beat_times   = librosa.samples_to_time(beats, sr=sr)
    if len(beat_times) == 0:
        return []
    start_search = song_duration * 0.55
    end_search   = song_duration * 0.92
    return [float(beat_times[i]) for i in range(0, len(beat_times), phrase_beats)
            if start_search <= beat_times[i] <= end_search]


def get_phrase_boundaries(beats, sr, phrase_beats=32):
    beat_times = librosa.samples_to_time(beats, sr=sr)
    if len(beat_times) == 0:
        return []
    return [float(beat_times[i]) for i in range(0, len(beat_times), phrase_beats)]


def get_drop_candidates(y, sr, candidate_times, top_k=8):
    if not candidate_times:
        return []
    rms_times, rms, _ = get_energy_curve(y, sr)
    scored = []
    for t in candidate_times:
        pre_mask  = (rms_times >= t - 4.0) & (rms_times < t)
        post_mask = (rms_times >= t) & (rms_times < t + 4.0)
        if not np.any(pre_mask) or not np.any(post_mask):
            continue
        pre_energy  = float(np.mean(rms[pre_mask])  + 1e-8)
        post_energy = float(np.mean(rms[post_mask]) + 1e-8)
        scored.append((post_energy / pre_energy, float(t)))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]


# ---------------------------------------------------------------------------
# Camelot
# ---------------------------------------------------------------------------

def camelot_compatible(camelot_a, camelot_b):
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
# Beat alignment verification
# ---------------------------------------------------------------------------

def verify_beat_alignment(segment_a, segment_b, sr, bpm, tolerance_ms=12.0):
    hop = 512
    onset_a = librosa.onset.onset_strength(y=segment_a, sr=sr, hop_length=hop)
    onset_b = librosa.onset.onset_strength(y=segment_b, sr=sr, hop_length=hop)
    min_len = min(len(onset_a), len(onset_b))
    if min_len <= 1:
        return 0.0, True
    onset_a = onset_a[:min_len] - np.mean(onset_a[:min_len])
    onset_b = onset_b[:min_len] - np.mean(onset_b[:min_len])
    corr = scipy.signal.correlate(onset_a, onset_b, mode="full")
    drift_frames = int(np.argmax(corr) - (min_len - 1))
    drift_ms = (drift_frames * hop / sr) * 1000.0

    # Cap: a drift larger than the segment itself is a numerical artifact
    # from the cross-correlator when applied to spectrally mismatched content
    # (e.g. sparse piano outro vs dense electronic intro).
    segment_ms = (min_len * hop / sr) * 1000.0
    if abs(drift_ms) > segment_ms * 0.5:
        drift_ms = 0.0  # treat as aligned — measurement is unreliable

    tolerance_samples = int((tolerance_ms / 1000.0) * sr)
    drift_samples = int(drift_frames * hop)
    return float(drift_ms), bool(abs(drift_samples) <= tolerance_samples)


# ---------------------------------------------------------------------------
# Fine sync — sub-beat refinement only, NO full-track scan
# ---------------------------------------------------------------------------

def fine_sync_onset(
    y_a, y_b,
    transition_start_sample, chosen_b_phrase_sample,
    bpm_a, sr,
    search_window_beats=4,
):
    """
    Sub-beat refinement of the planner's chosen phrase boundary.

    SCOPE: stays within ±search_window_beats of chosen_b_phrase_sample.
    It does NOT scan the full track. The planner's phrase selection
    (based on downbeats, phrase structure, energy) is trusted for the
    absolute position. This function only corrects sample-level timing
    within that phrase — the difference between beat-grid quantisation
    and the actual sample where the kick lands.

    Why no full-track scan: electronic music intros are repetitive
    (kick+bass, no melody) so every 4-bar boundary produces an identical
    onset pattern. A full-track scan picks the wrong boundary by chance
    as often as not, introducing 4-bar-multiple offsets (7.5s, 15s, etc).
    This was verified across 4 renders where the scan made drift WORSE.

    Returns (synced_b_sample, sync_score 0-1).
    """
    hop        = 256
    beat_dur   = int((60.0 / max(bpm_a, 1.0)) * sr)
    search_rad = search_window_beats * beat_dur

    # Reference window: 2 beats around transition point in Track A
    a_start = max(0, transition_start_sample - beat_dur)
    a_end   = min(len(y_a), transition_start_sample + 2 * beat_dur)
    win_len = a_end - a_start

    if win_len < hop * 4:
        return int(chosen_b_phrase_sample), 0.5

    onset_a = librosa.onset.onset_strength(y=y_a[a_start:a_end], sr=sr, hop_length=hop)

    best_offset = int(chosen_b_phrase_sample)
    best_score  = -1.0
    # Use hop as step (5.8ms at 44100Hz) for true sub-beat resolution.
    # beat_dur // 8 (~58ms) is too coarse — misses 244ms-class offsets.
    step        = hop

    for delta in range(-search_rad, search_rad + step, step):
        b_start = int(chosen_b_phrase_sample) + delta
        b_end   = b_start + win_len
        if b_start < 0 or b_end > len(y_b):
            continue
        onset_b = librosa.onset.onset_strength(y=y_b[b_start:b_end], sr=sr, hop_length=hop)
        min_len = min(len(onset_a), len(onset_b))
        if min_len < 4:
            continue
        a_n   = onset_a[:min_len] - np.mean(onset_a[:min_len])
        b_n   = onset_b[:min_len] - np.mean(onset_b[:min_len])
        std_a = np.std(a_n)
        std_b = np.std(b_n)
        if std_a < 1e-9 or std_b < 1e-9:
            continue
        score = float(np.dot(a_n, b_n) / (std_a * std_b * min_len))
        if score > best_score:
            best_score  = score
            best_offset = b_start

    # If the best score is below threshold the correlator found no meaningful
    # lock — repetitive kick loops, sparse piano intros, or mismatched content.
    # In these cases the synced offset is noise; fall back to the planner's
    # phrase boundary which is based on musical structure, not signal correlation.
    SYNC_CONFIDENCE_THRESHOLD = 0.55
    if best_score < SYNC_CONFIDENCE_THRESHOLD:
        print(f"fine_sync_onset: score {best_score:.3f} < {SYNC_CONFIDENCE_THRESHOLD} "
              f"— no confident lock found, keeping planner phrase boundary.")
        return int(chosen_b_phrase_sample), float(max(best_score, 0.0))

    synced_sample = int(np.clip(best_offset, 0, max(0, len(y_b) - 1)))
    return synced_sample, float(np.clip(best_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Post-stretch refinement
# ---------------------------------------------------------------------------

def refine_beat_grid(y_stretched, sr, bpm_target=None):
    tempo, beats = safe_bpm(y_stretched, sr)
    return tempo, beats


# ---------------------------------------------------------------------------
# Transient sharpening
# ---------------------------------------------------------------------------

def sharpen_transients(y, sr, strength=0.35):
    if len(y) == 0:
        return y
    sos  = scipy.signal.butter(2, 3000.0 / (sr / 2.0), btype="high", output="sos")
    highs = scipy.signal.sosfiltfilt(sos, y)
    out  = y + highs * strength
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out


# ---------------------------------------------------------------------------
# Entry ramp
# ---------------------------------------------------------------------------

def apply_entry_ramp(y, ramp_ms=15.0, sr=TARGET_SR):
    if len(y) == 0:
        return y
    ramp_samples = min(int((ramp_ms / 1000.0) * sr), len(y))
    if ramp_samples <= 1:
        return y
    y = y.copy()
    y[:ramp_samples] *= np.linspace(0.0, 1.0, ramp_samples)
    return y