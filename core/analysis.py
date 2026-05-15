import numpy as np
import librosa


TARGET_SR = 44100

# Wraps librosa's beat tracking to safely return the estimated tempo and beat timestamps.
def safe_bpm(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    tempo = float(np.asarray(tempo).squeeze())
    return tempo, beats.astype(int)

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
    end_search = song_duration * 0.88

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

    phrase_times = []
    for i in range(0, len(beat_times), phrase_beats):
        phrase_times.append(float(beat_times[i]))

    return phrase_times


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

# Calculates the Root Mean Square (RMS) of an audio signal to determine its average loudness.
def rms_level(y):
    return np.sqrt(np.mean(y ** 2) + 1e-9)