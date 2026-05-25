"""
core/stem_analysis.py

Demucs-based stem analysis for smarter transition planning.

Design:
  - Runs at SCAN time, not render time. Results stored in library metadata.
  - Extracts four stems: drums, bass, other (melodic), vocals.
  - Stores only RMS energy ENVELOPES (10 Hz) — not raw stems.
    Storage: ~56 KB per track (vs 4× file size for raw stems).
  - Rendering still uses the original full-mix audio.
  - Graceful degradation: if Demucs unavailable or fails, planner
    continues with existing RMS-only analysis.

Envelope fields added to library metadata:
  stem_envelopes: {
    "drums":   [float, ...],   # RMS at 10Hz (one value per 100ms)
    "bass":    [float, ...],
    "other":   [float, ...],   # melodic/harmonic content
    "vocals":  [float, ...],
    "times":   [float, ...],   # timestamp for each value (seconds)
    "hop_ms":  100,            # envelope resolution in ms
    "model":   "htdemucs",     # which Demucs model was used
  }

Usage in planner:
  stem_envelopes = metadata.get("stem_envelopes", {})
  drums_env = stem_envelopes.get("drums", [])
  # → check if drums are active at a candidate time
  # → detect sparse intros (drums+bass both near zero)
  # → detect vocal presence for clash avoidance
"""

import os
import tempfile
import subprocess
import shutil
import numpy as np

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.audio import convert_audio
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

# Envelope resolution: one RMS value per HOP_MS milliseconds
HOP_MS      = 100          # 10 Hz envelope
DEMUCS_SR   = 44100        # Demucs native sample rate
MODEL_NAME  = "htdemucs"   # best quality 4-stem model
STEM_NAMES  = ["drums", "bass", "other", "vocals"]

# Minimum energy threshold — below this a stem is "inactive"
STEM_ACTIVE_THRESHOLD = 0.02   # normalised RMS


def demucs_available() -> bool:
    """Return True if Demucs and PyTorch are installed and usable."""
    return DEMUCS_AVAILABLE and TORCH_AVAILABLE


def _load_model(device: str = "cpu"):
    """Load and cache the Demucs model. Returns model or None on failure."""
    try:
        model = get_model(MODEL_NAME)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[stem_analysis] Failed to load Demucs model: {e}")
        return None


def _compute_envelope(waveform: np.ndarray, sr: int, hop_ms: int = HOP_MS) -> np.ndarray:
    """
    Compute RMS energy envelope of a mono waveform.

    Args:
        waveform: 1D numpy array, audio samples
        sr:       sample rate
        hop_ms:   envelope resolution in milliseconds

    Returns:
        1D numpy array of RMS values, one per hop_ms window.
    """
    hop_samples = int(sr * hop_ms / 1000)
    if hop_samples <= 0 or len(waveform) == 0:
        return np.array([], dtype=np.float32)

    # Pad so length is divisible by hop_samples
    n_hops  = int(np.ceil(len(waveform) / hop_samples))
    padded  = np.zeros(n_hops * hop_samples, dtype=np.float32)
    padded[:len(waveform)] = waveform.astype(np.float32)

    frames  = padded.reshape(n_hops, hop_samples)
    rms     = np.sqrt(np.mean(frames ** 2, axis=1))
    return rms.astype(np.float32)


def _normalise_envelopes(envelopes: dict) -> dict:
    """
    Normalise all stem envelopes to 0-1 range using the global maximum
    across all stems. This preserves relative stem levels (drums louder
    than bass, etc.) while putting everything in a comparable range.

    Falls back to per-stem normalisation if global max is near zero.
    """
    all_values = np.concatenate([v for v in envelopes.values() if len(v) > 0])
    global_max = float(np.max(all_values)) if len(all_values) > 0 else 1.0

    if global_max < 1e-6:
        # Track is essentially silent — return zeros
        return {k: np.zeros_like(v) for k, v in envelopes.items()}

    return {k: (v / global_max).astype(np.float32) for k, v in envelopes.items()}


def extract_stem_envelopes(
    track_path: str,
    device: str = "cpu",
    model=None,
) -> dict:
    """
    Run Demucs on a track and return stem energy envelopes.

    Args:
        track_path: path to audio file (wav, mp3, flac, etc.)
        device:     "cpu" or "cuda"
        model:      pre-loaded Demucs model (pass to avoid reloading per track)

    Returns:
        dict with keys: drums, bass, other, vocals, times, hop_ms, model
        Returns empty dict if Demucs is unavailable or fails.
    """
    if not demucs_available():
        print("[stem_analysis] Demucs not available — skipping stem analysis.")
        return {}

    if not os.path.exists(track_path):
        print(f"[stem_analysis] File not found: {track_path}")
        return {}

    try:
        # Load model if not provided
        if model is None:
            model = _load_model(device)
        if model is None:
            return {}

        # Load audio with torchaudio
        waveform, file_sr = torchaudio.load(track_path)

        # Demucs expects stereo at its native SR
        waveform = convert_audio(
            waveform,
            file_sr,
            model.samplerate,
            model.audio_channels,
        )

        # Add batch dimension: [1, channels, samples]
        waveform = waveform.unsqueeze(0)
        if device == "cuda" and torch.cuda.is_available():
            waveform = waveform.cuda()

        # Run separation
        with torch.no_grad():
            sources = apply_model(model, waveform, device=device)

        # sources shape: [1, n_stems, channels, samples]
        sources = sources.squeeze(0).cpu().numpy()

        # Map stem index to name (htdemucs order: drums, bass, other, vocals)
        stem_order = model.sources  # e.g. ["drums", "bass", "other", "vocals"]

        envelopes = {}
        for i, stem_name in enumerate(stem_order):
            if stem_name not in STEM_NAMES:
                continue
            # Mix to mono for envelope computation
            stem_mono = sources[i].mean(axis=0)   # [samples]
            env = _compute_envelope(stem_mono, model.samplerate, HOP_MS)
            envelopes[stem_name] = env

        # Ensure all four stems are present
        n_frames = max(len(v) for v in envelopes.values()) if envelopes else 0
        for stem_name in STEM_NAMES:
            if stem_name not in envelopes:
                envelopes[stem_name] = np.zeros(n_frames, dtype=np.float32)

        # Normalise
        envelopes = _normalise_envelopes(envelopes)

        # Build times array
        hop_s  = HOP_MS / 1000.0
        times  = np.arange(n_frames) * hop_s

        return {
            "drums":  envelopes["drums"].tolist(),
            "bass":   envelopes["bass"].tolist(),
            "other":  envelopes["other"].tolist(),
            "vocals": envelopes["vocals"].tolist(),
            "times":  [round(float(t), 3) for t in times],
            "hop_ms": HOP_MS,
            "model":  MODEL_NAME,
        }

    except Exception as e:
        print(f"[stem_analysis] Stem extraction failed for {track_path}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Query helpers — used by the planner to interrogate stem envelopes
# ---------------------------------------------------------------------------

def get_stem_energy_at(
    stem_envelopes: dict,
    stem: str,
    time_start: float,
    time_end: float,
) -> float:
    """
    Return mean stem energy in [time_start, time_end] seconds.
    Returns 0.0 if stem data not available.
    """
    values = stem_envelopes.get(stem, [])
    times  = stem_envelopes.get("times", [])
    hop_ms = stem_envelopes.get("hop_ms", HOP_MS)

    if not values or not times:
        return 0.0

    times_arr  = np.asarray(times)
    values_arr = np.asarray(values)

    mask = (times_arr >= time_start) & (times_arr <= time_end)
    if not np.any(mask):
        return 0.0

    return float(np.mean(values_arr[mask]))


def is_stem_active(
    stem_envelopes: dict,
    stem: str,
    time_start: float,
    time_end: float,
    threshold: float = STEM_ACTIVE_THRESHOLD,
) -> bool:
    """Return True if mean stem energy in window exceeds threshold."""
    return get_stem_energy_at(stem_envelopes, stem, time_start, time_end) > threshold


def is_sparse_intro(
    stem_envelopes: dict,
    time_start: float,
    time_end: float,
    drums_threshold: float = 0.05,
    bass_threshold: float = 0.05,
) -> bool:
    """
    Return True if the window has near-zero drums AND bass energy.
    A sparse intro means: piano-only, ambient, or pad-only section.
    These are problematic as B entry points unless the mix is very long.
    """
    drums_energy = get_stem_energy_at(stem_envelopes, "drums", time_start, time_end)
    bass_energy  = get_stem_energy_at(stem_envelopes, "bass",  time_start, time_end)
    return drums_energy < drums_threshold and bass_energy < bass_threshold


def has_vocal_clash(
    stem_envelopes_a: dict,
    stem_envelopes_b: dict,
    a_start: float,
    b_start: float,
    mix_duration: float,
    vocal_threshold: float = 0.10,
) -> bool:
    """
    Return True if both Track A and Track B have significant vocal energy
    during the mix window. Vocal-over-vocal is the most audible clash.

    a_start: A's transition start time (seconds into Track A)
    b_start: B's entry time (seconds into Track B)
    mix_duration: length of the overlap window
    """
    a_vocal = get_stem_energy_at(
        stem_envelopes_a, "vocals",
        a_start, a_start + mix_duration
    )
    b_vocal = get_stem_energy_at(
        stem_envelopes_b, "vocals",
        b_start, b_start + mix_duration
    )
    return a_vocal > vocal_threshold and b_vocal > vocal_threshold


def stem_activity_profile(
    stem_envelopes: dict,
    time_start: float,
    time_end: float,
) -> dict:
    """
    Return a summary of stem activity in a window.
    Used for strategy selection and debug logging.
    """
    return {
        stem: round(get_stem_energy_at(stem_envelopes, stem, time_start, time_end), 3)
        for stem in STEM_NAMES
    }


def find_kick_onset(
    stem_envelopes: dict,
    search_start: float = 0.0,
    search_end: float = 120.0,
    drums_threshold: float = 0.08,
    min_sustained_seconds: float = 4.0,
) -> float:
    """
    Find the first time the drums stem sustains above threshold for
    at least min_sustained_seconds. This is where the kick "arrives"
    in a sparse intro — the correct B entry point for tracks like
    Nils Hoffmann 9 Days (piano-only until bar 33).

    Returns the timestamp in seconds, or search_end if not found.
    """
    values = stem_envelopes.get("drums", [])
    times  = stem_envelopes.get("times", [])
    hop_ms = stem_envelopes.get("hop_ms", HOP_MS)

    if not values or not times:
        return search_end

    times_arr  = np.asarray(times)
    values_arr = np.asarray(values)

    min_frames = int(min_sustained_seconds * 1000 / hop_ms)

    mask = (times_arr >= search_start) & (times_arr <= search_end)
    t_window = times_arr[mask]
    v_window = values_arr[mask]

    if len(v_window) == 0:
        return search_end

    active  = (v_window > drums_threshold).astype(int)
    run_len = 0

    for i, a in enumerate(active):
        if a:
            run_len += 1
            if run_len >= min_frames:
                # Return time at start of this run
                start_idx = i - run_len + 1
                return float(t_window[start_idx])
        else:
            run_len = 0

    return search_end