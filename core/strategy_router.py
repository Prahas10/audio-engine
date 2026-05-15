# Determines the most appropriate DJ transition technique based on BPM difference and harmonic compatibility.
from fastapi import HTTPException
from core.transitions import *
import math
# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
 
MIN_CONFIDENT_SCORE = 0.35
AUTO_LOOP_BOOST_FLOOR = 0.90
FALLBACK_BPM_DELTA = 12
DURATION_SPREAD_SECONDS = 16.0
RHYTHMIC_TAU_BPM = 4.0
 
 
# ---------------------------------------------------------------------------
# Strategy tables
# ---------------------------------------------------------------------------
 
STRATEGY_REQUIREMENTS = {
    "harmonic_mix":       {"harmonic": True,  "min_sync": 0.5,  "max_bpm_delta": 8,  "min_dur": 24},
    "phrase_mix":         {"harmonic": None,  "min_sync": 0.5,  "max_bpm_delta": 10, "min_dur": 16},
    "bass_swap":          {"harmonic": None,  "min_sync": 0.6,  "max_bpm_delta": 6,  "min_dur": 16},
    "hpf_sweep":          {"harmonic": None,  "min_sync": 0.3,  "max_bpm_delta": 14, "min_dur": 12},
    "lpf_sweep":          {"harmonic": None,  "min_sync": 0.3,  "max_bpm_delta": 14, "min_dur": 12},
    "reverb_wash":        {"harmonic": None,  "min_sync": 0.2,  "max_bpm_delta": 99, "min_dur": 8},
    "echo_out":           {"harmonic": None,  "min_sync": 0.2,  "max_bpm_delta": 99, "min_dur": 6},
    "loop_roll":          {"harmonic": None,  "min_sync": 0.55, "max_bpm_delta": 8,  "min_dur": 8},
    "long_eq_blend":      {"harmonic": True,  "min_sync": 0.5,  "max_bpm_delta": 6,  "min_dur": 32},
    "ambient_transition": {"harmonic": None,  "min_sync": 0.2,  "max_bpm_delta": 99, "min_dur": 16},
    "breakdown_blend":    {"harmonic": None,  "min_sync": 0.4,  "max_bpm_delta": 8,  "min_dur": 24},
    "energy_blend":       {"harmonic": None,  "min_sync": 0.45, "max_bpm_delta": 8,  "min_dur": 20},
    "drop_mix":           {"harmonic": None,  "min_sync": 0.65, "max_bpm_delta": 4,  "min_dur": 8},
    "percussion_blend":   {"harmonic": None,  "min_sync": 0.6,  "max_bpm_delta": 5,  "min_dur": 16},
    "techno_filter_drive":{"harmonic": None,  "min_sync": 0.5,  "max_bpm_delta": 6,  "min_dur": 16},
    "auto_loop":          {"harmonic": None,  "min_sync": 0.0,  "max_bpm_delta": 99, "min_dur": 0},
}
 
STRATEGY_WEIGHTS = {
    "harmonic_mix":        [0.40, 0.25, 0.10, 0.15, 0.05, 0.05],
    "phrase_mix":          [0.15, 0.25, 0.15, 0.20, 0.20, 0.05],
    "bass_swap":           [0.10, 0.20, 0.25, 0.30, 0.10, 0.05],
    "hpf_sweep":           [0.05, 0.30, 0.15, 0.15, 0.15, 0.20],
    "lpf_sweep":           [0.05, 0.30, 0.15, 0.15, 0.15, 0.20],
    "reverb_wash":         [0.05, 0.10, 0.20, 0.10, 0.20, 0.35],
    "echo_out":            [0.05, 0.10, 0.15, 0.10, 0.30, 0.30],
    "loop_roll":           [0.05, 0.25, 0.20, 0.30, 0.15, 0.05],
    "long_eq_blend":       [0.35, 0.20, 0.15, 0.15, 0.05, 0.10],
    "ambient_transition":  [0.10, 0.05, 0.30, 0.05, 0.20, 0.30],
    "breakdown_blend":     [0.15, 0.15, 0.35, 0.15, 0.15, 0.05],
    "energy_blend":        [0.10, 0.20, 0.35, 0.15, 0.10, 0.10],
    "drop_mix":            [0.10, 0.15, 0.30, 0.30, 0.10, 0.05],
    "percussion_blend":    [0.05, 0.25, 0.20, 0.35, 0.10, 0.05],
    "techno_filter_drive": [0.05, 0.20, 0.35, 0.20, 0.10, 0.10],
    "auto_loop":           [0.00, 0.10, 0.10, 0.10, 0.30, 0.40],
}
 
IDEAL_ENERGY = {
    "harmonic_mix": 0.55, "phrase_mix": 0.55, "bass_swap": 0.75,
    "hpf_sweep": 0.50,    "lpf_sweep": 0.45,  "reverb_wash": 0.30,
    "echo_out": 0.35,     "loop_roll": 0.65,  "long_eq_blend": 0.55,
    "ambient_transition": 0.20, "breakdown_blend": 0.25, "energy_blend": 0.65,
    "drop_mix": 0.85,     "percussion_blend": 0.70, "techno_filter_drive": 0.80,
    "auto_loop": 0.50,
}
 
IDEAL_DURATION = {
    "harmonic_mix": 48,  "phrase_mix": 32,   "bass_swap": 24,
    "hpf_sweep": 20,     "lpf_sweep": 20,    "reverb_wash": 32,
    "echo_out": 24,      "loop_roll": 12,    "long_eq_blend": 56,
    "ambient_transition": 48, "breakdown_blend": 40, "energy_blend": 32,
    "drop_mix": 12,      "percussion_blend": 24, "techno_filter_drive": 24,
    "auto_loop": 16,
}
 
EARLY_STRATEGIES = {"long_eq_blend", "energy_blend", "ambient_transition", "harmonic_mix"}
MID_STRATEGIES = {
    "phrase_mix", "bass_swap", "percussion_blend", "loop_roll",
    "breakdown_blend", "drop_mix", "techno_filter_drive",
}
LATE_STRATEGIES = {"echo_out", "reverb_wash", "auto_loop", "hpf_sweep", "lpf_sweep"}
 
 
def _validate_tables():
    for s, w in STRATEGY_WEIGHTS.items():
        if abs(sum(w) - 1.0) > 1e-6:
            raise ValueError(f"STRATEGY_WEIGHTS[{s!r}] sums to {sum(w)}, not 1.0")
        if s not in STRATEGY_REQUIREMENTS:
            raise ValueError(f"STRATEGY_WEIGHTS[{s!r}] has no requirement entry")
    for s in STRATEGY_REQUIREMENTS:
        if s not in STRATEGY_WEIGHTS:
            raise ValueError(f"STRATEGY_REQUIREMENTS[{s!r}] has no weight entry")
    all_positioned = EARLY_STRATEGIES | MID_STRATEGIES | LATE_STRATEGIES
    missing = set(STRATEGY_REQUIREMENTS) - all_positioned
    if missing:
        import warnings
        warnings.warn(f"Strategies without position grouping: {sorted(missing)}")
 
 
_validate_tables()
 
 
def _harmonic_factor(harmonic_ok, key_confidence_a, key_confidence_b):
    confidence = min(key_confidence_a, key_confidence_b)
    if harmonic_ok:
        return 0.5 + 0.5 * confidence
    return max(0.0, 0.2 * confidence)
 
 
def _rhythmic_factor(bpm_a, bpm_b):
    return math.exp(-abs(bpm_a - bpm_b) / RHYTHMIC_TAU_BPM)
 
 
def _energy_factor(energy_score, strategy):
    ideal = IDEAL_ENERGY.get(strategy, 0.5)
    return max(0.0, 1.0 - abs(energy_score - ideal) / 0.5)
 
 
def _sync_factor(sync_accuracy):
    return float(max(0.0, min(1.0, sync_accuracy)))
 
 
def _position_factor(best_time, song_duration_a, strategy):
    pos = best_time / max(song_duration_a, 1e-6)
    pos = max(0.0, min(1.0, pos))
    if strategy in EARLY_STRATEGIES:
        return max(0.0, 1.0 - pos / 0.65)
    if strategy in MID_STRATEGIES:
        return max(0.0, 1.0 - abs(pos - 0.75) / 0.25)
    if strategy in LATE_STRATEGIES:
        return max(0.0, (pos - 0.65) / 0.35)
    return 0.5
 
 
def _duration_factor(mix_duration, strategy):
    ideal = IDEAL_DURATION.get(strategy, 32)
    return math.exp(-0.5 * ((mix_duration - ideal) / DURATION_SPREAD_SECONDS) ** 2)
 
 
def _is_eligible(strategy, harmonic_ok, bpm_delta, sync_accuracy, mix_duration):
    req = STRATEGY_REQUIREMENTS[strategy]
    if req["harmonic"] is True and not harmonic_ok:
        return False
    if sync_accuracy < req["min_sync"]:
        return False
    if bpm_delta > req["max_bpm_delta"]:
        return False
    if mix_duration < req["min_dur"]:
        return False
    return True
 
 
def _safe_fallback(bpm_delta, mix_duration):
    echo_min = STRATEGY_REQUIREMENTS["echo_out"]["min_dur"]
    if bpm_delta <= FALLBACK_BPM_DELTA and mix_duration >= echo_min:
        return "echo_out"
    return "reverb_wash"
 
# Returns default transition duration based on selected transition strategy
def get_strategy_mix_duration(transition_strategy: str, bpm: float) -> float:
    """
    Returns a phrase-aligned mix duration in seconds for the given strategy and BPM.
    
    All durations are computed as multiples of one 8-bar phrase (32 beats) at the
    playing BPM, so transitions always start and end on a musically correct boundary
    regardless of tempo.
    
    phrase_duration = (60 / bpm) * 32
    """
    seconds_per_phrase = (60.0 / bpm) * 32

    # Number of 8-bar phrases each strategy needs to complete musically.
    # 0.5 = half a phrase (16 beats) for instant cuts.
    strategy_phrases = {
        "drop_mix":            0.5,
        "loop_roll":           1.0,
        "echo_out":            1.0,
        "reverb_wash":         2.0,
        "techno_filter_drive": 2.0,
        "bass_swap":           2.0,
        "hpf_sweep":           3.0,
        "lpf_sweep":           3.0,
        "percussion_blend":    2.0,
        "phrase_mix":          3.0,
        "energy_blend":        3.0,
        "auto_loop":           3.0,
        "harmonic_mix":        4.0,
        "breakdown_blend":     4.0,
        "long_eq_blend":       5.0,
        "ambient_transition":  6.0,
    }

    phrases = strategy_phrases.get(transition_strategy, 2.0)
    return round(seconds_per_phrase * phrases, 3)

def choose_strategy_with_scores(
    harmonic_ok,
    bpm_a,
    bpm_b,
    best_time,
    song_duration_a,
    mix_duration,
    energy_score=0.5,
    runway_score_value=1.0,
    sync_accuracy=0.5,
    key_confidence_a=0.7,
    key_confidence_b=0.7,
):
    """
    Returns (best_strategy, scores_dict).
 
    scores_dict is sorted by score (descending) and contains only eligible
    strategies. On fallback (top score < MIN_CONFIDENT_SCORE), scores_dict
    is still returned for diagnostics.
 
    runway_score_value is accepted for API compatibility but not used in
    scoring; runway shortage is detected via remaining < mix_duration.
    """
    bpm_delta = abs(bpm_a - bpm_b)
    remaining = song_duration_a - best_time
    runway_short = remaining < mix_duration
 
    f_harmonic = _harmonic_factor(harmonic_ok, key_confidence_a, key_confidence_b)
    f_rhythmic = _rhythmic_factor(bpm_a, bpm_b)
    f_sync = _sync_factor(sync_accuracy)
 
    scores = {}
    for strategy in STRATEGY_REQUIREMENTS:
        if strategy == "auto_loop" and not runway_short:
            continue
        if not _is_eligible(strategy, harmonic_ok, bpm_delta, sync_accuracy, mix_duration):
            continue
        f_energy = _energy_factor(energy_score, strategy)
        f_position = _position_factor(best_time, song_duration_a, strategy)
        f_duration = _duration_factor(mix_duration, strategy)
        factors = (f_harmonic, f_rhythmic, f_energy, f_sync, f_position, f_duration)
        weights = STRATEGY_WEIGHTS[strategy]
        scores[strategy] = sum(f * w for f, w in zip(factors, weights))
 
    if runway_short and "auto_loop" in scores:
        scores["auto_loop"] = max(scores["auto_loop"], AUTO_LOOP_BOOST_FLOOR)
 
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked_dict = dict(ranked)
 
    if not ranked or ranked[0][1] < MIN_CONFIDENT_SCORE:
        return _safe_fallback(bpm_delta, mix_duration), ranked_dict
 
    return ranked[0][0], ranked_dict
 
 
def _hpf_sweep_dispatch(segment_a, segment_b, sr, fx_parameters):
    end_freq = getattr(fx_parameters, "hpf_sweep_end_freq", None) if fx_parameters else None
    if end_freq is None:
        end_freq = 3500.0   # FIX: was 5000 — too aggressive
    return hpf_sweep_transition(segment_a, segment_b, sr, end_freq=end_freq)
 
 
def _lpf_sweep_dispatch(segment_a, segment_b, sr, fx_parameters):
    end_freq = getattr(fx_parameters, "lpf_sweep_end_freq", None) if fx_parameters else None
    if end_freq is None:
        end_freq = 400.0    # FIX: was 300 — completely muffled
    return lpf_sweep_transition(segment_a, segment_b, sr, end_freq=end_freq)
 
 
def _bpm_from_fx(fx_parameters, default=128.0):
    """Extract bpm from fx_parameters if present, else use default."""
    return float(getattr(fx_parameters, "bpm", None) or default)
 
def _auto_loop_dispatch(segment_a, segment_b, sr, fx_parameters):
    return harmonic_mix_transition(segment_a, segment_b, sr)
 
 
# FIX: echo_out, loop_roll, drop_mix all need BPM to work correctly.
# They now read bpm from fx_parameters (added to FXParameters schema).
_TRANSITION_DISPATCH = {
    "bass_swap":lambda a, b, sr, fx: bass_swap_transition(a, b, sr),
    "harmonic_mix":lambda a, b, sr, fx: harmonic_mix_transition(a, b, sr),
    "phrase_mix":lambda a, b, sr, fx: phrase_mix_transition(a, b, sr),
    "hpf_sweep":_hpf_sweep_dispatch,
    "lpf_sweep":_lpf_sweep_dispatch,
    "reverb_wash":lambda a, b, sr, fx: reverb_wash_transition(a, b, sr),
    "drop_mix":lambda a, b, sr, fx: drop_mix_transition(a, b, sr, bpm=_bpm_from_fx(fx)),
    "echo_out":lambda a, b, sr, fx: echo_out_transition(a, b, sr, bpm=_bpm_from_fx(fx)),
    "loop_roll":lambda a, b, sr, fx: loop_roll_transition(a, b, sr, bpm=_bpm_from_fx(fx)),
    "long_eq_blend":lambda a, b, sr, fx: long_eq_blend_transition(a, b, sr),
    "energy_blend":lambda a, b, sr, fx: energy_blend_transition(a, b, sr),
    "percussion_blend":lambda a, b, sr, fx: percussion_blend_transition(a, b, sr),
    "breakdown_blend":lambda a, b, sr, fx: breakdown_blend_transition(a, b, sr),
    "ambient_transition":lambda a, b, sr, fx: ambient_transition(a, b, sr),
    "techno_filter_drive":lambda a, b, sr, fx: techno_filter_drive_transition(a, b, sr),
    "auto_loop":_auto_loop_dispatch,
}
 
 
def apply_transition_strategy(segment_a, segment_b, sr, transition_strategy, fx_parameters=None):
    handler = _TRANSITION_DISPATCH.get(transition_strategy)
    if handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transition_strategy: {transition_strategy}",
        )
    return handler(segment_a, segment_b, sr, fx_parameters)
 


