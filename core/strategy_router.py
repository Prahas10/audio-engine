"""
core/strategy_router.py
Strategy selection and transition dispatch.

Changes vs original:
  - All dispatch lambdas pass energy_score from fx_parameters to transitions
    that accept it (bass_swap, harmonic_mix, energy_blend, etc.)
  - get_strategy_mix_duration: unchanged
  - choose_strategy_with_scores: unchanged
"""

import math
import warnings
from fastapi import HTTPException
from core.transitions import (
    bass_swap_transition,
    harmonic_mix_transition,
    phrase_mix_transition,
    hpf_sweep_transition,
    lpf_sweep_transition,
    reverb_wash_transition,
    drop_mix_transition,
    echo_out_transition,
    loop_roll_transition,
    long_eq_blend_transition,
    energy_blend_transition,
    percussion_blend_transition,
    breakdown_blend_transition,
    ambient_transition,
    techno_filter_drive_transition,
    vocal_safe_blend_transition,
)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MIN_CONFIDENT_SCORE    = 0.35
AUTO_LOOP_BOOST_FLOOR  = 0.90
FALLBACK_BPM_DELTA     = 12
DURATION_SPREAD_SECONDS = 16.0
RHYTHMIC_TAU_BPM       = 4.0


# ---------------------------------------------------------------------------
# Strategy tables
# ---------------------------------------------------------------------------

STRATEGY_REQUIREMENTS = {
    "harmonic_mix":        {"harmonic": True,  "min_sync": 0.4,  "max_bpm_delta": 10, "min_dur": 32},
    "phrase_mix":          {"harmonic": None,  "min_sync": 0.5,  "max_bpm_delta": 10, "min_dur": 16},
    # bass_swap: removed from auto-selection — unreliable in continuous mode.
    # Left in dispatch table for manual API use only.
    # "bass_swap": {"harmonic": None, "min_sync": 0.80, "max_bpm_delta": 3, "min_dur": 16},
    "hpf_sweep":           {"harmonic": None,  "min_sync": 0.3,  "max_bpm_delta": 14, "min_dur": 12},
    "lpf_sweep":           {"harmonic": None,  "min_sync": 0.3,  "max_bpm_delta": 14, "min_dur": 12},
    # "reverb_wash": removed from auto-selection — sounds wrong on melodic electronic music
    #                  still available for manual API use via dispatch table
    "echo_out":            {"harmonic": None,  "min_sync": 0.2,  "max_bpm_delta": 99, "min_dur": 6},
    "loop_roll":           {"harmonic": None,  "min_sync": 0.55, "max_bpm_delta": 8,  "min_dur": 8},
    "long_eq_blend":       {"harmonic": True,  "min_sync": 0.4,  "max_bpm_delta": 8,  "min_dur": 48},
    "ambient_transition":  {"harmonic": None,  "min_sync": 0.2,  "max_bpm_delta": 99, "min_dur": 16},
    "breakdown_blend":     {"harmonic": None,  "min_sync": 0.4,  "max_bpm_delta": 8,  "min_dur": 24},
    "energy_blend":        {"harmonic": None,  "min_sync": 0.45, "max_bpm_delta": 8,  "min_dur": 20},
    "drop_mix":            {"harmonic": None,  "min_sync": 0.90, "max_bpm_delta": 2,  "min_dur": 8},
    "percussion_blend":    {"harmonic": None,  "min_sync": 0.6,  "max_bpm_delta": 5,  "min_dur": 16},
    "techno_filter_drive": {"harmonic": False, "min_sync": 0.5,  "max_bpm_delta": 6,  "min_dur": 16},
    # "auto_loop": removed from auto-selection
}

STRATEGY_WEIGHTS = {
    "harmonic_mix":        [0.50, 0.15, 0.10, 0.15, 0.05, 0.05],
    "phrase_mix":          [0.10, 0.25, 0.15, 0.20, 0.20, 0.10],
    # "bass_swap":         [0.10, 0.20, 0.25, 0.30, 0.10, 0.05],  # removed
    "hpf_sweep":           [0.05, 0.30, 0.15, 0.15, 0.15, 0.20],
    "lpf_sweep":           [0.05, 0.30, 0.15, 0.15, 0.15, 0.20],
    # "reverb_wash":       [0.05, 0.10, 0.20, 0.10, 0.20, 0.35],  # removed
    "echo_out":            [0.05, 0.10, 0.15, 0.10, 0.30, 0.30],
    "loop_roll":           [0.05, 0.25, 0.20, 0.30, 0.15, 0.05],
    "long_eq_blend":       [0.45, 0.15, 0.10, 0.15, 0.05, 0.10],
    "ambient_transition":  [0.10, 0.05, 0.30, 0.05, 0.20, 0.30],
    "breakdown_blend":     [0.15, 0.15, 0.35, 0.15, 0.15, 0.05],
    "energy_blend":        [0.10, 0.20, 0.35, 0.15, 0.10, 0.10],
    "drop_mix":            [0.10, 0.15, 0.30, 0.30, 0.10, 0.05],
    "percussion_blend":    [0.20, 0.20, 0.20, 0.25, 0.10, 0.05],
    "techno_filter_drive": [0.05, 0.20, 0.35, 0.20, 0.10, 0.10],
    # "auto_loop":         [0.00, 0.10, 0.10, 0.10, 0.30, 0.40],  # removed
}

IDEAL_ENERGY = {
    "harmonic_mix": 0.55,      "phrase_mix": 0.55,   "bass_swap": 0.75,
    "hpf_sweep": 0.50,         "lpf_sweep": 0.45,    "reverb_wash": 0.30,
    "echo_out": 0.35,          "loop_roll": 0.65,    "long_eq_blend": 0.55,
    "ambient_transition": 0.20,"breakdown_blend": 0.25,"energy_blend": 0.65,
    "drop_mix": 0.85,          "percussion_blend": 0.60,"techno_filter_drive": 0.80,
    "auto_loop": 0.50,
}

IDEAL_DURATION = {
    "harmonic_mix": 60,  "phrase_mix": 48,   "bass_swap": 24,
    "hpf_sweep": 20,     "lpf_sweep": 20,    "reverb_wash": 32,
    "echo_out": 24,      "loop_roll": 12,    "long_eq_blend": 56,
    "ambient_transition": 48,"breakdown_blend": 40,"energy_blend": 32,
    "drop_mix": 12,      "percussion_blend": 24,"techno_filter_drive": 24,
    "auto_loop": 16,
}

EARLY_STRATEGIES = {"long_eq_blend", "energy_blend", "ambient_transition", "harmonic_mix"}
MID_STRATEGIES   = {
    "phrase_mix", "percussion_blend", "loop_roll",
    "breakdown_blend", "drop_mix", "techno_filter_drive",
}
LATE_STRATEGIES  = {"echo_out", "hpf_sweep", "lpf_sweep"}


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
        warnings.warn(f"Strategies without position grouping: {sorted(missing)}")


_validate_tables()


# ---------------------------------------------------------------------------
# Scoring factors
# ---------------------------------------------------------------------------

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
    if bpm_delta <= FALLBACK_BPM_DELTA and mix_duration >= STRATEGY_REQUIREMENTS["echo_out"]["min_dur"]:
        return "echo_out"
    return "reverb_wash"


# ---------------------------------------------------------------------------
# Public: get mix duration
# ---------------------------------------------------------------------------

def get_strategy_mix_duration(transition_strategy: str, bpm: float) -> float:
    if bpm is None or bpm <= 0:
        bpm = 124.0
    seconds_per_beat = 60.0 / bpm
    strategy_beats = {
        "drop_mix": 16,    "loop_roll": 16,   "echo_out": 16,
        "bass_swap": 32,   "hpf_sweep": 32,   "lpf_sweep": 32,  "reverb_wash": 32,
        "phrase_mix": 64,  "energy_blend": 64,"percussion_blend": 64,
        "long_eq_blend": 64,"harmonic_mix": 64,"breakdown_blend": 64,
        "ambient_transition": 64,"techno_filter_drive": 64,"auto_loop": 32,
    }
    beats    = strategy_beats.get(transition_strategy, 64)
    duration = seconds_per_beat * beats
    duration = max(16.0, duration)
    duration = min(30.0, duration)
    return round(duration, 3)


# ---------------------------------------------------------------------------
# Public: choose strategy
# ---------------------------------------------------------------------------

def choose_strategy_with_scores(
    harmonic_ok, bpm_a, bpm_b, best_time, song_duration_a, mix_duration,
    energy_score=0.5, runway_score_value=1.0, sync_accuracy=0.5,
    key_confidence_a=0.7, key_confidence_b=0.7,
):
    bpm_delta    = abs(bpm_a - bpm_b)
    remaining    = song_duration_a - best_time
    runway_short = remaining < mix_duration

    f_harmonic = _harmonic_factor(harmonic_ok, key_confidence_a, key_confidence_b)
    f_rhythmic = _rhythmic_factor(bpm_a, bpm_b)
    f_sync     = _sync_factor(sync_accuracy)

    scores = {}
    for strategy in STRATEGY_REQUIREMENTS:
        if strategy == "auto_loop" and not runway_short:
            continue
        if not _is_eligible(strategy, harmonic_ok, bpm_delta, sync_accuracy, mix_duration):
            continue
        f_energy   = _energy_factor(energy_score, strategy)
        f_position = _position_factor(best_time, song_duration_a, strategy)
        f_duration = _duration_factor(mix_duration, strategy)
        factors    = (f_harmonic, f_rhythmic, f_energy, f_sync, f_position, f_duration)
        weights    = STRATEGY_WEIGHTS[strategy]
        scores[strategy] = sum(f * w for f, w in zip(factors, weights))

    if runway_short and "auto_loop" in scores:
        scores["auto_loop"] = max(scores["auto_loop"], AUTO_LOOP_BOOST_FLOOR)

    ranked      = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranked_dict = dict(ranked)

    if not ranked or ranked[0][1] < MIN_CONFIDENT_SCORE:
        return _safe_fallback(bpm_delta, mix_duration), ranked_dict

    return ranked[0][0], ranked_dict


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _bpm_from_fx(fx_parameters, default=128.0):
    return float(getattr(fx_parameters, "bpm", None) or default)


def _energy_from_fx(fx_parameters, default=0.5):
    return float(getattr(fx_parameters, "energy_score", None) or default)



def _hpf_sweep_dispatch(segment_a, segment_b, sr, fx_parameters):
    end_freq     = getattr(fx_parameters, "hpf_sweep_end_freq", None) if fx_parameters else None
    end_freq     = end_freq or 3500.0
    energy_score = _energy_from_fx(fx_parameters)
    return hpf_sweep_transition(segment_a, segment_b, sr, end_freq=end_freq, energy_score=energy_score)


def _lpf_sweep_dispatch(segment_a, segment_b, sr, fx_parameters):
    end_freq     = getattr(fx_parameters, "lpf_sweep_end_freq", None) if fx_parameters else None
    end_freq     = end_freq or 400.0
    energy_score = _energy_from_fx(fx_parameters)
    return lpf_sweep_transition(segment_a, segment_b, sr, end_freq=end_freq, energy_score=energy_score)


def _auto_loop_dispatch(segment_a, segment_b, sr, fx_parameters):
    return harmonic_mix_transition(segment_a, segment_b, sr,
                                   energy_score=_energy_from_fx(fx_parameters))


_TRANSITION_DISPATCH = {
    "bass_swap":
        lambda a, b, sr, fx: bass_swap_transition(
            a, b, sr, bpm=_bpm_from_fx(fx), energy_score=_energy_from_fx(fx)),
    "harmonic_mix":
        lambda a, b, sr, fx: harmonic_mix_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "phrase_mix":
        lambda a, b, sr, fx: phrase_mix_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "hpf_sweep":       _hpf_sweep_dispatch,
    "lpf_sweep":       _lpf_sweep_dispatch,
    "reverb_wash":
        lambda a, b, sr, fx: reverb_wash_transition(a, b, sr),
    "drop_mix":
        lambda a, b, sr, fx: drop_mix_transition(
            a, b, sr, bpm=_bpm_from_fx(fx)),
    "echo_out":
        lambda a, b, sr, fx: echo_out_transition(
            a, b, sr, bpm=_bpm_from_fx(fx)),
    "loop_roll":
        lambda a, b, sr, fx: loop_roll_transition(
            a, b, sr, bpm=_bpm_from_fx(fx)),
    "long_eq_blend":
        lambda a, b, sr, fx: long_eq_blend_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "energy_blend":
        lambda a, b, sr, fx: energy_blend_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "percussion_blend":
        lambda a, b, sr, fx: percussion_blend_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "breakdown_blend":
        lambda a, b, sr, fx: breakdown_blend_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "ambient_transition":
        lambda a, b, sr, fx: ambient_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "techno_filter_drive":
        lambda a, b, sr, fx: techno_filter_drive_transition(
            a, b, sr, energy_score=_energy_from_fx(fx)),
    "auto_loop":       _auto_loop_dispatch,
}


def apply_transition_strategy(segment_a, segment_b, sr, transition_strategy, fx_parameters=None):
    handler = _TRANSITION_DISPATCH.get(transition_strategy)
    if handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported transition_strategy: {transition_strategy}",
        )
    return handler(segment_a, segment_b, sr, fx_parameters)