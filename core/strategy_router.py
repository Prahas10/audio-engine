# Determines the most appropriate DJ transition technique based on BPM difference and harmonic compatibility.
from fastapi import HTTPException
from core.transitions import *

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
    key_confidence_b=0.7
):
    bpm_delta = abs(bpm_a - bpm_b)
    remaining = song_duration_a - best_time
    song_position = best_time / song_duration_a

    key_confidence = min(key_confidence_a, key_confidence_b)

    scores = {
        "harmonic_mix": 0.0,
        "phrase_mix": 0.0,
        "bass_swap": 0.0,
        "hpf_sweep": 0.0,
        "lpf_sweep": 0.0,
        "reverb_wash": 0.0,
        "echo_out": 0.0,
        "loop_roll": 0.0,
        "long_eq_blend": 0.0,
        "ambient_transition": 0.0,
        "breakdown_blend": 0.0,
        "energy_blend": 0.0,
        "drop_mix": 0.0,
        "percussion_blend": 0.0,
        "techno_filter_drive": 0.0,
        "auto_loop": 0.0,
    }

    # Hard constraints / safety routing
    if remaining < mix_duration:
        scores["auto_loop"] += 80

    if bpm_delta > 12:
        scores["reverb_wash"] += 70
        scores["echo_out"] += 45
        scores["ambient_transition"] += 35

    # Harmonic compatibility with confidence
    if harmonic_ok and key_confidence >= 0.45:
        scores["harmonic_mix"] += 45
        scores["long_eq_blend"] += 35
        scores["energy_blend"] += 25
        scores["phrase_mix"] += 20
    elif harmonic_ok:
        scores["harmonic_mix"] += 20
        scores["long_eq_blend"] += 20
        scores["phrase_mix"] += 15
    else:
        scores["reverb_wash"] += 35
        scores["echo_out"] += 30
        scores["hpf_sweep"] += 20
        scores["ambient_transition"] += 15

    # BPM compatibility
    if bpm_delta <= 2:
        scores["long_eq_blend"] += 35
        scores["harmonic_mix"] += 25
        scores["phrase_mix"] += 25
        scores["percussion_blend"] += 15
    elif bpm_delta <= 5:
        scores["energy_blend"] += 30
        scores["phrase_mix"] += 25
        scores["bass_swap"] += 20
        scores["long_eq_blend"] += 15
    elif bpm_delta <= 8:
        scores["hpf_sweep"] += 25
        scores["lpf_sweep"] += 20
        scores["energy_blend"] += 15
    elif bpm_delta <= 12:
        scores["echo_out"] += 30
        scores["reverb_wash"] += 25
        scores["hpf_sweep"] += 20

    # Sync accuracy
    scores["bass_swap"] += sync_accuracy * 20
    scores["phrase_mix"] += sync_accuracy * 20
    scores["long_eq_blend"] += sync_accuracy * 15
    scores["percussion_blend"] += sync_accuracy * 15

    if sync_accuracy < 0.45:
        scores["reverb_wash"] += 25
        scores["echo_out"] += 20
        scores["ambient_transition"] += 15
        scores["bass_swap"] -= 20
        scores["drop_mix"] -= 20

    # Energy profile
    scores["energy_blend"] += energy_score * 35
    scores["phrase_mix"] += energy_score * 20
    scores["long_eq_blend"] += energy_score * 15

    if energy_score < 0.35:
        scores["ambient_transition"] += 25
        scores["breakdown_blend"] += 25
        scores["reverb_wash"] += 15

    if energy_score > 0.70:
        scores["bass_swap"] += 20
        scores["drop_mix"] += 15
        scores["techno_filter_drive"] += 15
        scores["percussion_blend"] += 15

    # Song position
    if song_position < 0.65:
        scores["long_eq_blend"] += 20
        scores["energy_blend"] += 15
        scores["ambient_transition"] += 10
    elif song_position < 0.85:
        scores["phrase_mix"] += 25
        scores["bass_swap"] += 20
        scores["percussion_blend"] += 15
        scores["harmonic_mix"] += 10
    else:
        scores["echo_out"] += 25
        scores["reverb_wash"] += 20
        scores["auto_loop"] += 15

    # Mix duration
    if mix_duration >= 45:
        scores["long_eq_blend"] += 35
        scores["ambient_transition"] += 20
        scores["breakdown_blend"] += 20
        scores["harmonic_mix"] += 15
    elif mix_duration >= 30:
        scores["phrase_mix"] += 20
        scores["energy_blend"] += 20
        scores["bass_swap"] += 15
        scores["harmonic_mix"] += 10
    else:
        scores["drop_mix"] += 35
        scores["loop_roll"] += 25
        scores["echo_out"] += 20

    # Runway
    scores["long_eq_blend"] += runway_score_value * 15
    scores["phrase_mix"] += runway_score_value * 10
    scores["bass_swap"] += runway_score_value * 10

    if runway_score_value < 0.6:
        scores["auto_loop"] += 35
        scores["echo_out"] += 15

    # Prevent risky choices unless conditions are strong
    if not harmonic_ok:
        scores["harmonic_mix"] -= 100
        scores["long_eq_blend"] -= 20

    if bpm_delta > 8:
        scores["bass_swap"] -= 25
        scores["drop_mix"] -= 20
        scores["percussion_blend"] -= 15

    if sync_accuracy < 0.4:
        scores["drop_mix"] -= 30
        scores["techno_filter_drive"] -= 20

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[0][0], dict(ranked)



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