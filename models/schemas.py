from pydantic import BaseModel
from typing import Optional, Literal, List

CAMELOT_MAP = {
    ("A", "minor"): "8A",
    ("E", "minor"): "9A",
    ("B", "minor"): "10A",
    ("F#", "minor"): "11A",
    ("C#", "minor"): "12A",
    ("G#", "minor"): "1A",
    ("D#", "minor"): "2A",
    ("A#", "minor"): "3A",
    ("F", "minor"): "4A",
    ("C", "minor"): "5A",
    ("G", "minor"): "6A",
    ("D", "minor"): "7A",

    ("C", "major"): "8B",
    ("G", "major"): "9B",
    ("D", "major"): "10B",
    ("A", "major"): "11B",
    ("E", "major"): "12B",
    ("B", "major"): "1B",
    ("F#", "major"): "2B",
    ("C#", "major"): "3B",
    ("G#", "major"): "4B",
    ("D#", "major"): "5B",
    ("A#", "major"): "6B",
    ("F", "major"): "7B",
}

class FXParameters(BaseModel):
    apply_reverb_tail: bool = False
    loop_track_a: bool = False
    hpf_sweep_end_freq: Optional[float] = None

class AutoRenderRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    preferred_mix_duration: int = 30
    output_dir: str = "outputs"
    
class TransitionRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    transition_start_time: float
    track_b_entry_time: Optional[float] = None
    mix_duration: int = 30
    output_dir: str = "outputs"

    transition_strategy: Literal[
        "bass_swap",
        "hpf_sweep",
        "auto_loop",
        "reverb_wash",
        "drop_mix",
        "harmonic_mix",
        "phrase_mix",
        "lpf_sweep",
        "echo_out",
        "loop_roll",
        "long_eq_blend",
        "energy_blend",
        "percussion_blend",
        "breakdown_blend",
        "ambient_transition",
        "techno_filter_drive"
    ] = "bass_swap"

    fx_parameters: FXParameters = FXParameters()

class PlanTransitionRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    preferred_mix_duration: int = 30
    
class ScanTrackRequest(BaseModel):
    track_path: str
    library_path: str = "storage/library_metadata.json"


class ScanFolderRequest(BaseModel):
    folder_path: str
    library_path: str = "storage/library_metadata.json"

class RecommendNextRequest(BaseModel):
    current_track_id: str
    library_path: str = "storage/library_metadata.json"
    preferred_mix_duration: int = 30
    max_results: int = 5

class SmartRouteRequest(BaseModel):
    current_track_id: str
    library_path: str = "storage/library_metadata.json"
    preferred_mix_duration: int = 30
    output_dir: str = "rendered_clips"
    render: bool = False
    
class CreateQueueRequest(BaseModel):
    queue_path: str = "storage/queue_state.json"


class AddTrackToQueueRequest(BaseModel):
    track_id: str
    queue_path: str = "storage/queue_state.json"
    library_path: str = "storage/library_metadata.json"


class AdvanceQueueRequest(BaseModel):
    queue_path: str = "storage/queue_state.json"


class QueueStateResponse(BaseModel):
    status: str
    queue_path: str
    current_track_id: Optional[str]
    upcoming_track_ids: List[str]
    
class QueueSmartRenderRequest(BaseModel):
    queue_path: str = "storage/queue_state.json"
    library_path: str = "storage/library_metadata.json"
    preferred_mix_duration: int = 30
    output_dir: str = "rendered_clips"
    auto_advance: bool = False

class SetlistAddTransitionRequest(BaseModel):
    current_track_id: str
    next_track_id: str
    transition_file: str
    transition_start_time: float
    track_b_entry_time: float
    strategy: str
    setlist_path: str = "storage/setlist_state.json"


class SetlistStateRequest(BaseModel):
    setlist_path: str = "storage/setlist_state.json"

class AssemblePlaybackRequest(BaseModel):
    setlist_path: str = "storage/setlist_state.json"
    library_path: str = "storage/library_metadata.json"
    output_path: str = "rendered_clips/final_playback_mix.wav"