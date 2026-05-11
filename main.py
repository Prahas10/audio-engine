import uuid
from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rendered_clips"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Audio Mixing Engine")

app.mount("/clips", StaticFiles(directory=str(OUTPUT_DIR)), name="clips")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TransitionRequest(BaseModel):
    track_a_url: str
    track_b_url: str
    transition_start_time: float = Field(..., ge=0)
    mix_duration: int = Field(default=30, ge=1, le=120)


class TransitionResponse(BaseModel):
    audio_clip_url: str


def analyze_samples_librosa(y, sr, hop_length=512):
    tempo, beat_samples = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
        units="samples"
    )

    return {
        "beat_samples": beat_samples.astype(int),
        "duration": librosa.get_duration(y=y, sr=sr),
        "tempo_float": float(np.asarray(tempo).squeeze())
    }


def find_best_sync_point(bottom_file_beats, top_file_beats, max_mix_sample, offset, mode):
    matches_per_round = []

    bottom_file_beats = np.array(bottom_file_beats)
    top_file_beats = np.array(top_file_beats)

    if len(bottom_file_beats) == 0 or len(top_file_beats) == 0:
        return 0, 0, 0.0

    for rn in range(bottom_file_beats.shape[0]):
        try:
            zero_sync_samples = bottom_file_beats[rn] - top_file_beats[0]
            slider = top_file_beats + zero_sync_samples

            for i in range(len(slider)):
                if slider[i] > max_mix_sample:
                    slider[i] = slider[i] - max_mix_sample

            matches = []
            tested_beat_index = 0

            all_sample_beats = np.concatenate((slider, bottom_file_beats))
            all_sample_beats.sort()

            for i in range(1, all_sample_beats.shape[0]):
                if (
                    all_sample_beats[i] == all_sample_beats[tested_beat_index]
                    or abs(all_sample_beats[i] - all_sample_beats[tested_beat_index]) <= offset
                ):
                    matches.append(all_sample_beats[i])
                    matches.append(all_sample_beats[tested_beat_index])

                tested_beat_index += 1

            matches_per_round.append(len(matches) / 2 / len(top_file_beats))

        except Exception:
            matches_per_round.append(0)

    if mode == "random":
        sync_beat_number = np.random.choice(
            np.argwhere(matches_per_round == np.amax(matches_per_round)).reshape(-1)
        )
    else:
        sync_beat_number = int(np.argmax(matches_per_round))

    sync_sample = int(bottom_file_beats[sync_beat_number] - top_file_beats[0])
    sync_beat_accuracy = float(np.max(matches_per_round))

    return sync_sample, sync_beat_number, sync_beat_accuracy


def render_transition_clip(
    track_a_path,
    track_b_path,
    output_path,
    transition_start_time,
    mix_duration=30,
    sr=22050,
    offset=880
):
    track_a_path = Path(track_a_path)
    track_b_path = Path(track_b_path)

    if not track_a_path.exists():
        raise HTTPException(status_code=400, detail=f"Track A not found: {track_a_path}")

    if not track_b_path.exists():
        raise HTTPException(status_code=400, detail=f"Track B not found: {track_b_path}")

    transition_start_ms = int(transition_start_time * 1000)
    mix_duration_ms = int(mix_duration * 1000)

    track_a_seg = AudioSegment.from_file(track_a_path)
    track_b_seg = AudioSegment.from_file(track_b_path)

    if transition_start_ms >= len(track_a_seg):
        raise HTTPException(
            status_code=400,
            detail="transition_start_time is beyond Track A duration"
        )

    track_a_clip = track_a_seg[transition_start_ms:transition_start_ms + mix_duration_ms]
    track_b_clip = track_b_seg[:mix_duration_ms]

    if len(track_a_clip) < mix_duration_ms:
        track_a_clip += AudioSegment.silent(duration=mix_duration_ms - len(track_a_clip))

    if len(track_b_clip) < mix_duration_ms:
        track_b_clip += AudioSegment.silent(duration=mix_duration_ms - len(track_b_clip))

    y_a, sr = librosa.load(
        str(track_a_path),
        sr=sr,
        offset=transition_start_time,
        duration=mix_duration
    )

    y_b, sr = librosa.load(
        str(track_b_path),
        sr=sr,
        duration=mix_duration
    )

    a_data = analyze_samples_librosa(y_a, sr)
    b_data = analyze_samples_librosa(y_b, sr)

    sync_sample, _, sync_accuracy = find_best_sync_point(
        bottom_file_beats=a_data["beat_samples"],
        top_file_beats=b_data["beat_samples"],
        max_mix_sample=len(y_a),
        offset=offset,
        mode="first"
    )

    sync_time_ms = int(sync_sample / sr * 1000)

    fade_out_a = track_a_clip.fade_out(mix_duration_ms)
    fade_in_b = track_b_clip.fade_in(mix_duration_ms)

    if sync_time_ms < 0:
        mixed = fade_out_a.overlay(fade_in_b[abs(sync_time_ms):], position=0)
    else:
        mixed = fade_out_a.overlay(fade_in_b, position=sync_time_ms)

    mixed = mixed[:mix_duration_ms]

    if mixed.max_dBFS != float("-inf"):
        mixed = mixed.apply_gain(-1.0 - mixed.max_dBFS)

    mixed.export(output_path, format="wav")

    return sync_accuracy


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/v1/engine/render-transition", response_model=TransitionResponse)
def render_transition(req: TransitionRequest):
    output_filename = f"transition_test.wav"
    output_path = OUTPUT_DIR / output_filename

    render_transition_clip(
        track_a_path=req.track_a_url,
        track_b_path=req.track_b_url,
        output_path=str(output_path),
        transition_start_time=req.transition_start_time,
        mix_duration=req.mix_duration
    )

    return {
        "audio_clip_url": f"/clips/{output_filename}"
    }