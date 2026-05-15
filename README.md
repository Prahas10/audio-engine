# 🐻🎧 Audio Engine — Automated DJ Transition Engine

A headless, API-first automated DJ system that analyzes audio tracks, matches beats, and renders seamless crossfade transitions. Comes with a full Streamlit UI console for real-time control.

---

## Overview

Audio Engine is a Python backend that acts like a digital DJ. It takes audio tracks from a library, analyzes their BPM, key, and beat positions using Librosa, then intelligently plans and renders smooth transitions between them. The system is entirely headless and API-driven — every action (scanning, queuing, rendering, assembling) is exposed as a REST endpoint via FastAPI.

A companion **Streamlit UI** (`app_ui.py`) wraps all API calls into a tabbed console called the **Alalu DJ Console**, giving you a browser-based interface for library management, queue control, transition rendering, and full-set assembly.

---

## Features

- **Smart Beat Matching** — Detects BPM and beat positions of each track to find the mathematically optimal synchronization point for transitions.
- **Automated Crossfading** — Gradually fades out the outgoing track while fading in the incoming one, keeping beats aligned.
- **Multiple Transition Strategies** — The engine selects a transition strategy (e.g., standard crossfade, beat-locked mix) based on track analysis. Mix duration is configurable from 5 to 120 seconds.
- **Track Library Scanner** — Recursively scans a folder, extracts BPM, key, Camelot wheel notation, and duration for every audio file, and persists metadata to a JSON library.
- **Queue Manager** — Maintain an ordered playback queue with support for adding tracks, advancing positions, and inspecting state at any time.
- **Setlist Timeline** — Each rendered transition is logged in a persistent setlist, building a full timeline of the DJ set.
- **Smart Set Builder** — Automatically sequences your entire library into an optimized set order (by BPM and harmonic compatibility) or renders in manual queue order.
- **Playback Assembler** — Stitches all rendered transition clips into a single final WAV mix.
- **Alalu DJ Console** — A full Streamlit UI with tabs for Library, Queue, Smart Render, Setlist, Playback, and Smart Stream.

---

## Project Structure

```
audio-engine/
│
├── main.py               # FastAPI app — registers all routers under /v1/autodj
├── app_ui.py             # Streamlit UI — Alalu DJ Console
├── requirements.txt      # Python dependencies
│
├── api/                  # FastAPI route modules
│   ├── engine_routes.py          # Core transition render endpoint
│   ├── library_routes.py         # Library scan & track listing
│   ├── queue_routes.py           # Queue operations
│   ├── queue_state_routes.py     # Queue state persistence
│   ├── queue_smart_render_routes.py  # Smart render for current → next
│   ├── smart_routes.py           # Smart transition planning
│   ├── setlist_routes.py         # Setlist read/write/clear
│   ├── playback_routes.py        # Final mix assembly
│   └── smart_set_routes.py       # Full set build & render
│
├── core/                 # Business logic
│   └── beat_matcher.py   # BPM analysis, beat sync, crossfade rendering
│
├── models/               # Pydantic request/response models
│
├── storage/              # Persisted JSON state files
│   ├── library_metadata.json
│   ├── queue_state.json
│   └── setlist_state.json
│
├── templates/            # Jinja2 HTML templates (legacy/alternate UI)
│
└── outputs/              # Rendered transition WAV clips
```

---

## Prerequisites

### Python 3.8+

```bash
python --version
```

### FFmpeg

Required for audio encoding, slicing, and export.

**macOS**
```bash
brew install ffmpeg
```

**Linux**
```bash
sudo apt install ffmpeg
```

**Windows**
```bash
choco install ffmpeg
```
Or download from [ffmpeg.org](https://ffmpeg.org/download.html).

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Prahas10/audio-engine.git
cd audio-engine
```

**2. Create a virtual environment (recommended)**

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `librosa`, `numpy`, `pydub`, `soundfile`, `jinja2`, `python-multipart`

**4. Create output directories**

```bash
mkdir -p outputs storage
```

**5. Start the API server**

```bash
uvicorn main:app --reload
```

Server runs at `http://127.0.0.1:8000`.

**6. Start the Streamlit UI (optional, separate terminal)**

```bash
streamlit run app_ui.py
```

---

## API Reference

All routes are prefixed with `/v1/autodj`.

### Library

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/library/scan-folder` | Scan a folder and extract metadata for all tracks |
| `GET`  | `/library/tracks` | List all tracks in the library |

**Scan folder payload**
```json
{
  "folder_path": "/path/to/tracks",
  "library_path": "storage/library_metadata.json",
  "force_rescan": false,
  "clear_existing": false
}
```

### Queue State

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/queue-state/create` | Initialize or reset the queue |
| `GET`  | `/queue-state/state` | Get current queue contents |
| `POST` | `/queue-state/add-track` | Add a track to the queue |
| `POST` | `/queue-state/advance` | Advance the queue to the next track |

### Transition Rendering

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/queue/smart-render` | Render a transition from the current queue position to the next |
| `POST` | `/engine/render-transition` | Render a direct transition between two specified tracks |

**Smart render payload**
```json
{
  "queue_path": "storage/queue_state.json",
  "library_path": "storage/library_metadata.json",
  "preferred_mix_duration": 30,
  "output_dir": "outputs",
  "auto_advance": false
}
```

**Direct render payload**
```json
{
  "track_a_url": "tracks/song_a.wav",
  "track_b_url": "tracks/song_b.wav",
  "transition_start_time": 60.0,
  "mix_duration": 30
}
```

**Response**
```json
{
  "audio_clip_url": "outputs/rendered_mix.wav"
}
```

### Setlist

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/setlist/state` | Get the full setlist timeline |
| `POST` | `/setlist/clear` | Clear the setlist |

### Playback Assembly

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/playback/assemble` | Stitch all transition clips into a final WAV |

**Payload**
```json
{
  "setlist_path": "storage/setlist_state.json",
  "library_path": "storage/library_metadata.json",
  "output_path": "outputs/final_mix.wav"
}
```

### Smart Set Builder

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/set/build-and-render` | Auto-sequence and render a full optimized DJ set |
| `POST` | `/set/render-queue-order` | Render the full set in the current queue order |

**Build-and-render payload**
```json
{
  "library_path": "storage/library_metadata.json",
  "setlist_path": "storage/setlist_state.json",
  "starting_track_id": null,
  "preferred_mix_duration": 30,
  "output_dir": "outputs",
  "final_output_path": "outputs/final_set.wav",
  "render": true
}
```

---

## Alalu DJ Console (Streamlit UI)

Run `streamlit run app_ui.py` and open the displayed URL in your browser. The console has six tabs:

| Tab | What it does |
|-----|-------------|
| **Library** | Scan a folder of tracks and browse the library with BPM, key, and Camelot info |
| **Queue** | Build and manage your playback queue, add tracks, and advance position |
| **Smart Render** | Render the next transition from the current queue position, preview audio in-browser |
| **Setlist** | View the full transition timeline, play back individual clips |
| **Playback** | Assemble all clips into a final continuous mix WAV |
| **Smart Stream** | Build an entire DJ set — either auto-optimized or in manual queue order — and preview the full set with a timeline view |

Configure library, queue, and output paths in the sidebar.

---

## Typical Workflow

1. **Scan** your tracks folder via the Library tab → metadata extracted and saved.
2. **Build a queue** by adding tracks in the order you want (or let Smart Stream auto-sequence).
3. **Smart Render** each transition one by one, previewing the output audio each time.
4. **Assemble** the final set in the Playback tab to get a single continuous WAV.

Or use **Smart Stream → Build Smart Optimized Set** to do steps 2–4 in one click.

---

## Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/v1/autodj/engine/render-transition" \
  -H "Content-Type: application/json" \
  -d '{
    "track_a_url": "tracks/song_a.wav",
    "track_b_url": "tracks/song_b.wav",
    "transition_start_time": 45.0,
    "mix_duration": 30
  }'
```

---

## Tech Stack

| Technology | Role |
|------------|------|
| Python 3.8+ | Core language |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Librosa | BPM detection & beat analysis |
| Pydub | Audio slicing & crossfade |
| SoundFile | WAV read/write |
| FFmpeg | Audio encoding & export |
| Streamlit | DJ Console UI |
| Jinja2 | HTML templating |

---

## Roadmap

- AI-based harmonic key matching
- Automatic EQ and frequency balancing at transition points
- Streaming audio output support
- Multi-track DJ queue with lookahead planning
- Spotify / SoundCloud integration for track sourcing
- GPU-accelerated audio analysis

---

## License

This project is unlicensed. Contact the author for usage permissions.