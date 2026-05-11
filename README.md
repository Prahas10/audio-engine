# Audio Mixing Engine 🐻🎧

## Overview
Audio Mixing Engine is an automated audio mixing tool designed to act like a digital DJ. It takes two separate audio tracks and seamlessly blends them together to create a smooth transition from one song to the next.

Instead of simply fading one song out and another in, the engine actually “listens” to the music. It analyzes the rhythm, tempo, and beats of both tracks to find the mathematically optimal moment to synchronize them. This ensures that when the transition occurs, the drum beats align naturally, preventing the messy and out-of-sync sound that happens when two tracks clash rhythmically.

Whether you are building a custom DJ bot, a workout mix generator, or experimenting with intelligent audio systems, Audio Mixing Engine handles the heavy lifting of beat matching for you.

---

# Core Features

## Smart Beat Matching
Automatically analyzes the beats and tempo of two audio files to determine the best synchronization point for a seamless transition.

## Automated Crossfading
Applies smooth volume transitions by gradually fading out Track A while fading in Track B over a configurable timeline.

## Customizable Transitions
Control exactly:
- When the transition begins
- How long the transition lasts

Supported transition durations range from **1 to 120 seconds**.

## Interactive Web UI
Includes a browser-based dashboard where you can:
- Paste transition JSON payloads
- Generate transitions
- Instantly preview the mixed output audio

## Developer-Ready API
Built using FastAPI, allowing developers to:
- Send JSON payloads
- Trigger transition rendering
- Receive generated audio URLs programmatically

---

# API Contract

## Endpoint
```http
POST /v1/engine/render-transition
```

## Request Payload
```json
{
  "track_a_url": "track_a.wav",
  "track_b_url": "track_b.wav",
  "transition_start_time": 60.0,
  "mix_duration": 30
}
```

## Payload Fields

| Field | Type | Description |
|---|---|---|
| `track_a_url` | String | Path to the currently playing high-resolution audio |
| `track_b_url` | String | Path to the incoming audio track |
| `transition_start_time` | Float | Time (in seconds) inside Track A where the transition begins |
| `mix_duration` | Integer | Length of the transition in seconds (default: 30s) |

## Response
```json
{
  "audio_clip_url": "/outputs/rendered_mix.wav"
}
```

---

# Installation and Prerequisites

## Prerequisites

Before starting, ensure the following are installed on your system.

### 1. Python 3.8+
The audio engine is built using Python.

Verify installation:
```bash
python --version
```

---

### 2. FFmpeg
FFmpeg is required for audio processing operations such as:
- Cutting audio
- Fading tracks
- Exporting WAV/MP3 files

### Install FFmpeg

#### macOS
```bash
brew install ffmpeg
```

#### Linux
```bash
sudo apt install ffmpeg
```

#### Windows
Download from the official FFmpeg website or install via Chocolatey:

```bash
choco install ffmpeg
```

---

# Installation Steps

## 1. Clone or Download the Project

Ensure your project structure looks like this:

```text
project/
│
├── main.py
├── requirements.txt
├── templates/
│   └── index.html
└── outputs/
```

---

## 2. Create a Virtual Environment (Recommended)

Using a virtual environment keeps dependencies isolated.

### macOS/Linux
```bash
python -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Typical dependencies include:
- FastAPI
- Uvicorn
- Librosa
- NumPy
- SoundFile
- Pydub
- Jinja2

---

## 4. Prepare Audio Files

Place your audio files inside the project directory or provide absolute paths.

Example:
```text
track_a.wav
track_b.wav
```

You can reference them directly inside the JSON payload.

---

## 5. Start the FastAPI Server

Run the application using Uvicorn:

```bash
uvicorn main:app --reload
```

If successful, you should see:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## 6. Access the Web App

Open your browser and navigate to:

```text
http://127.0.0.1:8000
```

You will see the Audio Mixing Engine dashboard where you can:
- Upload transition payloads
- Generate transitions
- Listen to rendered mixes directly in the browser

---

# Example Request Using cURL

```bash
curl -X POST "http://127.0.0.1:8000/v1/engine/render-transition" \
-H "Content-Type: application/json" \
-d '{
  "track_a_url": "track_a.wav",
  "track_b_url": "track_b.wav",
  "transition_start_time": 45.0,
  "mix_duration": 30
}'
```

---

# Example Workflow

1. User uploads or references two tracks
2. Audio Mixing Engine analyzes BPM and beat positions
3. The engine determines the best synchronization point
4. Crossfading and beat alignment are applied
5. A rendered transition clip is exported
6. The generated audio URL is returned to the client

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core backend logic |
| FastAPI | REST API framework |
| Librosa | Beat and tempo analysis |
| Pydub | Audio manipulation |
| FFmpeg | Audio encoding/export |
| Uvicorn | ASGI server |
| Jinja2 | HTML templating |

---

# Future Improvements

Potential enhancements for MixingBear include:
- AI-based harmonic key matching
- Automatic EQ balancing
- Streaming audio support
- Multi-track DJ queue support
- Spotify/SoundCloud integration
- GPU-accelerated audio analysis

---

# License

This project is intended for educational and experimental purposes.

---

# Author

Built with ❤️ using Python, FastAPI, and intelligent audio processing.