import os
import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"
BASE_PREFIX = "/v1/autodj"

st.set_page_config(page_title="Alalu DJ Console", layout="wide")

st.title("Alalu DJ Console 🐻🎧")

library_path = st.sidebar.text_input(
    "Library metadata path",
    value="storage/library_metadata.json"
)

queue_path = st.sidebar.text_input(
    "Queue state path",
    value="storage/queue_state.json"
)

setlist_path = st.sidebar.text_input(
    "Setlist state path",
    value="storage/setlist_state.json"
)

output_dir = st.sidebar.text_input(
    "Output folder",
    value="outputs"
)


# Calls backend API safely
def api_post(endpoint, payload):
    try:
        res = requests.post(f"{API_BASE}{endpoint}", json=payload)
        if res.status_code >= 400:
            st.error(res.text)
            return None
        return res.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def format_mmss(seconds):
    if seconds is None:
        return None

    seconds = float(seconds)
    minutes = int(seconds // 60)
    secs = int(round(seconds % 60))
    return f"{minutes}:{secs:02d}"

# Calls backend GET safely
def api_get(endpoint, params=None):
    try:
        res = requests.get(f"{API_BASE}{endpoint}", params=params)
        if res.status_code >= 400:
            st.error(res.text)
            return None
        return res.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


tab_library, tab_queue, tab_render, tab_setlist, tab_playback, tab_smart_stream = st.tabs([
    "Library",
    "Queue",
    "Smart Render",
    "Setlist",
    "Playback",
    "Smart Stream"
])


with tab_library:
    st.header("Library Scanner")

    folder_path = st.text_input(
        "Folder containing tracks",
        value="E:/Project/audio-engine/tracks"
    )
    force_rescan = st.checkbox("Force rescan existing tracks", value=False)
    clear_existing = st.checkbox("Clear library first, then rescan", value=False)
    if st.button("Scan Folder"):
        data = api_post("/v1/autodj/library/scan-folder", {
            "folder_path": folder_path,
            "library_path": library_path,
            "force_rescan": force_rescan,
            "clear_existing": clear_existing
        })

        if data:
            st.success(f"Scanned {data['scanned_count']} tracks")
            st.json(data)

    st.subheader("Library Tracks")

    if st.button("Refresh Library"):
        data = api_get("/v1/autodj/library/tracks", {
            "library_path": library_path
        })

        if data:
            st.session_state["library_data"] = data

    library_data = st.session_state.get("library_data")

    if library_data:
        tracks = library_data.get("tracks", [])

        for track in tracks:
            with st.expander(f"{track.get('filename')} | {track.get('bpm')} BPM | {track.get('camelot')}"):
                st.write("Track ID:", track.get("track_id"))
                st.write("Path:", track.get("path"))
                st.write("Key:", track.get("key"))
                st.write("Duration:", track.get("duration"))


with tab_queue:
    st.header("Queue Manager")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Create/Clear Queue"):
            data = api_post("/v1/autodj/queue-state/create", {
                "queue_path": queue_path
            })
            if data:
                st.success("Queue reset")
                st.json(data)

    with col2:
        if st.button("Refresh Queue"):
            data = api_get("/v1/autodj/queue-state/state", {
                "queue_path": queue_path,
                "library_path": library_path
            })
            if data:
                st.session_state["queue_data"] = data

    with col3:
        if st.button("Advance Queue"):
            data = api_post("/v1/autodj/queue-state/advance", {
                "queue_path": queue_path
            })
            if data:
                st.success("Queue advanced")
                st.json(data)

    library_data = api_get("/v1/autodj/library/tracks", {
        "library_path": library_path
    })

    if library_data:
        tracks = library_data.get("tracks", [])
        track_options = {
            f"{t.get('filename')} | {t.get('bpm')} BPM | {t.get('camelot')}": t.get("track_id")
            for t in tracks
        }

        selected_track_label = st.selectbox(
            "Select track to add",
            options=list(track_options.keys()) if track_options else []
        )

        if st.button("Add Track to Queue") and selected_track_label:
            selected_track_id = track_options[selected_track_label]

            data = api_post("/v1/autodj/queue-state/add-track", {
                "track_id": selected_track_id,
                "queue_path": queue_path,
                "library_path": library_path
            })

            if data:
                st.success("Track added to queue")
                st.json(data)

    queue_data = st.session_state.get("queue_data")

    if queue_data:
        st.subheader("Current Queue")
        st.json(queue_data)


with tab_render:
    st.header("Smart Render")

    preferred_mix_duration = st.number_input(
        "Preferred mix duration",
        min_value=5,
        max_value=120,
        value=30
    )

    auto_advance = st.checkbox("Auto advance queue after render", value=False)

    if st.button("Render Current → Next"):
        data = api_post("/v1/autodj/queue/smart-render", {
            "queue_path": queue_path,
            "library_path": library_path,
            "preferred_mix_duration": int(preferred_mix_duration),
            "output_dir": output_dir,
            "auto_advance": auto_advance
        })

        if data:
            st.success("Transition rendered")
            st.session_state["last_render"] = data
            st.json(data)

    last_render = st.session_state.get("last_render")

    if last_render:
        render_info = last_render.get("render", {})
        audio_path = render_info.get("audio_clip_url")

        if audio_path and os.path.exists(audio_path):
            st.subheader("Audio Preview")
            st.audio(audio_path)

        st.subheader("Plan")
        st.json(last_render.get("plan", {}))


with tab_setlist:
    st.header("Setlist Timeline")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Refresh Setlist"):
            data = api_get("/v1/autodj/setlist/state", {
                "setlist_path": setlist_path
            })

            if data:
                st.session_state["setlist_data"] = data

    with col2:
        if st.button("Clear Setlist"):
            data = api_post("/v1/autodj/setlist/clear", {
                "setlist_path": setlist_path
            })

            if data:
                st.success("Setlist cleared")
                st.json(data)

    setlist_data = st.session_state.get("setlist_data")

    if setlist_data:
        st.write("Transition count:", setlist_data.get("transition_count"))

        for idx, transition in enumerate(setlist_data.get("transitions", []), start=1):
            with st.expander(f"Transition {idx}: {transition.get('strategy')}"):
                st.json(transition)

                transition_file = transition.get("transition_file")
                if transition_file and os.path.exists(transition_file):
                    st.audio(transition_file)


with tab_playback:
    st.header("Playback Assembly")

    final_output_path = st.text_input(
        "Final playback output path",
        value="rendered_clips/final_playback_mix.wav"
    )

    if st.button("Assemble Final Playback Mix"):
        data = api_post("/v1/autodj/playback/assemble", {
            "setlist_path": setlist_path,
            "library_path": library_path,
            "output_path": final_output_path
        })

        if data:
            st.success("Final playback assembled")
            st.json(data)

            if os.path.exists(final_output_path):
                st.audio(final_output_path)

with tab_smart_stream:
    st.header("Smart Stream / Full Set Renderer")

    st.write("Build a complete DJ set from your library or from your custom queue order.")

    mode = st.radio(
        "Set mode",
        options=[
            "Smart optimized order",
            "Use current queue order"
        ]
    )

    duration_mode = st.radio(
        "Transition duration",
        ["Auto by strategy", "Manual override"]
    )

    preferred_mix_duration = None

    if duration_mode == "Manual override":
        preferred_mix_duration = st.number_input(
            "Manual mix duration",
            min_value=5,
            max_value=120,
            value=30
        )

    final_output_path = st.text_input(
        "Final set output file",
        value="outputs/final_set.wav"
    )

    starting_track_id = None

    if mode == "Smart optimized order":
        library_data = api_get(f"{BASE_PREFIX}/library/tracks", {
            "library_path": library_path
        })

        if library_data:
            tracks = library_data.get("tracks", [])

            start_options = {"Auto choose best starting track": None}

            for t in tracks:
                label = f"{t.get('filename')} | {t.get('bpm')} BPM | {t.get('camelot')}"
                start_options[label] = t.get("track_id")

            selected_start = st.selectbox(
                "Starting track",
                options=list(start_options.keys())
            )

            starting_track_id = start_options[selected_start]

        if st.button("Build Smart Optimized Set"):
            data = api_post(f"{BASE_PREFIX}/set/build-and-render", {
                "library_path": library_path,
                "setlist_path": setlist_path,
                "starting_track_id": starting_track_id,
                "preferred_mix_duration": preferred_mix_duration,
                "output_dir": output_dir,
                "final_output_path": final_output_path,
                "render": True
            })

            if data:
                st.success("Smart set rendered")
                st.session_state["smart_stream_result"] = data
                st.json(data)

    else:
        st.info("This uses the exact current queue order: current track → upcoming tracks.")

        if st.button("Render Current Queue Order"):
            data = api_post(f"{BASE_PREFIX}/set/render-queue-order", {
                "queue_path": queue_path,
                "library_path": library_path,
                "setlist_path": setlist_path,
                "preferred_mix_duration": preferred_mix_duration,
                "output_dir": output_dir,
                "final_output_path": final_output_path
            })

            if data:
                st.success("Queue-order set rendered")
                st.session_state["smart_stream_result"] = data
                st.json(data)

    result = st.session_state.get("smart_stream_result")

    if result:
        st.subheader("Set Timeline")

        timeline = result.get("timeline", [])
        transitions = result.get("transitions", [])

        st.write("DEBUG result keys:", list(result.keys()))
        st.write("DEBUG timeline length:", len(timeline))
        st.write("DEBUG transitions length:", len(transitions))

        if timeline:
            clean_timeline = []

            for entry in timeline:
                row_type = entry.get("type")

                if row_type == "transition":
                    from_track = entry.get("from_track", {})
                    to_track = entry.get("to_track", {})

                    clean_timeline.append({
                        "Type": "Transition",
                        "Start": entry.get("set_start"),
                        "End": entry.get("set_end"),
                        "Name": f"{from_track.get('filename')} → {to_track.get('filename')}",
                        "Strategy": entry.get("strategy"),
                        "Planner Strategy": entry.get("planner_strategy"),
                        "Mix Duration": entry.get("mix_duration"),
                        "Stretch Rate": entry.get("stretch_rate"),
                        "A Time": entry.get("track_a_original_transition_time"),
                        "B Entry": entry.get("track_b_original_entry_time"),
                        "Drift ms": entry.get("render", {}).get("beat_alignment_drift_ms"),
                    })

                else:
                    clean_timeline.append({
                        "Type": "Track",
                        "Start": entry.get("set_start"),
                        "End": entry.get("set_end"),
                        "Name": entry.get("filename"),
                        "Strategy": "-",
                        "Planner Strategy": "-",
                        "Mix Duration": "-",
                        "Stretch Rate": "-",
                        "A Time": entry.get("source_original_start"),
                        "B Entry": "-",
                        "Drift ms": "-",
                    })

            st.dataframe(clean_timeline, use_container_width=True)

        else:
            st.warning("No timeline data found in smart_stream_result.")
            st.json(result)

        output_path = (
            result.get("final_mix", {}).get("output_path")
            or result.get("output_path")
            or result.get("final_mix", {}).get("path")
        )

        if output_path and os.path.exists(output_path):
            st.audio(output_path)
            st.success(f"Full set ready: {output_path}")
        else:
            st.warning(f"Audio output not found: {output_path}")