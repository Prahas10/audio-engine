import os
import requests
import streamlit as st


API_BASE = "http://127.0.0.1:8000"

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
    "Rendered clips folder",
    value="rendered_clips"
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


tab_library, tab_queue, tab_render, tab_setlist, tab_playback = st.tabs([
    "Library",
    "Queue",
    "Smart Render",
    "Setlist",
    "Playback"
])


with tab_library:
    st.header("Library Scanner")

    folder_path = st.text_input(
        "Folder containing tracks",
        value="E:/Project/MixingBear"
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