# streamlit_app.py
import streamlit as st
import librosa
from pydub import AudioSegment, effects
from pathlib import Path
import tempfile, shutil, json, os
import numpy as np

st.set_page_config(page_title="Traktor RemixDeck Builder — MVP", layout="wide")
st.markdown("<h1 style='color:#00bfff;'>Traktor RemixDeck Builder — MVP</h1>", unsafe_allow_html=True)
st.write("This MVP performs: upload → BPM detection → kick-aligned (approx) loop slicing → MP3 320 export → ZIP download.")

st.sidebar.header("Loop & Export Settings")
bars = st.sidebar.slider("Loop bars", 1, 64, 8)
fade_ms = st.sidebar.slider("Fade (ms)", 0, 500, 40)
overlap = st.sidebar.slider("Overlap (%)", 0, 50, 0)
include_oneshots = st.sidebar.checkbox("Detect one-shots (simple)", value=True)
oneshot_threshold = st.sidebar.slider("One-shot dBFS threshold", -60, -10, -35)

uploaded = st.file_uploader("Upload MP3 or WAV (single file)", type=["mp3","wav"])

def detect_bpm_and_kicks(path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # simple kick-ish detection via onset envelope (approx)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.02, wait=3)
    kick_times = librosa.frames_to_time(peaks, sr=sr)
    return float(tempo), np.array(kick_times), y, sr

def slice_loops(audio_segment, kick_times, bars, bpm, fade_ms, overlap_pct):
    beat = 60.0/float(bpm)
    loop_dur = bars * 4 * beat
    step = loop_dur * (1 - overlap_pct/100.0)
    loops = []
    t = 0.0
    while t + 0.5 < audio_segment.duration_seconds:
        # snap to first kick after t if available
        future = kick_times[kick_times >= t] if len(kick_times)>0 else np.array([])
        t0 = float(future[0]) if len(future)>0 else t
        t1 = t0 + loop_dur
        if t1 > audio_segment.duration_seconds:
            break
        seg = audio_segment[int(t0*1000):int(t1*1000)]
        if fade_ms>0:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)
        loops.append({"start": t0, "end": t1, "audio": seg})
        t += step
    return loops

def detect_one_shots(path, threshold_db=-35, max_ms=600):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=False)
    audio = AudioSegment.from_file(str(path))
    shots = []
    for t in onset_times:
        start_ms = max(0, int(t*1000 - 20))
        clip = audio[start_ms:start_ms+max_ms]
        try:
            if clip.dBFS >= threshold_db:
                shots.append({"time": t, "audio": clip})
        except Exception:
            continue
    return shots

def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True)

def export_deck(out_dir:Path, loops, one_shots, bpm, metadata_extra):
    ensure_dir(out_dir)
    metadata = {"bpm": bpm, "loops": [], "one_shots": [], "extra": metadata_extra}
    # export loops
    for i, L in enumerate(loops, start=1):
        fn = out_dir / f"loop_{i:02d}.mp3"
        L["audio"].export(fn, format="mp3", bitrate="320k")
        metadata["loops"].append({"file": str(fn.name), "start": L["start"], "end": L["end"]})
    for i, S in enumerate(one_shots, start=1):
        fn = out_dir / f"shot_{i:02d}.mp3"
        S["audio"].export(fn, format="mp3", bitrate="320k")
        metadata["one_shots"].append({"file": str(fn.name), "time": S["time"]})
    with open(out_dir / "metadata.json","w") as f:
        json.dump(metadata, f, indent=2)
    return metadata

if uploaded:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        track_path = tmp / uploaded.name
        track_path.write_bytes(uploaded.getbuffer())

        st.info(f"Loaded: {uploaded.name} — analyzing...")
        try:
            tempo, kick_times, y, sr = detect_bpm_and_kicks(track_path)
        except Exception as e:
            st.error(f"Error analyzing audio: {e}")
            st.stop()
        st.success(f"Detected approximate BPM: {tempo:.2f}")

        # create pydub segment
        audio_seg = AudioSegment.from_file(str(track_path))
        with st.spinner("Slicing loops..."):
            loops = slice_loops(audio_seg, kick_times, bars=bars, bpm=tempo, fade_ms=fade_ms, overlap_pct=overlap)
        st.write(f"Generated {len(loops)} loops. Preview below.")

        if include_oneshots:
            with st.spinner("Detecting one-shots..."):
                shots = detect_one_shots(track_path, threshold_db=oneshot_threshold)
            st.write(f"Detected {len(shots)} one-shots (approx).")
        else:
            shots = []

        # Preview grid
        st.subheader("Loop previews")
        cols = st.columns(4)
        for i, L in enumerate(loops):
            fn = tmp / f"preview_loop_{i+1}.mp3"
            L["audio"].export(fn, format="mp3", bitrate="320k")
            with cols[i%4]:
                st.audio(str(fn))
                st.caption(f"Loop {i+1} — {L['start']:.2f}s to {L['end']:.2f}s")

        if shots:
            st.subheader("One-shot previews")
            cols2 = st.columns(6)
            for i, S in enumerate(shots):
                fn = tmp / f"preview_shot_{i+1}.mp3"
                S["audio"].export(fn, format="mp3", bitrate="320k")
                with cols2[i%6]:
                    st.audio(str(fn))
                    st.caption(f"Shot {i+1} — {S['time']:.2f}s")

        # Export deck
        st.markdown("---")
        deck_name = st.text_input("Deck name for export", value=uploaded.name.split(".")[0] + "_RemixDeck")
        export_btn = st.button("Export Remix Deck ZIP (MP3 320)")
        if export_btn:
            out_dir = tmp / deck_name
            out_dir.mkdir(exist_ok=True)
            meta = {"bpm": tempo, "loops": [], "one_shots": []}
            for i, L in enumerate(loops, start=1):
                fn = out_dir / f"loop_{i:02d}.mp3"
                L["audio"].export(fn, format="mp3", bitrate="320k")
                meta["loops"].append({"file": fn.name, "start": L["start"], "end": L["end"]})
            for i, S in enumerate(shots, start=1):
                fn = out_dir / f"shot_{i:02d}.mp3"
                S["audio"].export(fn, format="mp3", bitrate="320k")
                meta["one_shots"].append({"file": fn.name, "time": S["time"]})
            # write metadata
            with open(out_dir / "metadata.json","w") as f:
                json.dump(meta, f, indent=2)
            # zip
            zip_path = tmp / (deck_name + ".zip")
            shutil.make_archive(str(zip_path).replace('.zip',''), 'zip', out_dir)
            with open(str(zip_path), "rb") as fh:
                st.download_button("Download Remix Deck ZIP", fh.read(), file_name=deck_name + ".zip")
            st.success("Export ready. Download the ZIP and copy into Traktor or a USB drive.")
