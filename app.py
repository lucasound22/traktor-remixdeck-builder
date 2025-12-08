import streamlit as st
import numpy as np
from pydub import AudioSegment
import librosa
import soundfile as sf
import zipfile
import io
import os
import tempfile

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Traktor RemixDeck Builder",
    page_icon="🎧",
    layout="wide"
)

# -----------------------------------------------------------
# CUSTOM TRAKTOR UI THEME
# -----------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0E0E0E !important;
}

h1, h2, h3, h4 {
    color: #0A84FF !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

.stFileUploader, .stButton > button {
    background-color: #1A1A1A !important;
    border: 1px solid #0A84FF !important;
    color: white !important;
    padding: 8px 14px;
    border-radius: 6px;
}

.stSlider > div > div {
    background: #0A84FF !important;
}

.stAlert {
    background-color: #000 !important;
    border-left: 3px solid #0A84FF !important;
}

.block-container {
    padding-top: 2rem !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# TITLE
# -----------------------------------------------------------
st.title("Traktor RemixDeck Builder by Tuesdaynightfreak Productions")
st.write("Upload any MP3/WAV → BPM detect → auto slice → MP3 export → ZIP download.")


# -----------------------------------------------------------
# AUDIO LOADER (FFMPEG SAFE)
# -----------------------------------------------------------
def load_audio(file):
    """Load audio using pydub with ffmpeg support."""
    suffix = file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    audio = AudioSegment.from_file(tmp_path)
    os.remove(tmp_path)
    return audio


# -----------------------------------------------------------
# BPM DETECTION
# -----------------------------------------------------------
def detect_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo


# -----------------------------------------------------------
# AUTO SLICING
# -----------------------------------------------------------
def slice_audio(audio, slice_ms):
    chunks = []
    for start in range(0, len(audio), slice_ms):
        end = min(start + slice_ms, len(audio))
        chunks.append(audio[start:end])
    return chunks


# -----------------------------------------------------------
# EXPORT CHUNKS INTO ZIP
# -----------------------------------------------------------
def export_zip(chunks, bpm):
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for i, chunk in enumerate(chunks):
            temp_path = f"slice_{i+1}.mp3"
            chunk.export(temp_path, format="mp3", bitrate="320k")
            z.write(temp_path)
            os.remove(temp_path)

        # Add metadata
        z.writestr("info.txt", f"Generated BPM: {bpm}")

    buffer.seek(0)
    return buffer


# -----------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Drag & drop your MP3 or WAV here", type=["mp3", "wav"])

if uploaded_file:
    st.info(f"Loaded: **{uploaded_file.name}** — analyzing...")

    # Load audio
    audio = load_audio(uploaded_file)

    # Convert to numpy for analysis
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max
    sr = audio.frame_rate

    # BPM
    bpm = detect_bpm(samples, sr)
    st.success(f"Detected approximate BPM: **{bpm:.2f}**")

    # Slice size selector
    col1, col2 = st.columns(2)
    with col1:
        slice_bars = st.slider("Loop bars", 1, 16, 4)
    with col2:
        st.write(" ")

    slice_ms = int((60_000 / bpm) * slice_bars)

    st.write(f"Each slice = **{slice_bars} bars** ({slice_ms} ms)")

    # Process
    if st.button("Build Remix Deck ZIP"):
        st.info("Processing slices...")

        chunks = slice_audio(audio, slice_ms)
        bundle = export_zip(chunks, bpm)

        st.success("Your remix deck ZIP is ready!")
        st.download_button(
            label="⬇ Download Remix Deck ZIP",
            data=bundle,
            file_name="traktor-remixdeck.zip",
            mime="application/zip"
        )

