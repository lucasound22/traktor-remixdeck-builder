# app.py
"""
Traktor RemixDeck Builder by Tuesdaynightfreak Productions
Theme: Traktor-like dark (club purple / techno)
Logo: Circular DJ Wheel (SVG)
Features:
 - Robust BPM detection (safe fallbacks)
 - Kick-aligned loop slicing (approx)
 - One-shot detection (onset-based)
 - MP3 export (320k) + ZIP download
 - Polished Traktor-inspired UI (Inter font, purple theme)
NOTE: pydub requires ffmpeg/ffprobe. On Streamlit Cloud add 'ffmpeg' to packages.txt.
"""

import streamlit as st
import numpy as np
import librosa
from pydub import AudioSegment
import tempfile, shutil, json, io, os, zipfile
from pathlib import Path

# ---------------------------
# Page config + theme CSS
# ---------------------------
st.set_page_config(page_title="Traktor RemixDeck Builder", page_icon="🎧", layout="wide")

# Inline SVG logo (Circular DJ Wheel - Logo A)
SVG_LOGO = r'''
<svg width="148" height="148" viewBox="0 0 148 148" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Traktor RemixDeck Builder logo">
  <defs>
    <radialGradient id="g" cx="50%" cy="40%">
      <stop offset="0%" stop-color="#5be0ff"/>
      <stop offset="65%" stop-color="#0a84ff"/>
      <stop offset="100%" stop-color="#4a12a8"/>
    </radialGradient>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#000" flood-opacity="0.45"/>
    </filter>
  </defs>
  <g filter="url(#s)">
    <circle cx="74" cy="74" r="66" fill="url(#g)"/>
    <circle cx="74" cy="74" r="46" fill="#0e0e0e"/>
    <!-- center platter -->
    <circle cx="74" cy="74" r="18" fill="#111111" stroke="#2a2a2a" stroke-width="2"/>
    <!-- groove rings -->
    <g stroke="#151515" stroke-opacity="0.4" stroke-width="1">
      <circle cx="74" cy="74" r="32" fill="none"/>
      <circle cx="74" cy="74" r="38" fill="none"/>
    </g>
    <!-- stylized notch markers -->
    <g transform="rotate(0 74 74)" fill="#a8d8ff" opacity="0.95">
      <rect x="72" y="8" width="4" height="12" rx="2"/>
      <rect x="72" y="128" width="4" height="12" rx="2"/>
    </g>
    <g transform="rotate(45 74 74)" fill="#8ad1ff" opacity="0.85">
      <rect x="72" y="8" width="4" height="10" rx="2"/>
    </g>
    <g transform="rotate(90 74 74)" fill="#6fc1ff" opacity="0.75">
      <rect x="72" y="8" width="4" height="8" rx="2"/>
    </g>
    <!-- play indicator -->
    <polygon points="86,74 66,84 66,64" fill="#00f0ff" opacity="0.9"/>
  </g>
</svg>
'''

# Dark purple / techno CSS and layout
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(180deg, #0b0711 0%, #0f0820 45%, #15072b 100%) fixed !important;
        color: #e6eef8;
    }}

    /* Header area */
    .header {{
        display:flex;
        align-items:center;
        gap:18px;
        padding: 16px 8px;
    }}
    .app-title {{
        font-size:36px;
        font-weight:700;
        color: #34c0ff;
        margin:0;
        letter-spacing:-0.6px;
    }}
    .app-sub {{
        color:#cfefff;
        opacity:0.9;
        margin-top:4px;
        font-weight:400;
    }}

    /* uploader card */
    .uploader {{
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(67,120,255,0.08);
        padding:14px;
        border-radius:10px;
    }}

    /* Button styles */
    .stButton > button {
        background: linear-gradient(90deg,#0a84ff,#2aa6ff) !important;
        color: #041827 !important;
        font-weight:600;
        border: none !important;
        box-shadow: 0 6px 18px rgba(10,132,255,0.12);
        padding:10px 16px;
        border-radius:8px;
    }

    /* small output boxes */
    .stAlert {
        border-left: 4px solid #34c0ff !important;
        background: rgba(12,16,24,0.6) !important;
        color: #d8f3ff !important;
    }

    /* footer */
    .footer {
        margin-top:36px;
        padding:28px;
        border-radius:12px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.03);
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
    }
    .footer .left { color:#bfe9ff; }
    .footer .right { opacity:0.85; color:#d9f1ff; }

    /* responsive preview grid */
    .preview-grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(240px,1fr)); gap:12px; }

    /* small captions */
    .caption { color:#9fcff6; font-size:13px; opacity:0.95; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Header / Logo
# ---------------------------
st.markdown(
    f"""
    <div class="header">
      <div style="width:82px;height:82px">{SVG_LOGO}</div>
      <div>
        <div class="app-title">Traktor RemixDeck Builder</div>
        <div class="app-sub">by Tuesdaynightfreak Productions — Upload • Analyze • Slice • Export</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")  # spacer

# ---------------------------
# Controls (left column)
# ---------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("<div class='uploader'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag & drop your MP3 or WAV here", type=["mp3", "wav"])
    st.caption("Limit: 200MB per file • MP3, WAV")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Slice Settings")
    bars = st.slider("Loop bars", min_value=1, max_value=32, value=4, help="Number of 4/4 bars per loop (4 bars = 1 phrase commonly).")
    fade_ms = st.slider("Fade (ms)", min_value=0, max_value=500, value=40)
    overlap = st.slider("Overlap (%)", min_value=0, max_value=50, value=0)
    detect_shots = st.checkbox("Detect one-shots (onset-based)", value=True)
    shot_thresh = st.slider("One-shot loudness threshold (dBFS)", -60, -10, value=-35)

    st.markdown("---")
    st.markdown("**Export**")
    export_name = st.text_input("Deck name", value="MyRemixDeck")
    out_format = st.selectbox("Export format", ["mp3 (320k)"], index=0)  # future: wav option
    st.markdown("---")
    st.caption("Tip: Add 'ffmpeg' to packages.txt if deploying to Streamlit Cloud (pydub needs it).")

# ---------------------------
# Helper functions
# ---------------------------
def robust_detect_bpm(y, sr):
    """
    Try multiple ways to detect BPM and return a float.
    Falls back to 128 BPM if detection fails.
    """
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if tempo is None or np.isnan(tempo):
            raise ValueError("beat_track returned invalid tempo")
        tempo = float(tempo)
        # common half/double corrections heuristic:
        if tempo < 65:
            tempo *= 2.0
        if tempo > 220:
            tempo /= 2.0
        return tempo
    except Exception:
        # try onset autocorrelation fallback
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            ac = np.correlate(onset_env, onset_env, mode='full')
            ac = ac[ac.size // 2:]
            peak = np.argmax(ac[1:]) + 1
            if peak > 0:
                bpm_est = 60.0 / (peak / float(sr))
                if 40 <= bpm_est <= 220:
                    return float(bpm_est)
        except Exception:
            pass
    # ultimate fallback
    return 128.0

def detect_kicks(y, sr):
    """
    Simple kick / onset detection using low-frequency onset strength.
    Returns array of times (seconds).
    """
    try:
        # compute onset strength using spectral band emphasizing low freq
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.02, wait=3)
        times = librosa.frames_to_time(peaks, sr=sr)
        return np.array(times)
    except Exception:
        return np.array([])

def snap_to_next_kick(t, kicks):
    if kicks.size == 0:
        return t
    future = kicks[kicks >= t]
    if future.size > 0:
        return float(future[0])
    return float(kicks[-1])  # last kick

def slice_kick_aligned(audio_segment: AudioSegment, y, sr, bars, bpm, fade_ms, overlap_pct):
    """
    Slice the audio into loops that start on the nearest kick after the intended grid position.
    Returns list of dicts: {'start': sec, 'end': sec, 'audio': AudioSegment}
    """
    beat_dur = 60.0 / float(bpm)
    loop_dur_sec = bars * 4 * beat_dur
    step = loop_dur_sec * (1 - overlap_pct / 100.0)
    kicks = detect_kicks(y, sr)
    loops = []
    t = 0.0
    # guard: create minimal slices even if kick detection bad
    while t + 0.5 < audio_segment.duration_seconds:
        t0 = snap_to_next_kick(t, kicks)
        t1 = t0 + loop_dur_sec
        if t1 > audio_segment.duration_seconds:
            break
        seg = audio_segment[int(t0 * 1000): int(t1 * 1000)]
        if fade_ms > 0:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)
        loops.append({"start": t0, "end": t1, "audio": seg})
        t += step
    return loops

def detect_one_shots(path, threshold_db=-35, max_ms=600):
    """
    Onset-based one-shot detection on the full mix;
    returns list of dicts: {'time': sec, 'audio': AudioSegment}
    """
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=False)
        audio = AudioSegment.from_file(str(path))
        shots = []
        for t in onset_times:
            start_ms = max(0, int(t * 1000 - 10))
            clip = audio[start_ms:start_ms + max_ms]
            try:
                if clip.dBFS >= threshold_db:
                    shots.append({"time": float(t), "audio": clip})
            except Exception:
                # silent clip or other invalid; skip
                continue
        return shots
    except Exception:
        return []

def export_deck_zip(loops, shots, bpm, deck_name="RemixDeck"):
    """
    Export loops and one-shots into a zip file (in-memory) with metadata.json
    """
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # loops
        metadata = {"bpm": float(bpm), "loops": [], "one_shots": []}
        for i, L in enumerate(loops, start=1):
            fname = f"loops/loop_{i:02d}.mp3"
            # export to temporary file because pydub export requires filename
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmpf.close()
            L["audio"].export(tmpf.name, format="mp3", bitrate="320k")
            z.write(tmpf.name, arcname=fname)
            os.unlink(tmpf.name)
            metadata["loops"].append({"file": fname, "start": L["start"], "end": L["end"]})
        # one-shots
        for i, S in enumerate(shots, start=1):
            fname = f"oneshots/shot_{i:02d}.mp3"
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmpf.close()
            S["audio"].export(tmpf.name, format="mp3", bitrate="320k")
            z.write(tmpf.name, arcname=fname)
            os.unlink(tmpf.name)
            metadata["one_shots"].append({"file": fname, "time": S["time"]})
        # metadata.json
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
    memfile.seek(0)
    return memfile

# ---------------------------
# Main processing
# ---------------------------
with right_col:
    st.markdown("<div class='uploader'>", unsafe_allow_html=True)
    st.write(" ")
    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded_file:
        st.info("Upload an MP3 or WAV file using the panel on the left to begin.")
        st.stop()

    # Save uploaded file to a temp path (pydub + librosa friendly)
    tmpdir = tempfile.mkdtemp(prefix="trk_")
    try:
        ext = Path(uploaded_file.name).suffix.lower()
        safe_path = Path(tmpdir) / ("upload" + ext)
        with open(safe_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"Loaded: **{uploaded_file.name}** — analyzing...")

        # librosa analysis for BPM and kicks
        try:
            y, sr = librosa.load(str(safe_path), sr=None, mono=True)
        except Exception as e:
            st.error(f"Error reading audio for analysis: {e}")
            # attempt pydub->librosa fallback
            try:
                audio_pd = AudioSegment.from_file(str(safe_path))
                samples = np.array(audio_pd.get_array_of_samples()).astype(np.float32)
                sr = audio_pd.frame_rate
                samples = samples / np.iinfo(audio_pd.array_type).max
                y = samples
            except Exception as e2:
                st.error(f"Fatal audio load error: {e2}")
                raise

        bpm = robust_detect_bpm(y, sr)
        st.success(f"Detected approximate BPM: **{bpm:.2f}**")

        # pydub segment for slicing & export
        audio_segment = AudioSegment.from_file(str(safe_path))

        # Slicing
        with st.spinner("Generating kick-aligned loops..."):
            loops = slice_kick_aligned(audio_segment, y, sr, bars=bars, bpm=bpm, fade_ms=fade_ms, overlap_pct=overlap)
        st.write(f"Generated **{len(loops)}** loops.")

        # One-shots
        if detect_shots:
            with st.spinner("Detecting one-shots..."):
                shots = detect_one_shots(safe_path, threshold_db=shot_thresh)
            st.write(f"Detected **{len(shots)}** one-shots.")
        else:
            shots = []

        # Preview grid
        st.markdown("### Loop preview")
        if len(loops) == 0:
            st.warning("No loops generated — try reducing loop bars or uploading a longer track.")
        else:
            grid_html = "<div class='preview-grid'>"
            for i, L in enumerate(loops):
                # export small preview mp3 to temp file for playback
                tmpf = Path(tmpdir) / f"preview_loop_{i+1:02d}.mp3"
                L["audio"].export(str(tmpf), format="mp3", bitrate="192k")
                grid_html += f"<div><audio controls src='file://{str(tmpf)}'></audio><div class='caption'>Loop {i+1} — {L['start']:.2f}s → {L['end']:.2f}s</div></div>"
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)

        if shots:
            st.markdown("### One-shot preview")
            grid_html = "<div class='preview-grid'>"
            for i, S in enumerate(shots):
                tmpf = Path(tmpdir) / f"preview_shot_{i+1:02d}.mp3"
                S["audio"].export(str(tmpf), format="mp3", bitrate="192k")
                grid_html += f"<div><audio controls src='file://{str(tmpf)}'></audio><div class='caption'>Shot {i+1} — {S['time']:.2f}s</div></div>"
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)

        # Export
        st.markdown("---")
        if st.button("Build Remix Deck ZIP"):
            st.info("Exporting deck — packaging MP3s and metadata...")
            deck_zip = export_deck_zip(loops, shots, bpm, deck_name=export_name)
            st.success("Export ready — download below.")
            st.download_button("⬇ Download Remix Deck ZIP", data=deck_zip.getvalue(), file_name=f"{export_name}.zip", mime="application/zip")

    finally:
        # cleanup tempdir after user interaction (do not delete when testing in dev!)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# ---------------------------
# Footer (visual club-style)
# ---------------------------
st.markdown(
    """
    <div class="footer">
      <div class="left">
        <strong>Tuesdaynightfreak Productions</strong><br>
        Traktor RemixDeck Builder — fast loops, one-shots & DJ-ready exports
      </div>
      <div class="right">
        <svg width="220" height="48" viewBox="0 0 220 48" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="fg" x1="0" x2="1">
              <stop offset="0" stop-color="#34c0ff"/>
              <stop offset="1" stop-color="#9b5aff"/>
            </linearGradient>
          </defs>
          <rect x="0" y="6" width="220" height="12" rx="6" fill="url(#fg)" opacity="0.18"/>
          <rect x="0" y="24" width="160" height="6" rx="3" fill="url(#fg)" opacity="0.12"/>
        </svg>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
