# app.py
"""
Traktor RemixDeck Builder — with integrated TSI exporter
- Robust BPM detection
- Kick-aligned loop slicing & one-shot detection
- In-memory packaging of loops/one-shots into a .trak (zip) with:
    - loops/... mp3 files
    - oneshots/... mp3 files
    - metadata.json
    - pad_mapping.json
    - deck_name.tsi  (generated from pad_mapping)
- UI theme and preview remain the same
IMPORTANT: pydub requires ffmpeg in the environment (add 'ffmpeg' to packages.txt on Streamlit Cloud).
"""

import streamlit as st
import numpy as np
import librosa
from pydub import AudioSegment
import io, zipfile, json, tempfile, shutil, os
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Traktor RemixDeck Builder", page_icon="🎧", layout="wide")

# ----------------------------
# CSS + SVG Logo (safe string)
# ----------------------------
SVG_LOGO = r'''
<svg width="96" height="96" viewBox="0 0 148 148" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Logo">
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
    <circle cx="74" cy="74" r="18" fill="#111111" stroke="#2a2a2a" stroke-width="2"/>
    <g stroke="#151515" stroke-opacity="0.4" stroke-width="1">
      <circle cx="74" cy="74" r="32" fill="none"/>
      <circle cx="74" cy="74" r="38" fill="none"/>
    </g>
    <g transform="rotate(0 74 74)" fill="#a8d8ff" opacity="0.95">
      <rect x="72" y="8" width="4" height="12" rx="2"/>
      <rect x="72" y="128" width="4" height="12" rx="2"/>
    </g>
    <polygon points="86,74 66,84 66,64" fill="#00f0ff" opacity="0.9"/>
  </g>
</svg>
'''

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background: linear-gradient(180deg, #0b0711 0%, #0f0820 45%, #15072b 100%) fixed !important;
  color: #e6eef8;
}

/* Header */
.header { display:flex; align-items:center; gap:18px; padding:18px 6px; }
.app-title { font-size:32px; font-weight:700; color:#34c0ff; margin:0; letter-spacing:-0.6px; }
.app-sub { color:#cfefff; opacity:0.95; margin-top:4px; font-weight:400; font-size:14px; }

/* Uploader card */
.uploader { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid rgba(67,120,255,0.08); padding:12px; border-radius:10px; }

/* Buttons */
.stButton > button {
  background: linear-gradient(90deg,#0a84ff,#2aa6ff) !important;
  color:#041827 !important;
  font-weight:600;
  border-radius:8px;
  padding:8px 14px;
  border: none !important;
  box-shadow: 0 6px 18px rgba(10,132,255,0.12);
}

/* Alerts */
.stAlert { border-left: 4px solid #34c0ff !important; background: rgba(12,16,24,0.6) !important; color: #d8f3ff !important; }

/* Footer */
.footer { margin-top:28px; padding:18px; border-radius:10px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.03); display:flex; align-items:center; justify-content:space-between; gap:12px; color:#bfe9ff; }

.preview-grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(240px,1fr)); gap:12px; margin-top:12px; }
.caption { color:#9fcff6; font-size:13px; opacity:0.95; margin-top:6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Header content
# ----------------------------
st.markdown(
    f"""
    <div class="header">
      <div style="width:92px; height:92px">{SVG_LOGO}</div>
      <div>
        <div class="app-title">Traktor RemixDeck Builder</div>
        <div class="app-sub">by Tuesdaynightfreak Productions — Upload • Analyze • Slice • Export</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Controls layout
# ----------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown('<div class="uploader">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag & drop your MP3 or WAV here", type=["mp3", "wav"])
    st.caption("Limit: 200MB per file • MP3, WAV")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Slice Settings")
    bars = st.slider("Loop bars (4/4 bars)", min_value=1, max_value=32, value=4)
    fade_ms = st.slider("Fade (ms)", min_value=0, max_value=500, value=40)
    overlap = st.slider("Overlap (%)", min_value=0, max_value=50, value=0)
    detect_shots = st.checkbox("Detect one-shots (onset-based)", value=True)
    shot_thresh = st.slider("One-shot loudness threshold (dBFS)", -60, -10, value=-35)

    st.markdown("---")
    st.markdown("**Export settings**")
    deck_name = st.text_input("Deck name", value="MyRemixDeck")
    st.caption("Note: On Streamlit Cloud add 'ffmpeg' to packages.txt for pydub to work.")

# ----------------------------
# Helper functions (robust)
# ----------------------------
def robust_detect_bpm(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if tempo is None or np.isnan(tempo):
            raise ValueError("invalid tempo")
        tempo = float(tempo)
        # simple heuristics for half/double-time
        if tempo < 60:
            tempo *= 2.0
        if tempo > 220:
            tempo /= 2.0
        return tempo
    except Exception:
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
    return 128.0

def detect_kicks(y, sr):
    try:
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
    return float(future[0]) if future.size > 0 else float(kicks[-1])

def slice_kick_aligned(audio_segment, y, sr, bars, bpm, fade_ms, overlap_pct):
    beat_dur = 60.0 / float(bpm)
    loop_dur = bars * 4 * beat_dur
    step = loop_dur * (1 - overlap_pct / 100.0)
    kicks = detect_kicks(y, sr)
    loops = []
    t = 0.0
    # guard for safety: minimum loops if kicks empty
    while t + 0.5 < audio_segment.duration_seconds:
        t0 = snap_to_next_kick(t, kicks)
        t1 = t0 + loop_dur
        if t1 > audio_segment.duration_seconds:
            break
        seg = audio_segment[int(t0 * 1000): int(t1 * 1000)]
        if fade_ms > 0:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)
        loops.append({"start": float(t0), "end": float(t1), "audio": seg})
        t += step
    return loops

def detect_one_shots(path, threshold_db=-35, max_ms=600):
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
                continue
        return shots
    except Exception:
        return []

# ----------------------------
# TSI generation (public schema)
# ----------------------------
def prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')

def color_for_stem(stem_name: str) -> str:
    stem_name = (stem_name or "").lower()
    if "drum" in stem_name or "drums" in stem_name:
        return "#ff6b6b"   # red-ish for drums
    if "bass" in stem_name:
        return "#6bd6ff"   # cyan-ish for bass
    if "voc" in stem_name or "voice" in stem_name:
        return "#ffd56b"   # amber for vocal
    return "#9b7aff"      # purple for other

def generate_tsi(pad_map: dict, sample_base_path: str, deck_name: str, bpm: float, comment: str = "") -> str:
    """
    Build a TSI XML string following a public-schema mapping for a Remix Deck.
    pad_map: dict keyed 0..15 with values like:
        {0: {"assigned": True, "sample_type":"loop", "file":"loops/loop_01.mp3", "loop": True, "stem":"drums", "start":0.0, "end":8.0}, ...}
    sample_base_path: path prefix used inside the TSI to locate sample files (relative path inside the .trak)
    """
    root = ET.Element('TSI', {"version":"1.0"})
    info = ET.SubElement(root, "Info")
    ET.SubElement(info, "Name").text = deck_name
    ET.SubElement(info, "Comment").text = comment or f"Generated by Traktor RemixDeck Builder"
    ET.SubElement(info, "BPM").text = str(round(float(bpm),3))

    remixset = ET.SubElement(root, "RemixSet")
    ET.SubElement(remixset, "Name").text = deck_name
    slots = ET.SubElement(remixset, "Slots")
    for pad_idx in range(16):
        s = ET.SubElement(slots, "Slot", {"index": str(pad_idx)})
        entry = pad_map.get(pad_idx, {})
        assigned = entry.get("assigned", False)
        ET.SubElement(s, "Assigned").text = "1" if assigned else "0"
        if assigned:
            file_path = entry.get("file", "")
            ET.SubElement(s, "File").text = f"{sample_base_path}/{file_path}" if sample_base_path else file_path
            ET.SubElement(s, "Type").text = entry.get("sample_type", "loop")
            ET.SubElement(s, "Loop").text = "1" if entry.get("loop", True) else "0"
            if "start" in entry: ET.SubElement(s, "Start").text = f"{entry.get('start'):.3f}"
            if "end" in entry: ET.SubElement(s, "End").text = f"{entry.get('end'):.3f}"
            color = color_for_stem(entry.get("stem",""))
            ET.SubElement(s, "Color").text = color
            ET.SubElement(s, "Gain").text = str(entry.get("gain", 1.0))
            if "start" in entry and "end" in entry:
                loop_len = float(entry.get("end")) - float(entry.get("start"))
                ET.SubElement(s, "LoopLength").text = f"{loop_len:.3f}"
        else:
            ET.SubElement(s, "File").text = ""
            ET.SubElement(s, "Type").text = ""
            ET.SubElement(s, "Loop").text = "0"
            ET.SubElement(s, "Color").text = "#000000"

    cmap = ET.SubElement(root, "ControllerMapping")
    ET.SubElement(cmap, "Controller").text = "Generic"
    ET.SubElement(cmap, "MappingFor").text = "RemixDeck"
    padmaps = ET.SubElement(cmap, "PadMap")
    for pad_idx in range(16):
        pm = ET.SubElement(padmaps, "Pad", {"index": str(pad_idx)})
        entry = pad_map.get(pad_idx, {})
        if entry.get("assigned", False):
            ET.SubElement(pm, "Slot").text = str(pad_idx)
            ET.SubElement(pm, "Sample").text = entry.get("file","")
            ET.SubElement(pm, "TriggerMode").text = "Loop" if entry.get("loop", True) else "OneShot"
        else:
            ET.SubElement(pm, "Slot").text = str(pad_idx)
            ET.SubElement(pm, "Sample").text = ""
            ET.SubElement(pm, "TriggerMode").text = "None"

    return prettify_xml(root)

# ----------------------------
# Export with TSI integrated
# ----------------------------
def build_trak_with_tsi(loops, shots, bpm, deck_name="RemixDeck", sample_base_path=".", pad_map=None):
    """
    Create a .trak (zip) bytes object containing:
     - loops/... mp3 files
     - oneshots/... mp3 files
     - metadata.json
     - pad_mapping.json
     - deck_name.tsi (generated)
    pad_map should be dict 0..15 -> entry (if None, we'll auto-generate row-major mapping).
    """
    if pad_map is None:
        # auto-generate simple pad_map: loops first then shots
        pad_map = {}
        all_files = []
        for L in loops:
            all_files.append({"sample_type":"loop","file": None, "start": L.get("start"), "end": L.get("end")})
        for S in shots:
            all_files.append({"sample_type":"oneshot","file": None, "time": S.get("time")})
    # Build zip
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        metadata = {"bpm": float(bpm), "loops": [], "one_shots": []}
        # write loops
        for i, L in enumerate(loops, start=1):
            fname = f"loops/loop_{i:02d}.mp3"
            buf = io.BytesIO()
            L["audio"].export(buf, format="mp3", bitrate="320k")
            z.writestr(fname, buf.getvalue())
            metadata["loops"].append({"file": fname, "start": L["start"], "end": L["end"]})
        # write shots
        for i, S in enumerate(shots, start=1):
            fname = f"oneshots/shot_{i:02d}.mp3"
            buf = io.BytesIO()
            S["audio"].export(buf, format="mp3", bitrate="320k")
            z.writestr(fname, buf.getvalue())
            metadata["one_shots"].append({"file": fname, "time": S["time"]})
        # write metadata.json
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
        # create pad_map auto if not supplied: fill pads row-major with loops then shots
        if pad_map is None:
            pad_map = {}
            files_ordered = [m["file"] for m in metadata["loops"]] + [m["file"] for m in metadata["one_shots"]]
            idx = 0
            for pad in range(16):
                if idx < len(files_ordered):
                    f = files_ordered[idx]
                    pad_map[pad] = {"pad": pad, "assigned": True, "sample_type": "loop" if idx < len(metadata["loops"]) else "oneshot", "file": f, "loop": True if idx < len(metadata["loops"]) else False}
                    idx += 1
                else:
                    pad_map[pad] = {"pad": pad, "assigned": False}
        # write pad_mapping.json
        z.writestr("pad_mapping.json", json.dumps(pad_map, indent=2))
        # generate tsi and write
        tsi_text = generate_tsi(pad_map=pad_map, sample_base_path=sample_base_path, deck_name=deck_name, bpm=bpm)
        z.writestr(f"{deck_name}.tsi", tsi_text.encode('utf-8'))
    mem.seek(0)
    return mem

# ----------------------------
# Main: process uploaded file
# ----------------------------
with right:
    if not uploaded:
        st.info("Upload an MP3/WAV on the left to start analysis.")
        st.stop()

    # Save uploaded to a safe temp file for analysis & pydub
    tmpdir = tempfile.mkdtemp(prefix="trk_")
    try:
        ext = Path(uploaded.name).suffix.lower()
        safe_path = Path(tmpdir) / ("upload" + ext)
        with open(safe_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.info(f"Loaded: **{uploaded.name}** — analyzing...")

        # Attempt librosa load; fallback to pydub->numpy if needed
        try:
            y, sr = librosa.load(str(safe_path), sr=None, mono=True)
        except Exception:
            try:
                audio_pd = AudioSegment.from_file(str(safe_path))
                samples = np.array(audio_pd.get_array_of_samples()).astype(np.float32)
                sr = audio_pd.frame_rate
                samples = samples / np.iinfo(audio_pd.array_type).max
                y = samples
            except Exception as err:
                st.error(f"Error reading audio: {err}")
                raise

        # BPM robust detection
        bpm = robust_detect_bpm(y, sr)
        st.success(f"Detected approximate BPM: **{bpm:.2f}**")

        # Use pydub for slicing
        audio_segment = AudioSegment.from_file(str(safe_path))

        # Generate loops
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

        # Previews using st.audio (in-memory)
        st.markdown("### Loop preview")
        if len(loops) == 0:
            st.warning("No loops created — try shorter loop bars or a longer track.")
        else:
            # render a neat preview grid
            cols = st.columns(3)
            for i, L in enumerate(loops):
                buf = io.BytesIO()
                L["audio"].export(buf, format="mp3", bitrate="192k")
                buf.seek(0)
                with cols[i % 3]:
                    st.audio(buf.read(), format="audio/mp3")
                    st.markdown(f"<div class='caption'>Loop {i+1} — {L['start']:.2f}s → {L['end']:.2f}s</div>", unsafe_allow_html=True)

        if shots:
            st.markdown("### One-shot preview")
            cols = st.columns(4)
            for i, S in enumerate(shots):
                buf = io.BytesIO()
                S["audio"].export(buf, format="mp3", bitrate="192k")
                buf.seek(0)
                with cols[i % 4]:
                    st.audio(buf.read(), format="audio/mp3")
                    st.markdown(f"<div class='caption'>Shot {i+1} — {S['time']:.2f}s</div>", unsafe_allow_html=True)

        # Export to .trak (zip) with TSI embedded
        st.markdown("---")
        if st.button("Build Remix Deck (.trak) with TSI"):
            st.info("Exporting deck — packaging MP3s, metadata, pad map and TSI...")
            trak_mem = build_trak_with_tsi(loops, shots, bpm, deck_name=deck_name, sample_base_path=".")
            st.success("Export ready — download below.")
            st.download_button("⬇ Download Remix Deck (.trak)", data=trak_mem.getvalue(), file_name=f"{deck_name}.trak", mime="application/zip")

    finally:
        # Attempt cleanup (keep it tolerant for Cloud)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

# ----------------------------
# Footer
# ----------------------------
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
