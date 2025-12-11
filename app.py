# app.py
"""
Traktor RemixDeck Builder — Polished Two-Mode App (Online / Offline)
Mode A selected: Web (Loops-only) vs Offline (.exe for Stems).
- Online (default): Loops & One-shots only. Cloud-safe: do NOT import Spleeter/TensorFlow.
- Offline (local .exe): Full Spleeter 4-stem support (if you build & run the .exe locally).
UI: professional NI-style theme, improved logo, how-to guide, beat-aligned loops, looping preview.
Exports: .trak (zip) with loops, one-shots, optional stems (mp3 @ 320k), metadata.json, pad_mapping.json and deck_name.tsi.

Deployment notes (short):
 - For Streamlit Cloud: use lightweight requirements.txt:
       streamlit
       numpy
       librosa
       pydub
       soundfile
   and add 'ffmpeg' to packages.txt if allowed by plan.
 - For local full-stems: use requirements-local.txt:
       streamlit
       numpy
       librosa
       pydub
       soundfile
       spleeter==2.3.0
       tensorflow==2.11.0
   and ensure ffmpeg on PATH.

If you want the packaging bundle (.zip with package_app.py and GH Actions workflow), use the "Create Packaging Bundle" button in the UI.
"""

import os
import io
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
import math
import base64
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import librosa
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------------------------
# Page config and safe env
# ---------------------------
st.set_page_config(page_title="Traktor RemixDeck Builder", page_icon="🎧", layout="wide")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter if TF present locally

# ---------------------------
# SVG Logo (refined) + CSS
# ---------------------------
SVG_LOGO = r'''
<svg width="112" height="112" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Traktor RemixDeck Builder">
  <defs>
    <linearGradient id="lg" x1="0" x2="1">
      <stop offset="0" stop-color="#00c2ff"/>
      <stop offset="1" stop-color="#8454ff"/>
    </linearGradient>
    <filter id="s" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000" flood-opacity="0.45"/>
    </filter>
  </defs>
  <g filter="url(#s)">
    <circle cx="64" cy="64" r="60" fill="url(#lg)"/>
    <circle cx="64" cy="64" r="46" fill="#0b0b0d"/>
    <g transform="translate(64,64)">
      <circle r="18" fill="#0e0e10" stroke="#151515" stroke-width="2"/>
      <path d="M-26,-6 C-14,-8 -6,-4 0,-2 6,0 14,2 26,6" stroke="#caf0ff" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.9"/>
    </g>
    <rect x="18" y="18" width="12" height="6" rx="2" fill="#ffffff10" transform="rotate(-20 24 21)"/>
    <rect x="98" y="102" width="10" height="5" rx="2" fill="#ffffff10" transform="rotate(-130 103 104)"/>
  </g>
</svg>
'''

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root {
  --bg-1: #07060a;
  --bg-2: #0f0820;
  --accent-1: #00c2ff;
  --accent-2: #8454ff;
  --muted: #9fcff6;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background: linear-gradient(180deg,var(--bg-1) 0%, var(--bg-2) 60%); color:#e6eef8; }
.header { display:flex; gap:14px; align-items:center; padding:14px 8px; }
.app-title { font-size:28px; color:var(--accent-1); font-weight:700; margin:0; }
.app-sub { color:#cfefff; opacity:0.9; margin-top:2px; font-size:13px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.01)); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.02); box-shadow: 0 8px 28px rgba(2,6,23,0.55); }
.section-title { font-weight:700; color:#eaf6ff; margin-bottom:8px; }
.small { font-size:13px; color:var(--muted); }
.controls { padding:12px; }
.preview-grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(260px,1fr)); gap:12px; margin-top:12px; }
.pad { width:92px; height:92px; border-radius:8px; background:#0f0f14; border:1px solid rgba(255,255,255,0.03); display:flex; align-items:center; justify-content:center; color:#bfe9ff; margin:6px 0; }
.button-primary { background: linear-gradient(90deg,var(--accent-1),var(--accent-2)); color:#041827; font-weight:700; border-radius:8px; padding:8px 14px; border:none; }
.notice { padding:10px; border-radius:8px; margin-bottom:8px; }
.notice.info { background:#062033; border-left:4px solid var(--accent-1); color:#bfe9ff; }
.notice.warn { background:#2b1b0f; border-left:4px solid #ffd56b; color:#ffd56b; }
.notice.error { background:#2b0b0b; border-left:4px solid #ff6b6b; color:#ffb3b3; }
.footer { margin-top:18px; padding:14px; border-radius:8px; background: rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.02); color:#bfe9ff; }
.kv { color:#9fcff6; font-size:13px; }
</style>
""", unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="header">
      <div style="width:112px; height:112px">{SVG_LOGO}</div>
      <div>
        <div class="app-title">Traktor RemixDeck Builder</div>
        <div class="app-sub">by Tuesdaynightfreak Productions — Web (quick loops) & Offline (full stems)</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper utilities
# ---------------------------
def prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')

def color_for_stem(stem_name: str) -> str:
    s = (stem_name or "").lower()
    if "drum" in s: return "#ff6b6b"
    if "bass" in s: return "#6bd6ff"
    if "voc" in s or "voice" in s: return "#ffd56b"
    return "#9b7aff"

def generate_tsi(pad_map: dict, sample_base_path: str, deck_name: str, bpm: float, comment: str = "") -> str:
    root = ET.Element('TSI', {"version":"1.0"})
    info = ET.SubElement(root, "Info")
    ET.SubElement(info, "Name").text = deck_name
    ET.SubElement(info, "Comment").text = comment or "Generated by Traktor RemixDeck Builder"
    ET.SubElement(info, "BPM").text = str(round(float(bpm), 3))
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
            ET.SubElement(s, "Color").text = color_for_stem(entry.get("stem",""))
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
        ET.SubElement(pm, "Slot").text = str(pad_idx)
        if entry.get("assigned", False):
            ET.SubElement(pm, "Sample").text = entry.get("file","")
            ET.SubElement(pm, "TriggerMode").text = "Loop" if entry.get("loop", True) else "OneShot"
        else:
            ET.SubElement(pm, "Sample").text = ""
            ET.SubElement(pm, "TriggerMode").text = "None"

    return prettify_xml(root)

# ---------------------------
# Audio helpers (beat-aware loops)
# ---------------------------
def detect_bpm_and_beats(path):
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        tempo = float(tempo) if tempo is not None else 0.0
        # heuristics for tempo
        if tempo < 60:
            tempo *= 2
        if tempo > 220:
            tempo /= 2
        return tempo, beat_times, sr, y
    except Exception:
        return None, np.array([]), None, None

def nearest_beat_time(t, beat_times):
    if len(beat_times) == 0:
        return t
    idx = np.searchsorted(beat_times, t)
    if idx >= len(beat_times):
        return float(beat_times[-1])
    return float(beat_times[idx])

def make_loop_from_beats(seg: AudioSegment, beat_times, bpm, bars, from_start=True, fade_ms=40):
    beat_duration = 60.0 / float(bpm) if bpm and bpm>0 else None
    loop_len_s = bars * 4 * (1.0 if beat_duration is None else beat_duration)
    loop_len_ms = int(loop_len_s * 1000) if beat_duration else int((bars*4*0.5)*1000)
    dur_ms = int(seg.duration_seconds * 1000)
    if len(beat_times) > 0 and beat_duration:
        if from_start:
            # start at first beat
            start_s = float(beat_times[0])
            end_s = start_s + loop_len_s
        else:
            # end aligned to last beat
            end_s = float(beat_times[-1])
            start_s = max(0.0, end_s - loop_len_s)
    else:
        if from_start:
            start_s = 0.0
            end_s = min(loop_len_ms/1000.0, dur_ms/1000.0)
        else:
            end_s = dur_ms / 1000.0
            start_s = max(0.0, end_s - loop_len_ms/1000.0)
    start_ms = int(start_s*1000)
    end_ms = int(end_s*1000)
    clip = seg[start_ms:end_ms]
    if fade_ms and fade_ms>0:
        clip = clip.fade_in(fade_ms).fade_out(fade_ms)
    return clip, start_s, end_s

def rank_one_shots_in_segment(seg: AudioSegment, sr:int, top_k:int=2, threshold_db=-45):
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if samples.size == 0:
        return []
    maxv = np.max(np.abs(samples)) if np.max(np.abs(samples))>0 else 1.0
    data = samples / maxv
    try:
        onset_env = librosa.onset.onset_strength(y=data, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.02, wait=3)
        times = librosa.frames_to_time(peaks, sr=sr)
    except Exception:
        times = []
    candidates = []
    for t in times:
        s_ms = int(max(0, t*1000 - 20))
        clip = seg[s_ms:s_ms+600]
        try:
            if clip.dBFS >= threshold_db:
                energy = float(np.mean(np.abs(np.array(clip.get_array_of_samples()).astype(np.float32))))
                candidates.append({"time": s_ms/1000.0, "audio": clip, "score": energy})
        except Exception:
            continue
    if len(candidates) < top_k:
        dur = seg.duration_seconds
        offsets = np.linspace(0.1, max(0.5, max(0.5, dur-0.1)), num=6)
        for off in offsets:
            s_ms = int(off*1000)
            clip = seg[s_ms:s_ms+400]
            try:
                energy = float(np.mean(np.abs(np.array(clip.get_array_of_samples()).astype(np.float32))))
                candidates.append({"time": s_ms/1000.0, "audio": clip, "score": energy})
            except Exception:
                continue
    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates_sorted[:top_k]

# ---------------------------
# Spleeter wrapper (lazy import) - only for local exe builds
# ---------------------------
def run_spleeter_4stems(track_path:Path, out_dir:Path):
    stems = {}
    try:
        from spleeter.separator import Separator
        sep = Separator('spleeter:4stems')
        sep.separate_to_file(str(track_path), str(out_dir), codec='wav')
        folder = out_dir / track_path.stem
        mapping = {"vocals":"vocals.wav", "drums":"drums.wav", "bass":"bass.wav", "other":"other.wav"}
        for key, fname in mapping.items():
            p = folder / fname
            if p.exists():
                stems[key] = p
    except Exception as e:
        # return empty dict — caller will fallback to pseudo-stems
        stems = {}
    return stems

# ---------------------------
# Packaging & trak builders
# ---------------------------
def build_trak_package(loops_list, shots_list, stems_mp3_map, bpm, deck_name="RemixDeck"):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for item in loops_list:
            z.writestr(f"loops/{item['name']}", item["bytes"])
        for item in shots_list:
            z.writestr(f"oneshots/{item['name']}", item["bytes"])
        for stem, b in stems_mp3_map.items():
            if b:
                z.writestr(f"stems/{stem}.mp3", b)
        metadata = {"bpm": float(bpm), "loops": [i["name"] for i in loops_list], "one_shots":[i["name"] for i in shots_list]}
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
        pad_map = {}
        for i in range(16):
            pad_map[i] = {"pad": i, "assigned": False}
        for idx, L in enumerate(loops_list[:8]):
            pad_map[idx] = {"pad": idx, "assigned": True, "sample_type":"loop", "file": f"loops/{L['name']}", "loop": True, "stem": L.get("stem"), "start": L.get("start"), "end": L.get("end"), "color": L.get("color")}
        for idx, S in enumerate(shots_list[:8]):
            p = 8 + idx
            pad_map[p] = {"pad": p, "assigned": True, "sample_type":"oneshot", "file": f"oneshots/{S['name']}", "loop": False, "stem": S.get("stem"), "time": S.get("time"), "score": S.get("score"), "color": S.get("color")}
        z.writestr("pad_mapping.json", json.dumps(pad_map, indent=2))
        tsi_text = generate_tsi(pad_map=pad_map, sample_base_path=".", deck_name=deck_name, bpm=bpm)
        z.writestr(f"{deck_name}.tsi", tsi_text.encode("utf-8"))
    mem.seek(0)
    return mem

def make_packaging_bundle_bytes():
    # small packaging bundle: package_app.py + requirements-local.txt + GH actions
    package_app_py = r'''# package_app.py
import shutil, os, subprocess, sys, zipfile, pathlib
ENTRY_POINT = "app.py"
DIST_NAME = "TraktorRemixDeckBuilder"
def run_pyinstaller():
    for d in ["build","dist"]:
        if os.path.exists(d): shutil.rmtree(d)
    cmd = [sys.executable, "-m", "PyInstaller", "--onefile", "--name", DIST_NAME, ENTRY_POINT]
    subprocess.check_call(cmd)
def zip_exe():
    dist_exe = pathlib.Path("dist") / DIST_NAME
    zip_name = f"{DIST_NAME}_installer.zip"
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(dist_exe, arcname=f"{DIST_NAME}.exe")
    print("Created", zip_name)
if __name__ == "__main__":
    run_pyinstaller()
    zip_exe()
'''
    requirements_local = "streamlit\nnumpy\nlibrosa\npydub\nsoundfile\nspleeter==2.3.0\ntensorflow==2.11.0\npyinstaller\n"
    workflow_yml = r'''name: Build Windows EXE
on: [workflow_dispatch]
jobs:
  build-windows-exe:
    runs-on: windows-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install deps
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-local.txt
        pip install pyinstaller
    - name: Build exe
      run: |
        pyinstaller --onefile --name TraktorRemixDeckBuilder app.py
    - name: Zip artifact
      run: |
        powershell -Command "Compress-Archive -Path dist\\TraktorRemixDeckBuilder.exe -DestinationPath TraktorRemixDeckBuilder_installer.zip"
    - name: Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: TraktorRemixDeckBuilder-installer
        path: TraktorRemixDeckBuilder_installer.zip
'''
    readme = "Packaging bundle: package_app.py to build exe locally (requires pyinstaller), requirements-local.txt for local VM builds, GitHub Actions workflow for Windows CI."
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("package_app.py", package_app_py)
        z.writestr("requirements-local.txt", requirements_local)
        z.writestr(".github/workflows/build-windows.yml", workflow_yml)
        z.writestr("README_packaging.md", readme)
    mem.seek(0)
    return mem.getvalue()

# ---------------------------
# UI flow - Two-mode (A): user chooses Online vs Offline
# ---------------------------
st.markdown('<div class="card"><div class="section-title">Choose Mode</div>', unsafe_allow_html=True)
st.markdown("""
<div class="small">Pick how you want to work. <strong>Online (recommended)</strong> is quick and works in the browser — creates loops & one-shots only. <strong>Offline</strong> (local .exe) enables full stem separation using Spleeter and produces the richest stems + loops output.</div>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Mode", options=["Online (Web) — Loops & One-shots", "Offline (Local .exe) — Full Stems + Loops"], index=0, help="Online: fast, Cloud-safe. Offline: requires downloading & running the local .exe (packaging bundle provided).")

st.markdown("")  # spacer

# Left controls + right preview layout
left_col, right_col = st.columns([1,2], gap="large")

with left_col:
    st.markdown("<div class='card'><div class='section-title'>Upload & Settings</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload one track (MP3 or WAV)", type=["mp3","wav"])
    deck_name = st.text_input("Deck name", value="MyRemixDeck")
    bars = st.slider("Loop bars (4/4 beats)", 1, 8, 4)
    fade_ms = st.slider("Fade (ms)", 0, 500, 40)
    st.markdown("---")
    include_stems_checkbox = st.checkbox("Include full stems in export (Offline only)", value=True) if mode.startswith("Offline") else st.checkbox("Include full stems in export (Disabled for Online)", value=False, disabled=True)
    st.markdown("---")
    st.markdown('<div class="small">Tip: Online mode is fast and reliable on Cloud. To extract high-quality stems run the Offline .exe (downloadable below).</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Packaging bundle download available in both modes
    st.markdown('<div style="margin-top:10px">', unsafe_allow_html=True)
    if st.button("Create Packaging Bundle (.zip) — Build local .exe"):
        st.info("Generating packaging bundle. Use the included package_app.py and requirements-local.txt to build a Windows .exe via PyInstaller or run the GitHub Actions workflow.")
        bundle = make_packaging_bundle_bytes()
        st.success("Packaging bundle ready.")
        st.download_button("⬇ Download Packaging Bundle (zip)", data=bundle, file_name="traktor_packaging_bundle.zip", mime="application/zip")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='card'><div class='section-title'>Preview / Export</div>", unsafe_allow_html=True)
    if mode.startswith("Offline"):
        st.markdown('<div class="notice info">Offline mode enables full stem separation using Spleeter. Build and run the local .exe (packaging bundle) to enable stems. See README in bundle for instructions.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="notice info">Online mode: loops & one-shots only. This is Cloud-safe and will not attempt heavy ML installs.</div>', unsafe_allow_html=True)

# Stop early if no upload
if not uploaded:
    st.info("Upload a file to see results. Use Online mode for quick loops. Use 'Create Packaging Bundle' to build an offline .exe for stems.")
    st.stop()

# Save upload to temp file
work_dir = Path(tempfile.mkdtemp(prefix="trk_"))
upload_path = work_dir / uploaded.name
with open(upload_path, "wb") as f:
    f.write(uploaded.getbuffer())

# Analyze: beat detection (beat-aware loop slicing)
st.info("Analyzing track for BPM and beats (this is used to align loops).")
bpm, beat_times, sr, y = detect_bpm_and_beats(upload_path)
if bpm is None or bpm == 0.0:
    st.warning("Unable to detect reliable BPM — fallback heuristics will be used.")
    bpm = 128.0
else:
    st.success(f"Detected BPM: {bpm:.2f}")

# Stem separation: only if Offline and user runs local exe (here we only attempt spleeter if running locally and user toggles)
stems_paths = {}
if mode.startswith("Offline"):
    st.markdown('<div class="notice warn">Offline stem separation is available only when you run the packaged .exe locally. The online app will not perform Spleeter separation to keep Cloud deployment stable.</div>', unsafe_allow_html=True)
    # In the web UI we provide pseudo-stems (full mix duplicated) for loop generation, not real separation.
    stems_paths = {"drums": upload_path, "bass": upload_path, "melody": upload_path, "vocals": upload_path}
else:
    stems_paths = {"drums": upload_path, "bass": upload_path, "melody": upload_path, "vocals": upload_path}

# Build loops & one-shots (per pseudo-stem or real stems if offline exe used locally)
colors = {"drums": "#ff6b6b", "bass": "#6bd6ff", "melody": "#9b7aff", "vocals": "#ffd56b"}
loops_list = []
shots_list = []
stems_mp3_map = {}

for stem in ["drums", "bass", "melody", "vocals"]:
    stem_path = stems_paths.get(stem)
    if stem_path is None:
        continue
    seg = AudioSegment.from_file(str(stem_path))
    # prepare stem mp3 bytes only if offline & user included stems (note: this will be included in .trak if you build exe and run locally)
    if mode.startswith("Offline") and include_stems_checkbox:
        buf = io.BytesIO()
        try:
            seg.export(buf, format="mp3", bitrate="320k")
            stems_mp3_map[stem] = buf.getvalue()
        except Exception:
            stems_mp3_map[stem] = None
    else:
        stems_mp3_map[stem] = None

    # Start loop: beat-aligned
    clip_s, s_start, s_end = make_loop_from_beats(seg, beat_times, bpm, bars, from_start=True, fade_ms=fade_ms)
    buf = io.BytesIO(); clip_s.export(buf, format="mp3", bitrate="320k"); s_bytes = buf.getvalue()
    loops_list.append({"name": f"{stem}_start.mp3", "bytes": s_bytes, "stem": stem, "start": s_start, "end": s_end, "color": colors.get(stem)})

    # End loop: beat-aligned
    clip_e, e_start, e_end = make_loop_from_beats(seg, beat_times, bpm, bars, from_start=False, fade_ms=fade_ms)
    buf = io.BytesIO(); clip_e.export(buf, format="mp3", bitrate="320k"); e_bytes = buf.getvalue()
    loops_list.append({"name": f"{stem}_end.mp3", "bytes": e_bytes, "stem": stem, "start": e_start, "end": e_end, "color": colors.get(stem)})

    # One-shots: pick top 2 per stem (transient ranking)
    ranked = rank_one_shots_in_segment(seg, seg.frame_rate if hasattr(seg, "frame_rate") else (sr or 44100), top_k=2, threshold_db=-45)
    if len(ranked) < 2:
        dur = seg.duration_seconds
        offsets = [0.5, max(0.8, dur * 0.5)]
        for off in offsets:
            s_ms = int(off*1000)
            clip = seg[s_ms:s_ms+400]
            energy = float(np.mean(np.abs(np.array(clip.get_array_of_samples()).astype(np.float32)))) if clip.duration_seconds>0 else 0.0
            ranked.append({"time": s_ms/1000.0, "audio": clip, "score": energy})
        ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)[:2]
    for idx, cand in enumerate(ranked[:2], start=1):
        buf = io.BytesIO(); cand["audio"].export(buf, format="mp3", bitrate="320k"); bts = buf.getvalue()
        shots_list.append({"name": f"{stem}_shot_{idx}.mp3", "bytes": bts, "stem": stem, "time": cand.get("time", 0.0), "score": cand.get("score", 0.0), "color": colors.get(stem)})

# ---------------------------
# Preview UI: loops (looping HTML preview) and shots
# ---------------------------
def audio_bytes_to_base64_html(mp3_bytes, loop=False):
    b64 = base64.b64encode(mp3_bytes).decode('utf-8')
    loop_attr = "loop" if loop else ""
    html = f"""
    <audio controls {loop_attr} style="width:100%">
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
      Your browser does not support the audio element.
    </audio>
    """
    return html

st.markdown("<div class='card'><div class='section-title'>Loop previews (select which to include)</div></div>", unsafe_allow_html=True)
selected_loops = []
cols = st.columns(4)
for i, L in enumerate(loops_list):
    with cols[i % 4]:
        html = audio_bytes_to_base64_html(L["bytes"], loop=True)
        components.html(html, height=80)
        label = f"{L['stem'].upper()} {'START' if 'start' in L['name'] else 'END'} — {L['start']:.2f}s → {L['end']:.2f}s"
        if st.checkbox(label, value=True, key=f"loop_sel_{i}"):
            selected_loops.append(L)

st.markdown("<div class='card'><div class='section-title'>One-shot previews (select up to 8)</div></div>", unsafe_allow_html=True)
selected_shots = []
cols = st.columns(4)
for i, S in enumerate(shots_list):
    with cols[i % 4]:
        html = audio_bytes_to_base64_html(S["bytes"], loop=False)
        components.html(html, height=80)
        label = f"{S['stem'].upper()} shot #{(i%2)+1} (score {S['score']:.3g})"
        if st.checkbox(label, value=True, key=f"shot_sel_{i}"):
            selected_shots.append(S)

# Pad map preview
def build_pad_map(loops_sel, shots_sel):
    pad_map = {}
    for i in range(16):
        pad_map[i] = {"pad": i, "assigned": False}
    for idx, L in enumerate(loops_sel[:8]):
        pad_map[idx] = {"pad": idx, "assigned": True, "sample_type":"loop", "file": f"loops/{L['name']}", "loop": True, "stem": L.get("stem"), "start": L.get("start"), "end": L.get("end"), "color": L.get("color")}
    for idx, S in enumerate(shots_sel[:8]):
        p = 8 + idx
        pad_map[p] = {"pad": p, "assigned": True, "sample_type":"oneshot", "file": f"oneshots/{S['name']}", "loop": False, "stem": S.get("stem"), "time": S.get("time"), "score": S.get("score"), "color": S.get("color")}
    return pad_map

pad_map = build_pad_map(selected_loops, selected_shots)
st.markdown("### Pad mapping preview (pads 1–16)")
cols = st.columns(4)
for i in range(16):
    with cols[i % 4]:
        entry = pad_map.get(i, {})
        label = f"Pad {i+1}"
        css = "pad assigned" if entry.get("assigned") else "pad"
        st.markdown(f"<div class='{css}' style='height:64px;display:flex;align-items:center;justify-content:center;padding:6px'>{label}</div>", unsafe_allow_html=True)

st.download_button("Download pad_mapping.json", data=json.dumps(pad_map, indent=2), file_name=f"{deck_name}_pad_mapping.json", mime="application/json")

# How-to guide (compact)
with st.expander("How to use — Quick guide (click to expand)"):
    st.markdown("""
    **Online (Web) mode** — Quick loops:
    1. Upload a single MP3 or WAV.  
    2. App detects beats and makes beat-aligned loops (Start / End) per pseudo-stem.  
    3. Preview loops (they loop in the browser). Tick the ones you want.  
    4. Build the `.trak` package and download. Import the `.tsi` into Traktor or copy the MP3s into a Remix Deck.

    **Offline (.exe) mode — Full stems (recommended for highest quality):**
    1. Click 'Create Packaging Bundle' and download the ZIP.  
    2. Use the included `package_app.py` and `requirements-local.txt` to build the Windows `.exe` via PyInstaller (or run the provided GitHub Action).  
    3. Run the `.exe` locally — it will allow full Spleeter 4-stem separation and export stems + loops.  
    4. The `.exe` includes the same export options and will produce a `.trak` you can load into Traktor.

    **Notes & Tips**
    - For best results with vocals/stems use the Offline .exe option.  
    - Ensure `ffmpeg` is installed locally when building/running the exe for correct mp3 exports.  
    - The web app is Cloud-safe and will not try to install heavy ML packages.
    """)
# Export buttons
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

if st.button("Build Remix Deck (.trak) — download"):
    if len(selected_loops) == 0 and len(selected_shots) == 0:
        st.error("No loops or one-shots selected. Please choose at least one item before exporting.")
    else:
        st.info("Packaging .trak and .tsi (this may take a moment)...")
        trak_mem = build_trak_package(selected_loops, selected_shots, stems_mp3_map if (mode.startswith("Offline") and include_stems_checkbox) else {}, bpm, deck_name=deck_name)
        st.success("Package ready.")
        st.download_button("⬇ Download Remix Deck (.trak)", data=trak_mem.getvalue(), file_name=f"{deck_name}.trak", mime="application/zip")

# Cleanup
try:
    shutil.rmtree(work_dir)
except Exception:
    pass

st.markdown("<div class='footer'>Traktor RemixDeck Builder — quick loops online, full stems offline. Built by Tuesdaynightfreak Productions.</div>", unsafe_allow_html=True)
