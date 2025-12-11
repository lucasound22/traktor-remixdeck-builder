# app.py
"""
Traktor RemixDeck Builder — Patched UI + looping previews + export modes + packaging bundle
- Option A: Full Spleeter (local/VM). Lazy import to avoid boot crashes.
- Export modes: "Stems + Loops + One-shots" or "Loops & One-shots (no stems)"
- Fixed loop slicing and looping preview (browser <audio loop>)
- Generates .trak (zip) with loops/oneshots/stems (optional) + metadata + pad_mapping.json + generated .tsi
- Provides downloadable packaging bundle (scripts + workflow) to build a Windows .exe locally or with GitHub Actions
Notes:
 - Requires ffmpeg on PATH for pydub -> mp3 export
 - For Spleeter run locally: spleeter==2.3.0 and a compatible TensorFlow in requirements
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
# Page config + env
# ---------------------------
st.set_page_config(page_title="Traktor RemixDeck Builder", page_icon="🎧", layout="wide")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF logs if TF present

# ---------------------------
# Professional SVG logo + CSS (NI / Traktor inspired)
# ---------------------------
SVG_LOGO = r'''
<svg width="110" height="110" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Traktor RemixDeck Builder">
  <defs>
    <linearGradient id="lg" x1="0" x2="1">
      <stop offset="0" stop-color="#18b7ff"/>
      <stop offset="1" stop-color="#8f5aff"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="6" stdDeviation="12" flood-color="#000" flood-opacity="0.45"/>
    </filter>
  </defs>
  <g filter="url(#shadow)">
    <circle cx="64" cy="64" r="60" fill="url(#lg)"/>
    <circle cx="64" cy="64" r="46" fill="#070809"/>
    <g transform="translate(64,64)">
      <circle r="18" fill="#0d0d0f" stroke="#1b1b1f" stroke-width="2"/>
      <path d="M-26,-6 C-14,-8 -6,-4 0,-2 6,0 14,2 26,6" stroke="#cfefff" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.9"/>
    </g>
    <rect x="12" y="12" width="16" height="6" rx="2" fill="#ffffff20" transform="rotate(-22 20 15)"/>
    <rect x="100" y="110" width="12" height="6" rx="3" fill="#ffffff10" transform="rotate(-140 106 113)"/>
  </g>
</svg>
'''

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background: linear-gradient(180deg,#07060a 0%, #0f0620 60%); color:#e6eef8; }
.header { display:flex; gap:16px; align-items:center; padding:18px; }
.app-title { font-size:28px; color:#18b7ff; font-weight:700; margin:0; }
.app-sub { color:#cfefff; opacity:0.9; margin-top:2px; font-size:13px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.01)); padding:12px; border-radius:12px; border:1px solid rgba(255,255,255,0.025); box-shadow: 0 8px 30px rgba(2,6,23,0.55); }
.preview-grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(240px,1fr)); gap:12px; margin-top:12px; }
.caption { color:#9fcff6; font-size:13px; opacity:0.95; margin-top:6px; }
.pad { width:84px; height:84px; border-radius:8px; background:#0f0f14; border:1px solid rgba(255,255,255,0.03); display:flex; align-items:center; justify-content:center; color:#bfe9ff; }
.pad.assigned { background: linear-gradient(135deg, rgba(10,132,255,0.06), rgba(155,90,255,0.04)); border:1px solid rgba(10,132,255,0.14); }
.control-label { color:#cfefff; font-weight:600; }
.section-title { color:#eaf6ff; font-size:16px; font-weight:700; margin-top:8px; }
.small { font-size:13px; color:#9fcff6; }
.btn { background: linear-gradient(90deg,#0a84ff,#2aa6ff); color:#041827; font-weight:600; border-radius:8px; padding:8px 14px; border: none; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    f"""
    <div class="header">
      <div style="width:110px; height:110px">{SVG_LOGO}</div>
      <div>
        <div class="app-title">Traktor RemixDeck Builder</div>
        <div class="app-sub">by Tuesdaynightfreak Productions — Stems, Loops, One-shots & Traktor-ready .trak/.tsi exports</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper: XML pretty print
# ---------------------------
def prettify_xml(elem: ET.Element) -> str:
    rough = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')

# ---------------------------
# TSI generation (public-schema)
# ---------------------------
def color_for_stem(stem_name: str) -> str:
    s = (stem_name or "").lower()
    if "drum" in s: return "#ff6b6b"
    if "bass" in s: return "#6bd6ff"
    if "voc" in s or "voice" in s: return "#ffd56b"
    return "#9b7aff"  # melody/other

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
# Audio helpers
# ---------------------------
def robust_detect_bpm(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if tempo is None or np.isnan(tempo): raise ValueError
        tempo = float(tempo)
        if tempo < 60: tempo *= 2
        if tempo > 220: tempo /= 2
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

def make_loop_from_segment(seg: AudioSegment, bpm: float, bars: int, fade_ms: int, from_start: bool = True):
    # produce an exact loop-length clip (bars * 4 beats)
    beat = 60.0/float(bpm)
    loop_len_ms = int((bars * 4 * beat) * 1000)
    dur_ms = int(seg.duration_seconds * 1000)
    if from_start:
        start_ms = 0
        end_ms = min(loop_len_ms, dur_ms)
    else:
        end_ms = dur_ms
        start_ms = max(0, dur_ms - loop_len_ms)
    clip = seg[start_ms:end_ms]
    if fade_ms > 0:
        clip = clip.fade_in(fade_ms).fade_out(fade_ms)
    return clip, start_ms/1000.0, end_ms/1000.0

def rank_one_shots_in_segment(seg: AudioSegment, sr: int, top_k: int = 2, threshold_db: int = -45):
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if samples.size == 0:
        return []
    maxv = np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else 1.0
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
        offsets = np.linspace(0.1, max(0.5, dur-0.1), num=6)
        for off in offsets:
            s_ms = int(off*1000)
            clip = seg[s_ms:s_ms+400]
            try:
                energy = float(np.mean(np.abs(np.array(clip.get_array_of_samples()).astype(np.float32))))
                candidates.append({"time": s_ms/1000.0, "audio": clip, "score": energy})
            except Exception:
                continue
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]

# ---------------------------
# Spleeter wrapper (lazy import)
# ---------------------------
def run_spleeter_4stems(track_path: Path, out_dir: Path):
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
        st.error("Spleeter separation failed. See server logs.")
        st.text(str(e))
    return stems

# ---------------------------
# Build .trak zip (includes TSI)
# ---------------------------
def build_trak_package(loops_list, shots_list, stems_mp3_map, bpm, deck_name="RemixDeck"):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # loops
        for item in loops_list:
            z.writestr(f"loops/{item['name']}", item["bytes"])
        # oneshots
        for item in shots_list:
            z.writestr(f"oneshots/{item['name']}", item["bytes"])
        # stems
        for stem, b in stems_mp3_map.items():
            if b:
                z.writestr(f"stems/{stem}.mp3", b)
        # metadata
        metadata = {"bpm": float(bpm), "loops": [i["name"] for i in loops_list], "one_shots": [i["name"] for i in shots_list]}
        z.writestr("metadata.json", json.dumps(metadata, indent=2))
        # pad map: pads 0-7 loops, 8-15 oneshots
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

# ---------------------------
# Packaging bundle generator (scripts + workflow) which user can download and run locally/CI to produce .exe
# ---------------------------
def make_packaging_bundle_bytes():
    # package_app.py content
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
    # requirements.txt snippet
    requirements_txt = "streamlit\nnumpy\nlibrosa\npydub\nsoundfile\nspleeter==2.3.0\ntensorflow==2.11.0\n"
    # GitHub Actions workflow
    workflow_yml = r'''name: Build Windows EXE
on: [workflow_dispatch]
jobs:
  build-windows-exe:
    runs-on: windows-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with: python-version: '3.10'
    - name: Install deps
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
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
    readme = r'''# Packaging Bundle
Run package_app.py locally (requires pyinstaller) or use the included GitHub Actions workflow to build a Windows .exe.
Ensure you have ffmpeg installed on the target machine.
'''
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("package_app.py", package_app_py)
        z.writestr("requirements.txt", requirements_txt)
        z.writestr(".github/workflows/build-windows.yml", workflow_yml)
        z.writestr("README_packaging.md", readme)
    mem.seek(0)
    return mem.getvalue()

# ---------------------------
# UI and main flow
# ---------------------------
st.markdown("<div class='card'><div class='section-title'>Upload & Settings</div></div>", unsafe_allow_html=True)
left, right = st.columns([1,2], gap="large")

with left:
    uploaded = st.file_uploader("Upload one mixed track (MP3 or WAV)", type=["mp3","wav"])
    export_mode = st.radio("Export mode", options=["Stems + Loops + One-shots", "Loops & One-shots (no stems)"], index=0)
    use_spleeter = st.checkbox("Use Spleeter 4-stems (local/VM only - lazy import)", value=True)
    include_stems_checkbox = st.checkbox("Include full stems MP3s in output (when stems enabled)", value=True)
    st.markdown("---")
    bars = st.slider("Loop bars (4/4 beats)", 1, 8, 4)
    fade_ms = st.slider("Fade (ms)", 0, 500, 40)
    overlap = st.slider("Overlap (%) — preview only", min_value=0, max_value=50, value=0)
    st.markdown("---")
    deck_name = st.text_input("Deck name", value="MyRemixDeck")
    st.caption("Note: ensure ffmpeg is installed and on PATH for MP3 conversion (pydub).")

with right:
    st.markdown("<div class='card'><div class='section-title'>Preview / Export</div></div>", unsafe_allow_html=True)

if not uploaded:
    st.info("Upload a track to begin. If you're on Streamlit Cloud, Spleeter/TensorFlow may not run successfully—use local/VM for Spleeter.")
    st.stop()

# Save upload
work_dir = Path(tempfile.mkdtemp(prefix="trk_"))
upload_path = work_dir / uploaded.name
with open(upload_path, "wb") as f:
    f.write(uploaded.getbuffer())

# Load for analysis
st.info("Loading audio for analysis...")
try:
    y, sr = librosa.load(str(upload_path), sr=None, mono=True)
except Exception:
    try:
        pdseg = AudioSegment.from_file(str(upload_path))
        samples = np.array(pdseg.get_array_of_samples()).astype(np.float32)
        sr = pdseg.frame_rate
        y = samples / (np.iinfo(pdseg.array_type).max if hasattr(pdseg, "array_type") else (2**15))
    except Exception as e:
        st.error(f"Could not load audio: {e}")
        raise

bpm = robust_detect_bpm(y, sr)
st.success(f"Detected BPM: {bpm:.2f}")

# Stem separation decision
stems_paths = {}
if export_mode.startswith("Stems") and use_spleeter:
    st.info("Running Spleeter 4stems (lazy import) — this may take time on your machine.")
    spleeter_out = work_dir / "spleeter_out"
    spleeter_out.mkdir(exist_ok=True)
    stems_paths = run_spleeter_4stems(upload_path, spleeter_out)
    if not stems_paths:
        st.warning("Spleeter failed or returned no stems. Falling back to pseudo-stems (full mix).")
        stems_paths = {"drums": upload_path, "bass": upload_path, "melody": upload_path, "vocals": upload_path}
else:
    # loops-only (no stems) or spleeter disabled: use pseudo-stems (full mix split)
    if export_mode.startswith("Stems") and not use_spleeter:
        st.warning("Stems export selected but Spleeter disabled — using pseudo-stems (full mix).")
    stems_paths = {"drums": upload_path, "bass": upload_path, "melody": upload_path, "vocals": upload_path}

# Normalize mapping if Spleeter returns 'other'
if "other" in stems_paths and "melody" not in stems_paths:
    stems_paths["melody"] = stems_paths["other"]

# Generate loops & one-shots per stem (4 stems: drums,bass,melody,vocals)
colors = {"drums": "#ff6b6b", "bass": "#6bd6ff", "melody": "#9b7aff", "vocals": "#ffd56b"}
loops_list = []
shots_list = []
stems_mp3_map = {}

for stem in ["drums", "bass", "melody", "vocals"]:
    stem_path = stems_paths.get(stem)
    if stem_path is None:
        continue
    seg = AudioSegment.from_file(str(stem_path))
    # optional full stem MP3 bytes
    if export_mode.startswith("Stems") and include_stems_checkbox:
        buf = io.BytesIO()
        seg.export(buf, format="mp3", bitrate="320k")
        stems_mp3_map[stem] = buf.getvalue()
    else:
        stems_mp3_map[stem] = None

    # start loop (exact loop length)
    clip_s, s_start, s_end = make_loop_from_segment(seg, bpm, bars, fade_ms, from_start=True)
    buf = io.BytesIO(); clip_s.export(buf, format="mp3", bitrate="320k"); s_bytes = buf.getvalue()
    loops_list.append({"name": f"{stem}_start.mp3", "bytes": s_bytes, "stem": stem, "start": s_start, "end": s_end, "color": colors.get(stem)})

    # end loop (exact loop length)
    clip_e, e_start, e_end = make_loop_from_segment(seg, bpm, bars, fade_ms, from_start=False)
    buf = io.BytesIO(); clip_e.export(buf, format="mp3", bitrate="320k"); e_bytes = buf.getvalue()
    loops_list.append({"name": f"{stem}_end.mp3", "bytes": e_bytes, "stem": stem, "start": e_start, "end": e_end, "color": colors.get(stem)})

    # one-shots: rank top 2
    ranked = rank_one_shots_in_segment(seg, seg.frame_rate, top_k=2, threshold_db=-45)
    if len(ranked) < 2:
        dur = seg.duration_seconds
        offsets = [0.5, max(0.8, dur * 0.5)]
        for off in offsets:
            s_ms = int(off*1000)
            clip = seg[s_ms:s_ms+400]
            energy = float(np.mean(np.abs(np.array(clip.get_array_of_samples()).astype(np.float32)))) if clip.duration_seconds > 0 else 0.0
            ranked.append({"time": s_ms/1000.0, "audio": clip, "score": energy})
        ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)[:2]
    for idx, cand in enumerate(ranked[:2], start=1):
        buf = io.BytesIO(); cand["audio"].export(buf, format="mp3", bitrate="320k"); bts = buf.getvalue()
        shots_list.append({"name": f"{stem}_shot_{idx}.mp3", "bytes": bts, "stem": stem, "time": cand.get("time", 0.0), "score": cand.get("score", 0.0), "color": colors.get(stem)})

# UI: previews — use embedded <audio loop> for looping preview
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

st.markdown("## Loop previews (select which to include)")
selected_loops = []
cols = st.columns(4)
for i, L in enumerate(loops_list):
    with cols[i % 4]:
        html = audio_bytes_to_base64_html(L["bytes"], loop=True)
        components.html(html, height=80)
        label = f"{L['stem'].upper()} {'START' if 'start' in L['name'] else 'END'} — {L['start']:.2f}s → {L['end']:.2f}s"
        if st.checkbox(label, value=True, key=f"loop_sel_{i}"):
            selected_loops.append(L)

st.markdown("## One-shot previews (select up to 8)")
selected_shots = []
cols = st.columns(4)
for i, S in enumerate(shots_list):
    with cols[i % 4]:
        html = audio_bytes_to_base64_html(S["bytes"], loop=False)
        components.html(html, height=80)
        label = f"{S['stem'].upper()} shot #{(i%2)+1} (score {S['score']:.3g})"
        if st.checkbox(label, value=True, key=f"shot_sel_{i}"):
            selected_shots.append(S)

# Pad mapping preview builder
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

# Packaging bundle for building a Windows exe locally or in CI
if st.button("Create Packaging Bundle (zip) — build .exe locally/CI"):
    st.info("Generating packaging bundle (package_app.py, requirements.txt, GitHub Actions workflow)...")
    bundle_bytes = make_packaging_bundle_bytes()
    st.success("Packaging bundle ready — download below.")
    st.download_button("⬇ Download Packaging Bundle (zip)", data=bundle_bytes, file_name="packaging_bundle.zip", mime="application/zip")

# Final export button
if st.button("Build Remix Deck (.trak) with TSI and MP3s"):
    if len(selected_loops) == 0 and len(selected_shots) == 0:
        st.error("No loops or one-shots selected. Please select items to include.")
    else:
        st.info("Building .trak package (this may take a few seconds)...")
        trak_mem = build_trak_package(selected_loops, selected_shots, stems_mp3_map, bpm, deck_name=deck_name)
        st.success("Package built — download below.")
        st.download_button("⬇ Download Remix Deck (.trak)", data=trak_mem.getvalue(), file_name=f"{deck_name}.trak", mime="application/zip")

# Cleanup temp
try:
    shutil.rmtree(work_dir)
except Exception:
    pass

st.caption("Local mode: Spleeter 4-stem enabled (if chosen). Exports are MP3 @ 320 kbps. For large tracks ensure you have sufficient RAM and ffmpeg installed.")

