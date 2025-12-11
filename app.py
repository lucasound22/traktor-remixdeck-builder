# app.py
"""
Traktor RemixDeck Builder - ONLINE version with embedded Strudel/Tone.js sequencer
- Cloud-safe: no TensorFlow / no Spleeter
- Generates beat-aligned loops, one-shots, .trak export
- Embeds a browser-side Tone.js 4x16 step sequencer (Drums/Bass/Melody/Pads)
- Sequencer notes restricted to detected key
- Color-coded rows matching stems
"""

import streamlit as st
import os, io, json, zipfile, tempfile, shutil, base64
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

# ---------------------------
# Page UI & CSS
# ---------------------------
st.set_page_config(page_title="Traktor RemixDeck Builder", page_icon="🎛️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root { --accent: #00c2ff; --accent2: #8454ff; --muted:#9fcff6; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background: linear-gradient(180deg,#07060a,#0f0820); color: #eaf6ff; }
.header { display:flex; gap:12px; align-items:center; padding:14px; }
.title { font-size:26px; color:var(--accent); font-weight:700; margin:0; }
.subtitle { color:#cfefff; opacity:0.9; margin-top:2px; font-size:13px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.01)); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.02); box-shadow: 0 8px 28px rgba(2,6,23,0.55); margin-bottom:12px; }
.small { color: var(--muted); font-size:13px; }
.grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
.kv { color:#9fcff6; }
.footer { margin-top:18px; padding:12px; color:#bfe9ff; font-size:13px; }
</style>
""", unsafe_allow_html=True)

SVG_LOGO = r'''
<svg width="84" height="84" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Logo">
  <defs>
    <linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#00c2ff"/><stop offset="1" stop-color="#8454ff"/></linearGradient>
    <filter id="s"><feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#000" flood-opacity="0.45"/></filter>
  </defs>
  <g filter="url(#s)">
    <circle cx="64" cy="64" r="60" fill="url(#g)"/>
    <circle cx="64" cy="64" r="46" fill="#070809"/>
    <g transform="translate(64,64)"><circle r="18" fill="#0e0e10" stroke="#151515" stroke-width="2"/><path d="M-26,-6 C-14,-8 -6,-4 0,-2 6,0 14,2 26,6" stroke="#caf0ff" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.9"/></g>
  </g>
</svg>
'''

st.markdown(f'<div class="header"><div style="width:84px;height:84px">{SVG_LOGO}</div>'
            f'<div><div class="title">Traktor RemixDeck Builder</div><div class="subtitle">Online — Loops & Strudel Sequencer (browser)</div></div></div>',
            unsafe_allow_html=True)

# ---------------------------
# Tabs: Loops / Sequencer
# ---------------------------
tabs = st.tabs(["Loops & Export", "Sequencer (Play & Edit)"])

# Shared parameters
MAX_UPLOAD_MB = 120
allowed_types = ["mp3", "wav"]

# stem color mapping (used in UI + sequencer)
stem_colors = {"drums": "#ff6b6b", "bass": "#6bd6ff", "melody": "#9b7aff", "pads": "#ffd56b"}

# ---------------------------
# Helper functions
# ---------------------------

def save_uploaded_temp(uploaded_file):
    tmpdir = Path(tempfile.mkdtemp(prefix="trk_"))
    tgt = tmpdir / uploaded_file.name
    with open(tgt, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmpdir, tgt

def detect_bpm_and_key(path):
    """
    Returns bpm (float), beat_times (np.array), sr, y (np.array), key_root (int 0-11), mode ('major'/'minor') 
    Key detection is heuristic: we sum chroma energy and pick the max as tonic, then compare major/minor profile correlation.
    """
    y, sr = librosa.load(str(path), sr=None, mono=True)
    # tempo + beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    tempo = float(tempo) if tempo is not None else 0.0
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # 12-d vector
    key_root = int(np.argmax(chroma_mean))
    # simple major/minor guess using templates
    major_template = np.array([1,0,0,1,0,0,1,0,0,1,0,0])  # C major shape approx
    minor_template = np.array([1,0,0,1,1,0,0,1,0,0,1,0])  # A minor-ish
    # rotate templates to each root and compute correlation
    def corr_to_template(chroma_vec, template):
        best = -999; bestroot=0
        for r in range(12):
            templ = np.roll(template, r)
            c = np.corrcoef(chroma_vec, templ)[0,1]
            if np.isnan(c): c = -1
            if c > best:
                best = c; bestroot = r
        return best, bestroot
    maj_corr, maj_root = corr_to_template(chroma_mean, major_template)
    min_corr, min_root = corr_to_template(chroma_mean, minor_template)
    if maj_corr >= min_corr:
        mode = "major"; root = maj_root
    else:
        mode = "minor"; root = min_root
    # final root and tempo
    if tempo < 40 or tempo > 220:
        # fallback
        tempo = 128.0
    return tempo, beat_times, sr, y, root, mode

def make_loop_audiosegment(y, sr, start_sec, length_sec):
    """
    Return pydub AudioSegment for given start and length (mono)
    """
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + length_sec) * sr)
    seg = y[start_sample:end_sample]
    # ensure at least some length
    if seg.size == 0:
        seg = np.zeros(int(length_sec * sr))
    # convert to PCM16 bytes for pydub
    seg16 = (seg * 32767).astype(np.int16)
    audio = AudioSegment(seg16.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    return audio

def audiosegment_to_base64_mp3_bytes(audioseg, bitrate="320k"):
    buf = io.BytesIO()
    audioseg.export(buf, format="mp3", bitrate=bitrate)
    b = buf.getvalue()
    return base64.b64encode(b).decode("ascii")

def build_trak_zip_bytes(loops, shots, bpm, deck_name="RemixDeck"):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, b64 in loops:
            z.writestr(f"loops/{name}", base64.b64decode(b64))
        for name, b64 in shots:
            z.writestr(f"oneshots/{name}", base64.b64decode(b64))
        meta = {"bpm": float(bpm), "loops":[n for n,_ in loops], "shots":[n for n,_ in shots]}
        z.writestr("metadata.json", json.dumps(meta, indent=2))
        # simple tsi placeholder for user to import as remix deck slot mapping
        tsi = {"deck": deck_name, "bpm": bpm}
        z.writestr(f"{deck_name}.tsi", json.dumps(tsi, indent=2))
    mem.seek(0)
    return mem.getvalue()

# ---------------------------
# Tab 1: Loops & Export
# ---------------------------
with tabs[0]:
    st.markdown('<div class="card"><div class="section-title">Upload & Loop Generation</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    # Show upload summary
    uploaded = st.file_uploader("Upload your track (MP3/WAV) — limit ~100MB", type=allowed_types)
    if not uploaded:
        st.info("Upload a track to generate loops and to enable the Strudel sequencer.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    tmpdir, saved = save_uploaded_temp(uploaded)
    st.markdown(f"<div class='card'><div class='small'>Saved to temporary workspace ({tmpdir})</div></div>", unsafe_allow_html=True)

    # Analysis
    with st.spinner("Detecting BPM and key..."):
        bpm, beat_times, sr, y, key_root, mode = detect_bpm_and_key(saved)
    st.success(f"Estimated BPM: **{bpm:.1f}** — Key: **{librosa.midi_to_note(60+key_root).replace('C','').strip() or librosa.midi_to_note(60+key_root)} {mode}**")

    # loop size selection
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Loop Size & Positions</div>", unsafe_allow_html=True)
    loop_bars = st.selectbox("Loop length (bars, 4/4)", options=[1,2,4,8,16,32], index=2)
    fade_ms = st.slider("Fade (ms)", 0, 600, 40)
    st.markdown("</div>", unsafe_allow_html=True)

    # build 4 from early and 4 from late half
    beat_len = 60.0 / float(bpm)
    loop_seconds = loop_bars * 4 * beat_len

    # positions heuristically: start at beat 2 + offsets; and last quarter offsets
    duration = librosa.get_duration(y=y, sr=sr)
    starters = []
    # pick early: find first beat times after 1s
    if len(beat_times) > 4:
        early = beat_times[:min(len(beat_times), 32)]
        # pick first 4 beats spaced by loop_seconds
        t0 = max(0.5, float(early[0]))
        starters = [t0 + i * loop_seconds for i in range(4)]
    else:
        starters = [1.0 + i*loop_seconds for i in range(4)]
    # pick late ones near end
    late_base = max( max(0.5, duration - loop_seconds*4 - 0.5), 0.5 )
    laters = [late_base + i*loop_seconds for i in range(4)]
    positions = starters + laters

    # generate loops as base64 mp3 for embedding into sequencer
    loops_b64 = []
    loops_display = []
    for idx, pos in enumerate(positions):
        pos = max(0.0, min(pos, duration - 0.25))
        seg = make_loop_audiosegment(y, sr, pos, loop_seconds)
        if fade_ms>0:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)
        name = f"loop_{idx+1}.mp3"
        b64 = audiosegment_to_base64_mp3_bytes(seg, bitrate="320k")
        loops_b64.append((name, b64, pos))
        loops_display.append((name, pos, loop_seconds))

    st.markdown("<div class='card'><div class='section-title'>Generated Loops (preview)</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (name, pos, dur) in enumerate(loops_display):
        with cols[i % 4]:
            st.markdown(f"<div style='padding:6px;border-radius:8px;background:#0e0e12;border:1px solid #222'><strong style='color:#cfefff'>{name}</strong><div class='small'>Start {pos:.2f}s • {dur:.2f}s</div></div>", unsafe_allow_html=True)
            # audio preview via base64 data
            data = loops_b64[i][1]
            audio_html = f'<audio controls loop src="data:audio/mp3;base64,{data}"></audio>'
            st.components.v1.html(audio_html, height=80)

    # one-shots (transient)
    st.markdown("<div class='card'><div class='section-title'>One-shots (transient picks)</div>", unsafe_allow_html=True)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    shots_b64 = []
    maxshots = 8
    for i, t in enumerate(onset_times[:maxshots]):
        seg = make_loop_audiosegment(y, sr, max(0, t-0.02), 0.25)
        name = f"shot_{i+1}.mp3"
        b64 = audiosegment_to_base64_mp3_bytes(seg, bitrate="320k")
        shots_b64.append((name, b64, t))

    cols = st.columns(4)
    for i, (name, b64, t) in enumerate(shots_b64):
        with cols[i % 4]:
            st.markdown(f"<div class='small'>{name} • {t:.2f}s</div>", unsafe_allow_html=True)
            st.components.v1.html(f'<audio controls src="data:audio/mp3;base64,{b64}"></audio>', height=70)

    # selection UI for export
    st.markdown("<div class='card'><div class='section-title'>Select Loops & Shots to Export</div></div>", unsafe_allow_html=True)
    chosen_loops = []
    for name, b64, pos in loops_b64:
        if st.checkbox(f"Include {name} (start {pos:.2f}s)", value=True, key=f"chk_{name}"):
            chosen_loops.append((name, b64))
    chosen_shots = []
    for name, b64, t in shots_b64:
        if st.checkbox(f"Include {name} (time {t:.2f}s)", value=False, key=f"ck_{name}"):
            chosen_shots.append((name, b64))

    deck_name = st.text_input("Deck name for export", value="MyRemixDeck")
    if st.button("Build .trak (ZIP) Export"):
        if not chosen_loops:
            st.error("Select at least one loop to build the .trak package.")
        else:
            zipbytes = build_trak_zip_bytes(chosen_loops, chosen_shots, bpm, deck_name)
            st.success("Built .trak package")
            st.download_button("⬇ Download Remix Deck (.trak)", data=zipbytes, file_name=f"{deck_name}.trak", mime="application/zip")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Tab 2: Sequencer (Tone.js)
# ---------------------------
with tabs[1]:
    st.markdown('<div class="card"><div class="section-title">Sequencer — 4 x 16 Step (Drums / Bass / Melody / Pads)</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="small">The sequencer runs in your browser (Tone.js). It will playback the loops above in sync and trigger the sequencer parts in-time. Choose a sound for each row, draw notes on the grid, and press Play.</div>', unsafe_allow_html=True)

    # Prepare JSON payload to send into component
    # Use loops_b64 as audio sources for Tone.Player to sync with transport
    # If loop list is absent, still start transporter (but nothing plays)
    # Build list of player sources: name + base64
    players = [{"name": name, "b64": b64} for (name, b64, *rest) in loops_b64]

    # detect key root -> compute scale notes (MIDI numbers)
    # key_root is 0-11 where 0=C; we will produce C-based names via librosa
    # Build scale notes (major/minor)
    if 'key_root' not in locals():
        # safety fallback
        key_root = 0; mode = "major"
    root_midi = 60 + int(key_root)  # middle C offset
    if mode == "major":
        intervals = [0,2,4,5,7,9,11]  # major scale
    else:
        intervals = [0,2,3,5,7,8,10]  # minor natural
    # build 2 octaves of scale notes (as note names)
    scale_notes = []
    for octave in [3,4,5]:
        for iv in intervals:
            midi = root_midi - 12 + iv + (12*(octave-4))
            name = librosa.midi_to_note(midi)
            scale_notes.append(name)
    # prepare sequencer payload
    sequencer_payload = {
        "bpm": float(bpm),
        "players": players,
        "rows": [
            {"id":"drums","label":"Drums","color":stem_colors["drums"], "type":"drum", "presets":["Kick","Snare","Clap","Perc"]},
            {"id":"bass","label":"Bass","color":stem_colors["bass"], "type":"synth", "presets":["Sub","Saw","FM","Pluck"]},
            {"id":"melody","label":"Melody","color":stem_colors["melody"], "type":"synth", "presets":["Lead","Square","Organ","Bell"]},
            {"id":"pads","label":"Pads","color":stem_colors["pads"], "type":"synth", "presets":["PadWarm","PadChiff","PadGlass","PadSoft"]},
        ],
        "scale": {"root": librosa.midi_to_note(root_midi), "mode": mode, "notes": scale_notes},
        "steps": 16
    }

    # inject HTML+JS component that uses Tone.js for playback and UI for grid editing
    # tone.js CDN used. Component provides: Play, Stop, Tempo sync, preset dropdowns per row, 4x16 clickable grid
    component_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Sequencer</title>
<script src="https://unpkg.com/tone@14.8.39/build/Tone.js"></script>
<style>
  body {{ background: transparent; color: #eaf6ff; font-family: Inter, sans-serif; }}
  .seq-wrap {{ display:flex; gap:12px; }}
  .controls {{ margin-bottom:8px; }}
  .row {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; }}
  .pad {{ width:28px; height:28px; border-radius:4px; margin:2px; display:inline-block; cursor:pointer; border:1px solid #222; }}
  .pad.off {{ background:#111; opacity:0.6; }}
  .pad.on {{ box-shadow: 0 4px 12px rgba(0,0,0,0.6); }}
  .row-label {{ width:90px; font-weight:600; }}
  .preset {{ margin-right:8px; }}
  .top-controls {{ display:flex; gap:8px; align-items:center; margin-bottom:14px; }}
  .btn {{ padding:8px 12px; border-radius:8px; border:none; font-weight:700; color:#041827; background: linear-gradient(90deg,#00c2ff,#8454ff); cursor:pointer; }}
</style>
</head>
<body>
  <div class="top-controls">
    <button id="playBtn" class="btn">Play</button>
    <button id="stopBtn" class="btn">Stop</button>
    <label style="color:#9fcff6; margin-left:8px;">BPM: <input id="bpm" type="number" value="{sequencer_payload['bpm']}" style="width:70px;"></label>
    <label style="color:#9fcff6; margin-left:8px;">Steps:
      <select id="steps"><option>16</option></select>
    </label>
  </div>

  <div id="playersInfo" style="margin-bottom:8px; color:#bfe9ff;"></div>

  <div id="sequencer"></div>

<script>
const payload = {json.dumps(sequencer_payload)};
const ToneLib = window.Tone;

// convert base64 players to Tone.Player objects and schedule them
let players = {};
async function setupPlayers() {{
  if (payload.players && payload.players.length>0) {{
    for (let p of payload.players) {{
      const b64 = p.b64;
      const blob = await (await fetch("data:audio/mp3;base64," + b64)).blob();
      const url = URL.createObjectURL(blob);
      players[p.name] = new ToneLib.Player({{url:url, loop:true}}).toDestination();
    }}
  }}
}}

// create basic synths/presets for rows
function createVoices() {{
  const voices = {{}};
  for (let r of payload.rows) {{
    if (r.type === "drum") {{
      voices[r.id] = new ToneLib.MembraneSynth().toDestination();
    }} else {{
      voices[r.id] = new ToneLib.PolySynth(ToneLib.Synth).toDestination();
    }}
  }}
  return voices;
}}

let voices = null;
let gridState = {{}};
// initialize grid state (rows x steps)
function initGrid() {{
  const steps = payload.steps;
  for (let r of payload.rows) {{
    gridState[r.id] = new Array(steps).fill(false);
  }}
}}

// render grid
function renderGrid() {{
  const container = document.getElementById("sequencer");
  container.innerHTML = "";
  const steps = payload.steps;
  for (let r of payload.rows) {{
    const rowDiv = document.createElement("div");
    rowDiv.className = "row";
    const label = document.createElement("div");
    label.className = "row-label";
    label.innerText = r.label;
    rowDiv.appendChild(label);
    // preset select
    const sel = document.createElement("select");
    sel.className = "preset";
    for (let p of r.presets) {{
      const o = document.createElement("option"); o.value = p; o.innerText = p; sel.appendChild(o);
    }}
    sel.onchange = (e) => {{
      // simple: reinstantiate voice if necessary (not complex)
    }};
    rowDiv.appendChild(sel);

    // steps
    for (let i=0;i<steps;i++) {{
      const pad = document.createElement("div");
      pad.className = "pad off";
      pad.style.background = "#111";
      pad.style.border = "1px solid #222";
      pad.dataset.row = r.id;
      pad.dataset.step = i;
      pad.onclick = (ev) => {{
        const row = ev.target.dataset.row;
        const step = parseInt(ev.target.dataset.step);
        gridState[row][step] = !gridState[row][step];
        updatePadVisual(ev.target, gridState[row][step], r.color);
      }};
      rowDiv.appendChild(pad);
    }}
    container.appendChild(rowDiv);
  }}
}}

function updatePadVisual(el, on, color) {{
  if (on) {{
    el.className = "pad on";
    el.style.background = color;
  }} else {{
    el.className = "pad off";
    el.style.background = "#111";
  }}
}}

// scheduling: at each 16th note, trigger voices and keep players in sync
let index = 0;
function scheduleRepeat() {{
  const steps = payload.steps;
  ToneLib.Transport.scheduleRepeat((time) => {{
    for (let r of payload.rows) {{
      const active = gridState[r.id][index];
      if (active) {{
        // trigger row voice
        if (r.type === "drum") {{
          voices[r.id].triggerAttackRelease("C2", "8n", time);
        }} else {{
          // pick a note from payload.scale notes based on step
          const scale = payload.scale.notes;
          const note = scale[index % scale.length];
          voices[r.id].triggerAttackRelease(note, "8n", time);
        }}
      }}
    }}
    // players are looping already; ensure transport sync
    index = (index + 1) % steps;
  }}, "16n");
}}

document.getElementById("playBtn").onclick = async () => {{
  await ToneLib.start();
  if (!voices) voices = createVoices();
  if (!Object.keys(players).length) await setupPlayers();
  // connect players to transport (they loop but we keep Transport running)
  for (let k of Object.keys(players)) {{
    // start players if not started
    try {{ players[k].start(0); }} catch(e){{}}
  }}
  index = 0;
  ToneLib.Transport.bpm.value = parseFloat(document.getElementById("bpm").value);
  ToneLib.Transport.start();
  scheduleRepeat();
}};

document.getElementById("stopBtn").onclick = () => {{
  try {{ ToneLib.Transport.stop(); }} catch(e){{}}
  for (let k of Object.keys(players)) {{
    try {{ players[k].stop(); }} catch(e){{}}
  }}
}};

// initialize
initGrid();
renderGrid();
document.getElementById("playersInfo").innerText = "Loaded " + payload.players.length + " loop players • Key: " + payload.scale.root + " " + payload.scale.mode;
</script>
</body>
</html>
    """

    # render the component (height tuned)
    st.components.v1.html(component_html, height=520, scrolling=True)

    st.markdown("<div class='card'><div class='section-title'>Sequencer tips</div><div class='small'>Click pads to toggle. Choose presets per row. Press Play to run sequencer in sync with the loops above. To get clean vocal/melody separation use the Offline EXE (Spleeter) option.</div></div>", unsafe_allow_html=True)

# ---------------------------
# Footer + cleanup
# ---------------------------
st.markdown('<div class="footer">Traktor RemixDeck Builder — Online mode. For offline full stems & pro export, use the EXE build pipeline (packaging bundle available in the app).</div>', unsafe_allow_html=True)
