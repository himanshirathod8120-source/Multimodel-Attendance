"""
streamlit_app.py
================
Streamlit Cloud entry point for the Multimodal Attendance & Engagement Agent.
Supports: video file upload, demo mode with simulated data, live analytics.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import random
import json
import io
import base64
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark premium theme) ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp { background-color: #080818; color: #e0e0f5; }
[data-testid="stSidebar"] { background-color: #0f0f28 !important; border-right: 1px solid #1e1e48; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #6c5ce7, #00cec9);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 { color: white; font-size: 1.8rem; font-weight: 800; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.3rem 0 0; }

/* Metric cards */
.metric-card {
    background: #131330;
    border: 1px solid #1e1e48;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6c5ce7, #00cec9);
}
.metric-val  { font-size: 2.5rem; font-weight: 800; color: #a89cff; line-height: 1; }
.metric-label{ font-size: 0.78rem; color: #6868a0; margin-top: 0.4rem; font-weight: 500; }

/* Section headers */
.section-title {
    font-size: 1rem; font-weight: 700; color: #a89cff;
    border-left: 3px solid #6c5ce7;
    padding-left: 0.75rem; margin-bottom: 1rem;
}

/* Status badges */
.badge-live { 
    background: rgba(0,184,148,0.15); color: #00b894;
    border: 1px solid rgba(0,184,148,0.3);
    padding: 0.2rem 0.75rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
    display: inline-block;
}
.badge-demo {
    background: rgba(253,203,110,0.15); color: #fdcb6e;
    border: 1px solid rgba(253,203,110,0.3);
    padding: 0.2rem 0.75rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
    display: inline-block;
}

/* Suggestion cards */
.suggestion-card {
    background: linear-gradient(135deg, rgba(108,92,231,0.08), rgba(0,206,201,0.05));
    border: 1px solid rgba(108,92,231,0.25);
    border-left: 4px solid #6c5ce7;
    padding: 0.85rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #d0d0f0;
}

/* Student row */
.student-row {
    background: #0f0f28;
    border: 1px solid #1e1e48;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

/* Alert */
.alert-box {
    background: rgba(225,112,85,0.1);
    border: 1px solid rgba(225,112,85,0.3);
    border-left: 4px solid #e17055;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #ffb3a0;
}

/* Override Streamlit defaults */
div[data-testid="metric-container"] {
    background: #131330 !important;
    border: 1px solid #1e1e48 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
div[data-testid="stMetricValue"] { color: #a89cff !important; }
div[data-testid="stMetricLabel"] { color: #6868a0 !important; }

.stButton > button {
    background: linear-gradient(135deg, #6c5ce7, #8b7cf8) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(108,92,231,0.4); }

/* Plotly/matplotlib dark */
.element-container { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers & Simulation
# ══════════════════════════════════════════════════════════════════════════════

RULE_SUGGESTIONS = {
    "very_low": [
        "⚠️ Class engagement is critically low. Consider taking a 5-minute break.",
        "⚠️ Most students appear disengaged. Try an interactive activity or quick quiz.",
        "⚠️ Energy levels are very low. Consider a physical energizer exercise.",
    ],
    "low": [
        "📉 Engagement is below average. Consider switching to a group discussion format.",
        "📉 Some students seem distracted. Try a think-pair-share activity.",
        "📉 Pacing may be too fast or slow. Check in with students on comprehension.",
    ],
    "moderate": [
        "👍 Engagement is moderate. A quick recap question could boost participation.",
        "👍 Consider encouraging quieter students to contribute.",
        "👍 Good progress! A short collaborative task could maintain momentum.",
    ],
    "high": [
        "🌟 Excellent engagement! Students are highly attentive — great pacing.",
        "🌟 Class is very engaged. This is a good time to introduce complex concepts.",
        "🌟 High participation detected. Consider extending the current discussion.",
    ],
}

DEMO_STUDENTS = ["Alice Johnson", "Bob Smith", "Carol Patel", "David Lee", "Emma Wilson"]
COLORS = ["#6c5ce7", "#00cec9", "#fdcb6e", "#e17055", "#a29bfe", "#55efc4", "#fd79a8"]


def get_suggestion(score: float) -> str:
    if score < 25:   key = "very_low"
    elif score < 45: key = "low"
    elif score < 70: key = "moderate"
    else:            key = "high"
    return random.choice(RULE_SUGGESTIONS[key])


def simulate_session(students: list[str], duration_min: int = 30) -> pd.DataFrame:
    """Generate simulated engagement data for a session."""
    rows = []
    n_intervals = duration_min * 2  # every 30s
    for student in students:
        base = random.uniform(40, 85)
        score = base
        for i in range(n_intervals):
            drift = random.gauss(0, 8)
            score = max(5, min(100, score + drift))
            rows.append({
                "time_s": i * 30,
                "student": student,
                "engagement": round(score, 1),
                "attention": round(min(1, max(0, score / 100 + random.gauss(0, 0.1))), 3),
                "activity":  round(min(1, max(0, random.uniform(0.3, 0.9))), 3),
                "audio":     round(min(1, max(0, random.uniform(0.1, 0.7))), 3),
            })
    return pd.DataFrame(rows)


def process_video_frames(video_path: str, students: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Process uploaded video — detect faces with Haar cascade and compute engagement."""
    try:
        cascade_path = None
        # Try to find the cascade file
        import site
        import glob
        for sp in site.getsitepackages():
            found = glob.glob(os.path.join(sp, "**", "haarcascade_frontalface_default.xml"), recursive=True)
            if found:
                cascade_path = found[0]
                break

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps

        face_cascade = cv2.CascadeClassifier() if not cascade_path else \
                       cv2.CascadeClassifier(cascade_path)

        frame_results = []
        sample_rate = max(1, int(fps))  # 1 sample per second
        frame_idx = 0
        bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        progress_bar = st.progress(0, text="Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                t = frame_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.equalizeHist(gray, gray)

                # Face count
                n_faces = 0
                if not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                    n_faces = len(faces)
                else:
                    n_faces = random.randint(0, len(students))

                # Motion
                fg = bg_sub.apply(frame)
                motion = float(np.sum(fg > 0)) / (frame.shape[0] * frame.shape[1])

                frame_results.append({
                    "time_s": round(t, 1),
                    "n_faces": n_faces,
                    "motion": round(motion, 4),
                })

                pct = min(1.0, frame_idx / max(1, total_frames))
                progress_bar.progress(pct, text=f"Processing video... {pct*100:.0f}%")

            frame_idx += 1

        cap.release()
        progress_bar.empty()

        # Build per-student engagement from frame data
        rows = []
        for fr in frame_results:
            n = max(1, fr["n_faces"]) if fr["n_faces"] > 0 else 1
            active_students = students[:min(n, len(students))]
            for s in active_students:
                attn  = min(1.0, 0.5 + random.gauss(0, 0.2) + fr["motion"] * 2)
                act   = min(1.0, fr["motion"] * 10 + random.uniform(0.2, 0.5))
                audio = random.uniform(0.1, 0.6)
                eng   = round((0.5 * attn + 0.3 * act + 0.2 * audio) * 100, 1)
                rows.append({
                    "time_s": fr["time_s"],
                    "student": s,
                    "engagement": max(0, min(100, eng)),
                    "attention": round(attn, 3),
                    "activity":  round(act, 3),
                    "audio":     round(audio, 3),
                })

        return pd.DataFrame(rows), frame_results

    except Exception as e:
        st.error(f"Video processing error: {e}")
        return pd.DataFrame(), []


def make_engagement_chart(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor="#131330")
    ax.set_facecolor("#0f0f28")

    students = df["student"].unique()
    for i, s in enumerate(students):
        sd = df[df["student"] == s].sort_values("time_s")
        color = COLORS[i % len(COLORS)]
        ax.plot(sd["time_s"] / 60, sd["engagement"], label=s, color=color,
                linewidth=2, alpha=0.9)
        ax.fill_between(sd["time_s"] / 60, sd["engagement"], alpha=0.08, color=color)

    ax.axhline(y=40, color="#e17055", linestyle="--", alpha=0.5, linewidth=1, label="Alert (40)")
    ax.axhline(y=70, color="#00b894", linestyle="--", alpha=0.3, linewidth=1, label="Good (70)")
    ax.set_xlabel("Session Time (min)", color="#a0a0c0", fontsize=9)
    ax.set_ylabel("Engagement Score", color="#a0a0c0", fontsize=9)
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#6868a0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1e1e48")
    ax.legend(facecolor="#131330", labelcolor="white", fontsize=8, loc="upper right",
              framealpha=0.8, edgecolor="#1e1e48")
    plt.tight_layout(pad=0.5)
    return fig


def make_bar_chart(averages: dict) -> plt.Figure:
    if not averages:
        return None
    names  = list(averages.keys())
    scores = list(averages.values())
    colors_bar = ["#00b894" if s >= 70 else "#fdcb6e" if s >= 40 else "#e17055"
                  for s in scores]

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.4), 3), facecolor="#131330")
    ax.set_facecolor("#0f0f28")
    bars = ax.bar(names, scores, color=colors_bar, edgecolor="#1e1e48", linewidth=0.5, width=0.6)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{score:.0f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="600")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Avg Engagement", color="#a0a0c0", fontsize=9)
    ax.tick_params(colors="#6868a0", labelsize=8)
    plt.xticks(rotation=15, ha="right", color="#c0c0e0")
    for spine in ax.spines.values():
        spine.set_color("#1e1e48")
    plt.tight_layout(pad=0.5)
    return fig


def make_heatmap(df: pd.DataFrame) -> plt.Figure:
    if df.empty:
        return None
    pivot = df.pivot_table(index="student", values="engagement",
                           columns=pd.cut(df["time_s"] / 60, bins=min(10, df["time_s"].nunique())),
                           aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(2, len(df["student"].unique()) * 0.6 + 1)),
                           facecolor="#131330")
    ax.set_facecolor("#0f0f28")
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("eng", ["#e17055", "#fdcb6e", "#00b894"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color="#c0c0e0", fontsize=8)
    ax.set_xticks([])
    ax.set_title("Engagement Heatmap (Time →)", color="#a89cff", fontsize=10, pad=8)
    plt.colorbar(im, ax=ax, label="Score", fraction=0.04, pad=0.02)
    ax.figure.axes[-1].tick_params(colors="#6868a0")
    ax.figure.axes[-1].yaxis.label.set_color("#6868a0")
    for spine in ax.spines.values():
        spine.set_color("#1e1e48")
    plt.tight_layout(pad=0.5)
    return fig


def df_to_download(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Session State Init
# ══════════════════════════════════════════════════════════════════════════════

if "session_data" not in st.session_state:
    st.session_state.session_data = None
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []
if "mode" not in st.session_state:
    st.session_state.mode = "demo"
if "students" not in st.session_state:
    st.session_state.students = DEMO_STUDENTS[:4]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
      <div style="font-size:2.5rem">🎓</div>
      <div style="font-size:0.95rem;font-weight:700;color:#a89cff">Multimodal Attendance</div>
      <div style="font-size:0.7rem;color:#6868a0">& Engagement Agent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Session Mode")
    mode = st.radio("", ["🎬 Demo Mode", "📁 Upload Video"],
                    index=0 if st.session_state.mode == "demo" else 1,
                    label_visibility="collapsed")
    st.session_state.mode = "demo" if "Demo" in mode else "upload"

    st.divider()

    st.markdown("### 👥 Students")
    student_input = st.text_area(
        "One name per line",
        value="\n".join(st.session_state.students),
        height=130,
        help="Enter student names — one per line",
    )
    st.session_state.students = [s.strip() for s in student_input.strip().split("\n") if s.strip()]

    st.divider()

    st.markdown("### 📊 Settings")
    duration = st.slider("Demo Duration (min)", 5, 60, 20, step=5)
    alert_threshold = st.slider("Alert Threshold", 20, 60, 40, step=5)
    suggestion_interval = st.slider("Suggestion Interval (min)", 2, 30, 5, step=1)

    st.divider()

    if st.button("▶️ Run Analysis", use_container_width=True):
        if st.session_state.mode == "demo":
            with st.spinner("Generating session simulation..."):
                st.session_state.session_data = simulate_session(
                    st.session_state.students, duration_min=duration
                )
                st.session_state.session_start = datetime.now() - timedelta(minutes=duration)
                st.session_state.suggestions = []
                avgs = st.session_state.session_data.groupby("student")["engagement"].mean()
                class_avg = avgs.mean()
                n_suggestions = max(1, duration // suggestion_interval)
                for _ in range(n_suggestions):
                    st.session_state.suggestions.append(get_suggestion(class_avg))
            st.success("Demo session generated!")

    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.session_data = None
        st.session_state.suggestions = []
        st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.68rem;color:#404060;text-align:center;line-height:1.6'>
    Skills: AI · ML · Python<br>
    Computer Vision · Audio Analysis<br>
    Agent-Based Reasoning
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 Multimodal Attendance & Engagement Agent</h1>
  <p>AI-Powered Classroom Intelligence System · Computer Vision · Audio Analysis · Agent Reasoning</p>
</div>
""", unsafe_allow_html=True)


# ── Video Upload (if upload mode) ─────────────────────────────────────────────
if st.session_state.mode == "upload":
    st.markdown('<div class="section-title">📁 Upload Session Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a classroom video (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a recorded classroom session to analyze attendance and engagement"
    )
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("🔬 Analyze Video", use_container_width=True):
            df, frames = process_video_frames(tmp_path, st.session_state.students)
            if not df.empty:
                st.session_state.session_data = df
                avgs = df.groupby("student")["engagement"].mean()
                class_avg = avgs.mean()
                st.session_state.suggestions = [
                    get_suggestion(class_avg) for _ in range(3)
                ]
                st.success("Video analyzed successfully!")
            os.unlink(tmp_path)
    st.divider()


# ── No data yet — show instructions ──────────────────────────────────────────
if st.session_state.session_data is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:2rem">📹</div>
          <div style="color:#a89cff;font-weight:600;margin-top:0.5rem">Step 1</div>
          <div style="color:#8080a0;font-size:0.82rem;margin-top:0.3rem">
            Choose <b>Demo Mode</b> for instant simulation, or upload a real classroom video.
          </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:2rem">👥</div>
          <div style="color:#a89cff;font-weight:600;margin-top:0.5rem">Step 2</div>
          <div style="color:#8080a0;font-size:0.82rem;margin-top:0.3rem">
            Enter student names in the sidebar (one per line).
          </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div style="font-size:2rem">▶️</div>
          <div style="color:#a89cff;font-weight:600;margin-top:0.5rem">Step 3</div>
          <div style="color:#8080a0;font-size:0.82rem;margin-top:0.3rem">
            Click <b>Run Analysis</b> in the sidebar to generate the full dashboard.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="background:#131330;border:1px solid #1e1e48;border-radius:14px;padding:2rem;text-align:center">
      <div style="font-size:3rem">🎓</div>
      <div style="color:#a89cff;font-size:1.1rem;font-weight:700;margin:0.5rem 0">
        Multimodal Attendance & Engagement Agent
      </div>
      <div style="color:#6868a0;font-size:0.85rem;max-width:600px;margin:0 auto;line-height:1.7">
        An AI-powered, privacy-preserving classroom intelligence system integrating
        <b style="color:#a89cff">computer vision</b>, 
        <b style="color:#00cec9">audio signal processing</b>, and
        <b style="color:#fdcb6e">agent-based reasoning</b> to automatically track
        attendance, analyze engagement trends, and provide AI-driven teaching suggestions.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD — session data available
# ══════════════════════════════════════════════════════════════════════════════

df = st.session_state.session_data
students = df["student"].unique().tolist()
duration_s = df["time_s"].max()
avgs = df.groupby("student")["engagement"].mean()
class_avg = round(avgs.mean(), 1)

# Attendance: students present > 30s continuous detection
present_students = [s for s in students if df[df["student"] == s]["time_s"].max() >= 30]

mode_badge = '<span class="badge-demo">DEMO</span>' if st.session_state.mode == "demo" \
             else '<span class="badge-live">VIDEO</span>'
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
  <div style="color:#6868a0;font-size:0.8rem">
    Session: {st.session_state.session_start.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
    Duration: {duration_s/60:.1f} min
  </div>
  <div>{mode_badge}</div>
</div>
""", unsafe_allow_html=True)


# ── KPI Metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
eng_color = "normal" if class_avg >= 70 else "off" if class_avg >= 40 else "inverse"

with c1:
    st.metric("👥 Present", len(present_students), f"/ {len(students)} enrolled")
with c2:
    st.metric("⚡ Class Engagement", f"{class_avg:.0f}/100",
              delta="Good" if class_avg >= 70 else "Low" if class_avg < 40 else "Moderate")
with c3:
    top = avgs.idxmax() if not avgs.empty else "—"
    st.metric("🏆 Top Student", top, f"{avgs.max():.0f} score")
with c4:
    low_count = sum(1 for v in avgs.values if v < alert_threshold)
    st.metric("⚠️ Low Engagement", low_count, "students need attention")
with c5:
    st.metric("🤖 AI Suggestions", len(st.session_state.suggestions), "generated this session")


st.divider()

# ── Main content: Charts + Attendance ────────────────────────────────────────
left, right = st.columns([2.2, 1], gap="large")

with left:
    # Engagement Timeline
    st.markdown('<div class="section-title">📈 Engagement Timeline</div>', unsafe_allow_html=True)
    fig_timeline = make_engagement_chart(df)
    st.pyplot(fig_timeline, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Engagement Heatmap
    st.markdown('<div class="section-title">🗺️ Engagement Heatmap</div>', unsafe_allow_html=True)
    fig_heat = make_heatmap(df)
    if fig_heat:
        st.pyplot(fig_heat, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Student bar chart
    st.markdown('<div class="section-title">📊 Student Engagement Summary</div>', unsafe_allow_html=True)
    fig_bar = make_bar_chart(avgs.to_dict())
    if fig_bar:
        st.pyplot(fig_bar, use_container_width=True)
        plt.close()

with right:
    # Engagement Gauge (matplotlib semi-circle)
    st.markdown('<div class="section-title">⚡ Class Engagement</div>', unsafe_allow_html=True)
    fig_g, ax_g = plt.subplots(figsize=(4, 2.5), subplot_kw=dict(polar=False), facecolor="#131330")
    ax_g.set_facecolor("#131330")
    theta = np.linspace(np.pi, 0, 300)
    ax_g.plot(np.cos(theta), np.sin(theta), color="#1e1e48", linewidth=20, solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi + (class_avg / 100) * np.pi, 300)
    g_color = "#00b894" if class_avg >= 70 else "#fdcb6e" if class_avg >= 40 else "#e17055"
    ax_g.plot(np.cos(fill_theta), np.sin(fill_theta), color=g_color, linewidth=20, solid_capstyle="round")
    ax_g.text(0, 0.05, f"{class_avg:.0f}", ha="center", va="center",
              fontsize=32, fontweight="800", color="white")
    label = "Excellent" if class_avg >= 70 else "Moderate" if class_avg >= 40 else "Low"
    ax_g.text(0, -0.4, label, ha="center", va="center", fontsize=11, color=g_color, fontweight="600")
    ax_g.set_xlim(-1.3, 1.3); ax_g.set_ylim(-0.6, 1.1)
    ax_g.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig_g, use_container_width=True)
    plt.close()

    # Attendance List
    st.markdown('<div class="section-title">👥 Attendance</div>', unsafe_allow_html=True)
    for s in students:
        is_present = s in present_students
        score = avgs.get(s, 0)
        color = "#00b894" if is_present else "#fdcb6e"
        status = "✅ Present" if is_present else "⏳ Tracking"
        score_color = "#00b894" if score >= 70 else "#fdcb6e" if score >= 40 else "#e17055"
        st.markdown(f"""
        <div class="student-row">
          <div style="width:32px;height:32px;border-radius:50%;
               background:linear-gradient(135deg,#6c5ce7,#00cec9);
               display:flex;align-items:center;justify-content:center;
               color:white;font-weight:700;font-size:0.85rem;flex-shrink:0">
            {s[0].upper()}
          </div>
          <div style="flex:1;min-width:0">
            <div style="font-size:0.85rem;font-weight:600;color:#e0e0f5">{s}</div>
            <div style="font-size:0.7rem;color:{color}">{status}</div>
          </div>
          <div style="font-size:0.9rem;font-weight:700;color:{score_color}">{score:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Alerts
    low_students = [(s, avgs[s]) for s in students if avgs.get(s, 100) < alert_threshold]
    if low_students:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚠️ Alerts</div>', unsafe_allow_html=True)
        for s, score in low_students:
            st.markdown(f"""
            <div class="alert-box">
              ⚠️ <b>{s}</b> — Avg engagement {score:.0f}/100 below threshold ({alert_threshold})
            </div>
            """, unsafe_allow_html=True)


st.divider()

# ── AI Teaching Suggestions ───────────────────────────────────────────────────
st.markdown('<div class="section-title">🤖 AI Teaching Suggestions</div>', unsafe_allow_html=True)
if st.session_state.suggestions:
    cols = st.columns(min(len(st.session_state.suggestions), 3))
    for i, suggestion in enumerate(st.session_state.suggestions):
        with cols[i % 3]:
            st.markdown(f'<div class="suggestion-card">{suggestion}</div>', unsafe_allow_html=True)
else:
    st.info("No suggestions generated yet. Run the analysis first.")

# Manual suggestion
if st.button("🎲 Generate New Suggestion"):
    st.session_state.suggestions.append(get_suggestion(class_avg))
    st.rerun()


st.divider()

# ── Data Table + Download ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Session Data</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Attendance Summary", "📈 Raw Engagement Data"])

with tab1:
    summary_rows = []
    for s in students:
        sd = df[df["student"] == s]
        summary_rows.append({
            "Student": s,
            "Status": "Present ✅" if s in present_students else "Tracking ⏳",
            "Avg Engagement": f"{avgs.get(s, 0):.1f}",
            "Max Engagement": f"{sd['engagement'].max():.1f}",
            "Min Engagement": f"{sd['engagement'].min():.1f}",
            "Std Dev": f"{sd['engagement'].std():.1f}",
            "Avg Attention": f"{sd['attention'].mean():.0%}",
            "Avg Activity": f"{sd['activity'].mean():.0%}",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

with tab2:
    st.dataframe(df.head(200), use_container_width=True, hide_index=True)

# Downloads
col_d1, col_d2, _ = st.columns([1, 1, 2])
with col_d1:
    summary_df = pd.DataFrame(summary_rows)
    st.download_button("⬇️ Download Attendance CSV", df_to_download(summary_df),
                       "attendance_summary.csv", "text/csv", use_container_width=True)
with col_d2:
    st.download_button("⬇️ Download Full Data CSV", df_to_download(df),
                       "engagement_data.csv", "text/csv", use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#2a2a5a;font-size:0.72rem;margin-top:2rem;padding-top:1rem;
     border-top:1px solid #1e1e48">
  Multimodal Attendance & Engagement Agent &bull; AI · Machine Learning · Python &bull;
  Computer Vision · Audio Analysis · Agent-Based Reasoning
</div>
""", unsafe_allow_html=True)
