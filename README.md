# Multimodal Attendance & Engagement Agent

An AI-powered, privacy-preserving classroom intelligence system integrating **computer vision**, **audio signal processing**, and **agent-based reasoning** to automatically track attendance and engagement — running entirely offline.

---

## Verified Working Setup (Windows / Python 3.12)

### Step 1 — Install Required Packages

Open a terminal in the project folder and run:

```bash
pip install flask flask-socketio pyyaml numpy pandas matplotlib seaborn reportlab Pillow requests opencv-python mediapipe scipy
```

> This is the **minimal install** that gets the dashboard running immediately.

### Step 2 — Run the Dashboard

```bash
python main.py --dashboard
```

The server starts at **http://127.0.0.1:5000** and your browser opens automatically.

---

## All Available Commands

| Command | What it does |
|---|---|
| `python main.py --dashboard` | Launch full web dashboard (recommended) |
| `python main.py --mode live` | Live webcam session (OpenCV window) |
| `python main.py --mode file --input video.mp4` | Process a recorded video |
| `python main.py --enroll "Student Name"` | Enroll a student via webcam |
| `python main.py --list-students` | List enrolled students |
| `python main.py --report` | Generate HTML report from last session |
| `python main.py --dashboard --port 8080` | Run on a different port |
| `python main.py --dashboard --verbose` | Debug logging |

---

## Enrolling Students (Face Recognition)

### Option A — Via CLI (webcam)
```bash
python main.py --enroll "John Doe"
```
Press SPACE to capture 5 photos, Q to quit.

### Option B — Manual (fastest)
Create a folder for the student and add their photos:
```
data/known_faces/
├── John_Doe/
│   ├── john_1.jpg
│   └── john_2.jpg
```
Images are automatically loaded on next startup.

---

## Optional Enhancements

### Better AI Teaching Suggestions (Ollama)
1. Install [Ollama](https://ollama.ai)
2. Run: `ollama pull llama3.2`
3. The system auto-detects it on startup

### Microphone / Audio Analysis (PyAudio)
```bash
# Windows — use pipwin for easiest install:
pip install pipwin
pipwin install pyaudio
```
Audio is automatically disabled if PyAudio is not installed.

---

## Feature Status

| Feature | Status |
|---|---|
| Web dashboard | Working |
| Live video feed | Working (webcam) |
| Engagement gauge & timeline | Working |
| Attendance tracking | Working |
| Face detection (Haar cascade) | Working (built-in OpenCV) |
| Head pose estimation | Working (MediaPipe) |
| Motion detection | Working |
| AI teaching suggestions | Working (rule-based; Ollama if installed) |
| Audio analysis | Requires PyAudio |
| HTML/PDF session reports | Working |
| Face recognition | Working (LBPH; enroll students first) |

---

## Project Structure

```
multimodal-attendance/
├── main.py                        ← ENTRY POINT: python main.py --dashboard
├── requirements.txt
├── README.md
├── config/settings.yaml           ← All tunable settings
│
├── modules/
│   ├── vision/
│   │   ├── face_detector.py       ← OpenCV Haar + LBPH recognition
│   │   ├── head_pose_estimator.py ← MediaPipe / eye cascade fallback
│   │   ├── movement_tracker.py    ← Background subtraction
│   │   └── attendance_marker.py   ← Sustained presence logic
│   ├── audio/
│   │   ├── audio_capture.py       ← Mic input (optional)
│   │   ├── voice_activity.py      ← Speech detection
│   │   └── audio_analyzer.py      ← Audio engagement score
│   ├── analytics/
│   │   ├── engagement_engine.py   ← Multimodal fusion (0-100 score)
│   │   ├── recommendation.py      ← Ollama + rule-based suggestions
│   │   └── session_analytics.py   ← Charts + HTML/PDF reports
│   └── agent/
│       ├── orchestrator.py        ← Perceive → Reason → Act loop
│       └── memory.py              ← Session state
│
├── ui/
│   ├── dashboard/
│   │   ├── app.py                 ← Flask server + MJPEG streaming
│   │   └── templates/index.html  ← Web dashboard UI
│   └── cli.py                     ← (legacy CLI helpers)
│
└── data/
    ├── known_faces/               ← Student enrollment images go here
    ├── logs/                      ← JSON attendance logs
    └── reports/                   ← Generated HTML/PDF reports
```

---

## Engagement Score Formula

```
engagement (0-100) = 50% × attention_score  (head pose)
                   + 30% × activity_score   (body movement)
                   + 20% × audio_score      (speech participation)
```

Weights are configurable in `config/settings.yaml`.

---

## Privacy

- All processing runs **locally** — no data is sent anywhere
- No video or audio is recorded to disk
- Unknown faces can be tracked anonymously or excluded (config)

---

*Skills: Artificial Intelligence · Machine Learning · Python*
