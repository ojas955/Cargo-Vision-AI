# 🛡️ CIIBS — Real-Time Intelligent Cargo X-ray Risk Screening

**Vision + LLM Copilot | Hackathon Prototype**

A real-time AI system for automated X-ray cargo screening that detects prohibited items, generates anomaly heatmaps, computes risk scores, and provides LLM-powered explanations for customs officers.

---

## 🏗️ Architecture

```
X-ray Image/Frame
       │
       ├──→ [YOLOv8 Detector] ──→ Prohibited Item Detections (boxes, class, confidence)
       │
       ├──→ [Anomaly Engine] ──→ Suspicion Heatmap + Anomaly Score
       │
       ├──→ [Declaration Check] ──→ Mismatch Penalty
       │
       └──→ [Risk Fusion] ──→ Risk Score (0-100) + Level
                │
                └──→ [LLM Copilot] ──→ Explainable Assessment + Action Hint
```

### Key Components

| Module | Description |
|--------|-------------|
| `inference.py` | YOLOv8 detection, anomaly heatmap, mismatch analysis, risk fusion |
| `llm_assistant.py` | Async OpenAI/Gemini API + template fallback |
| `app.py` | Flask server with REST API + WebSocket for live video |
| `config.yaml` | All thresholds, weights, mode settings, LLM config |

### Detection Classes (PIDray 12)
Baton, Pliers, Hammer, Powerbank, Scissors, Wrench, Gun, Bullet, Sprayer, HandCuffs, Knife, Lighter

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ciibs
pip install -r requirements_demo.txt
```

### 2. (Optional) Set LLM API Key
```bash
# For OpenAI
export OPENAI_API_KEY="sk-..."

# Or for Gemini
export GEMINI_API_KEY="..."
```
> System works fully without API key — uses template-based explanations.

### 3. Run
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 📋 Usage

### Image Mode
1. Drag and drop or click to upload an X-ray image
2. (Optional) Enter cargo declaration text
3. Toggle **Accuracy** or **Realtime** mode
4. Toggle **LLM Explanations** on/off
5. Click **Analyze Scan**

### Live Camera Mode
1. Switch to **Live** tab
2. Click **Start** to activate webcam
3. Real-time analysis runs automatically with FPS overlay

---

## ⚙️ Configuration

Edit `config.yaml` to adjust:
- Detection thresholds and model sizes
- Risk fusion weights (`detection: 0.5, anomaly: 0.3, mismatch: 0.2`)
- Anomaly sensitivity parameters
- LLM provider, model, and timeout

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements_demo.txt` |
| Model download slow | First run downloads YOLOv8 weights (~6MB for nano). Use fast internet. |
| Webcam not working | Check browser permissions. HTTPS may be required for remote access. |
| LLM timeout | Set `OPENAI_API_KEY` env var or disable LLM toggle. Template fallback always works. |
| Port in use | `PORT=5001 python app.py` |

---

## 📊 Metrics

- **FPS**: 5-15 FPS (realtime mode, CPU) / 2-5 FPS (accuracy mode)
- **Inference latency**: ~50-200ms per frame
- **Risk levels**: LOW (<35) → MEDIUM (35-69) → HIGH (70-79) → CRITICAL (≥80)

---

*Built for TESSERACT'26 Hackathon — CIIBS Problem Statement*
