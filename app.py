"""
CIIBS — Real-Time Intelligent Cargo X-ray Risk Screening
Flask Web Application with Gemini Vision Dual-Brain Analysis
"""

import base64
import logging
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from inference import load_model, run_full_pipeline
from llm_assistant import generate_explanation_sync, analyze_vision_sync

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("ciibs.app")

app = Flask(__name__)
app.config["SECRET_KEY"] = "ciibs-xray-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

models = {}
scan_history = deque(maxlen=50)  # Last 50 scans


def preload_models():
    global models
    try:
        logger.info("Pre-loading realtime model...")
        models["realtime"] = load_model("realtime")
        logger.info("Realtime model ready")
    except Exception as e:
        logger.error(f"Failed to load realtime model: {e}")
    try:
        logger.info("Pre-loading accuracy model...")
        models["accuracy"] = load_model("accuracy")
        logger.info("Accuracy model ready")
    except Exception as e:
        logger.warning(f"Accuracy model not loaded: {e}")


def numpy_to_base64(img: np.ndarray, fmt: str = ".jpg") -> str:
    _, buffer = cv2.imencode(fmt, img)
    return base64.b64encode(buffer).decode("utf-8")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "models_loaded": list(models.keys()),
        "version": "2.0.0-dual-brain",
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY", "")),
        "scan_count": len(scan_history),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    t_start = time.time()

    mode = request.form.get("mode", "realtime")
    declared_cargo = request.form.get("declared_cargo", "")
    use_llm = request.form.get("use_llm", "false").lower() == "true"

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

        model = models.get(mode)
        if model is None:
            model = load_model(mode)
            models[mode] = model

        # Run core pipeline
        result = run_full_pipeline(
            image, mode=mode,
            declared_cargo=declared_cargo if declared_cargo else None,
            model=model
        )

        # Gemini Vision — Dual Brain Analysis ⭐
        vlm_result = {"vlm_analysis": "Vision analysis disabled", "source": "disabled", "latency_ms": 0}
        if use_llm:
            try:
                vlm_result = analyze_vision_sync(image)
            except Exception as e:
                logger.warning(f"Vision analysis failed: {e}")
                err_msg = str(e)
                vlm_result = {"vlm_analysis": f"Vision unavailable: {err_msg[:100]}", "source": "error", "latency_ms": 0}

        # LLM text explanation
        if use_llm:
            llm_result = generate_explanation_sync(result)
            result["explanation"] = llm_result["explanation"]
            result["explanation_source"] = llm_result["source"]
            result["llm_latency_ms"] = llm_result["latency_ms"]
        else:
            result["explanation_source"] = "template"
            result["llm_latency_ms"] = 0

        # Build response
        response = {
            "detections": result["detections"],
            "detection_count": result["detection_count"],
            "annotated_image": numpy_to_base64(result["annotated_image"]),
            "heatmap_overlay": numpy_to_base64(result["heatmap_overlay"]),
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "risk_breakdown": result["risk_breakdown"],
            "anomaly_score": result["anomaly_score"],
            "mismatch_info": result["mismatch_info"],
            "material_info": result.get("material_info", {}),
            "concealment_score": result.get("concealment_score", 0),
            "concealment_info": result.get("concealment_info", {}),
            "explanation": result["explanation"],
            "explanation_source": result.get("explanation_source", "template"),
            "vlm_analysis": vlm_result.get("vlm_analysis", ""),
            "vlm_source": vlm_result.get("source", "disabled"),
            "vlm_latency_ms": vlm_result.get("latency_ms", 0),
            "mode": result["mode"],
            "inference_time_ms": result["inference_time_ms"],
            "total_time_ms": round((time.time() - t_start) * 1000, 1),
            "fps": result["fps"],
            "llm_latency_ms": result.get("llm_latency_ms", 0),
        }

        # Save to scan history
        scan_history.appendleft({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "detection_count": result["detection_count"],
            "anomaly_score": result["anomaly_score"],
            "concealment_score": result.get("concealment_score", 0),
            "mode": mode,
        })

        return jsonify(response)

    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def get_history():
    return jsonify(list(scan_history))


# ──────────────────────────────────────────────
# WebSocket: Live Video
# ──────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info("Client connected")
    emit("status", {"message": "Connected to CIIBS server"})


@socketio.on("disconnect")
def on_disconnect():
    logger.info("Client disconnected")


@socketio.on("analyze_frame")
def handle_frame(data):
    try:
        mode = data.get("mode", "realtime")
        declared_cargo = data.get("declared_cargo", "")

        img_data = data["frame"].split(",")[1] if "," in data["frame"] else data["frame"]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            emit("frame_result", {"error": "Could not decode frame"})
            return

        model = models.get(mode)
        if model is None:
            model = load_model(mode)
            models[mode] = model

        result = run_full_pipeline(
            image, mode=mode,
            declared_cargo=declared_cargo if declared_cargo else None,
            model=model
        )

        emit("frame_result", {
            "annotated_image": numpy_to_base64(result["annotated_image"]),
            "heatmap_overlay": numpy_to_base64(result["heatmap_overlay"]),
            "detections": result["detections"],
            "detection_count": result["detection_count"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "anomaly_score": result["anomaly_score"],
            "concealment_score": result.get("concealment_score", 0),
            "explanation": result["explanation"],
            "fps": result["fps"],
            "inference_time_ms": result["inference_time_ms"],
        })

    except Exception as e:
        logger.exception("Frame analysis failed")
        emit("frame_result", {"error": str(e)})


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    preload_models()
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting CIIBS server on port {port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
